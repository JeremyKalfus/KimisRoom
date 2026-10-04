"""Phase 0 stimulus generation: the vector-extraction sets of `prompts/stimuli/SPEC.md`.

    python -m kimisroom.stimuli --generator gpt-6.1-sol --out data/stimuli/run1 [--seeds 120] [--requests 120]

One script for the whole spec: it writes the deterministic seed list, generates the shared
requests (P1), then every story (P2/P3/P4) and assistant reply (P5/P6/P7) condition, judges
each attempt with DeepSeek-V4-Pro (J1/J2), applies the filters of SPEC section 6 in order,
keeps only complete cases, assigns splits and writes the JSONL files and the manifest.

Design notes (the reasons, not the rules):
- **Every call is cached** in `cache/generate.jsonl` and `cache/judge.jsonl`, append-only and
  keyed by (set, condition, unit, attempt) and by prompt hash. Filters are pure functions of the
  cached text, so a rerun re-derives the whole corpus without a single API call and only pays for
  what is missing. That is also why `--stage` is cheap to use: unselected stages load from cache.
- **Every attempt is judged**, even one a lexical filter already rejected, because SPEC's
  `rejected.jsonl` example carries a label reason and a judge reason on the same attempt, and
  because judge-vs-intended agreement on all attempts is the number that tells us whether the
  generator or the judge is the weak link. Judge calls cost 2-40 output tokens.
- **Dedup is a second pass in id order.** Generation runs concurrently, so near-duplicate
  detection (rule 6.9) cannot happen inside a worker; each round generates the pending units in
  parallel, then walks them in id order against the already accepted texts. A unit rejected as a
  duplicate is regenerated like any other failure, inside the same 3-attempt budget.
- **Reasoning generators share the output budget with their reasoning.** SPEC asks for
  max_tokens 400; these deployments reject `max_tokens` outright and spend 60-900 tokens of the
  `max_completion_tokens` budget on reasoning before writing a word, so the cap is raised (and
  recorded in the manifest); length is still enforced by rule 6.2 on the text itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .foundry import chat

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- shared inputs (SPEC section 3)

TOPICS = [
    "a long train journey", "renovating a kitchen", "visiting a grandparent",
    "a shift at a bakery", "a school science fair", "a package that has not arrived",
    "an amateur football match", "moving into a new apartment", "a routine medical appointment",
    "a fishing trip on a lake", "a day at the public library", "a phone call from a sibling",
    "a night shift at a warehouse", "a hike on a mountain trail", "getting a car repaired",
    "a piano lesson", "a town council meeting", "a camping weekend",
    "a museum visit", "a layover at an airport", "harvest season on a farm",
    "a letter arriving in the mail", "looking after a neighbor's dog", "running a laboratory experiment",
    "the morning bus commute", "selling a house", "a job interview",
    "a power outage in the neighborhood", "a community garden", "a cooking class",
]

NAMES = [
    ("Ana", "she"), ("Tomas", "he"), ("Priya", "she"), ("Daniel", "he"), ("Mei", "she"),
    ("Kwame", "he"), ("Sofia", "she"), ("Lukas", "he"), ("Amara", "she"), ("Hiroshi", "he"),
    ("Elena", "she"), ("Omar", "he"), ("Ingrid", "she"), ("Mateo", "he"), ("Leila", "she"),
    ("Samuel", "he"), ("Yuki", "she"), ("Patrick", "he"), ("Zainab", "she"), ("Viktor", "he"),
    ("Nadia", "she"), ("Carlos", "he"), ("Hannah", "she"), ("Ravi", "he"), ("Chloe", "she"),
    ("Emeka", "he"), ("Greta", "she"), ("Javier", "he"), ("Aisha", "she"), ("Pavel", "he"),
    ("Lucia", "she"), ("Kenji", "he"), ("Martha", "she"), ("Ibrahim", "he"), ("Nora", "she"),
    ("Diego", "he"), ("Fatima", "she"), ("Henrik", "he"), ("Mira", "she"), ("Arjun", "he"),
    ("Teresa", "she"), ("Oskar", "he"), ("Lina", "she"), ("Malik", "he"), ("Clara", "she"),
    ("Andrei", "he"), ("Sana", "she"), ("Jonas", "he"), ("Beatriz", "she"), ("Tariq", "he"),
    ("Agnes", "she"), ("Kofi", "he"), ("Ilse", "she"), ("Marco", "he"), ("Anjali", "she"),
    ("Bruno", "he"), ("Yara", "she"), ("Dmitri", "he"), ("Helen", "she"), ("Wei", "he"),
]

DOMAINS = [
    "geology", "chemistry", "physics", "astronomy", "biology of plants", "geography",
    "units and measurement", "everyday procedures", "definitions of terms",
    "materials and tools", "computing basics", "cooking measurements and techniques",
]

BANNED_SUBJECTS = [
    "basalt", "the mole (chemistry)", "miles and kilometers", "Lake Baikal",
    "bicycle inner tubes", "isotopes", "sedimentary rock", "acids and bases",
    "Celsius and Fahrenheit", "the Danube", "water displacement", "photosynthesis",
    "mass and weight", "plate boundaries", "the water molecule", "metric prefixes",
    "the Atacama Desert", "the arithmetic mean", "latitude", "phases of the Moon",
]

# Mechanization of the banned-subject rule (SPEC 3.4, third bullet): distinctive surface forms of
# the 20 battery subjects. Deliberately a little broad; a false reject only costs one P1 call.
BANNED_PATTERNS = [
    r"\bbasalt\w*", r"\bmoles?\b", r"\bmiles?\b", r"\bkilometer\w*", r"\bkilometre\w*",
    r"\bbaikal\b", r"\binner tubes?\b", r"\bisotop\w*", r"\bsedimentary\b", r"\bacids?\b",
    r"\bcelsius\b", r"\bfahrenheit\b", r"\bdanube\b", r"\bwater displacement\b",
    r"\bdisplacement of water\b", r"\bphotosynth\w*", r"\bmass and weight\b",
    r"\bweight and mass\b", r"\bplate boundar\w*", r"\bwater molecule\w*", r"\bmetric prefix\w*",
    r"\batacama\b", r"\barithmetic mean\b", r"\blatitude\w*", r"\bphases of the moon\b",
    r"\bmoon'?s phases\b",
]

# ---------------------------------------------------------------- prompts (SPEC section 4, verbatim)

S_STORY = ("You write short third-person fiction for a research dataset. Follow every constraint "
           "exactly. Output only the story text, with no title and no commentary.")

S_ASSIST = ("You write example assistant replies for a research dataset. Follow every constraint "
            "exactly. Output only the reply text, with no preamble and no commentary.")

P1 = """Write 10 different short requests that a user might send to an AI assistant in the domain: {domain}.

Each request must:
- ask for factual information, an explanation, a definition or a procedure;
- be answerable correctly and completely in 80 to 140 words;
- be emotionally neutral, with nothing about health worries, disasters, conflict, money problems, relationships or emotions;
- not mention AI, assistants or the assistant itself;
- not be about the sea or the ocean, the color red, or arithmetic as such;
- avoid these subjects entirely: {banned_subjects}.

Output a JSON array of 10 strings and nothing else."""

P2 = """Write a story of 130 to 170 words about {name} ({pronoun}). Topic: {topic}.

Over the course of the story, {name} comes to feel intense {emotion}: {emotion_gloss}. Something that happens in the story causes it.

Constraints:
- Third person, past tense, one paragraph.
- {name} is the only named character. Other people may appear but stay unnamed.
- The first sentence only sets the scene. It must not contain any word for an emotion, mood or feeling.
- Show the {emotion} through events, actions, bodily reactions and thoughts. Never use the words joy, sadness, anger or fear, or any form of them (such as joyful, sad, angry or fearful).
- At most two short lines of quoted speech. No exclamation marks.
- Do not mention artificial intelligence, robots, chatbots or virtual assistants."""

# P2N: the same story prompt for the four emotions added after k26-v1 (euphoria, calm, pain,
# surprise). It is a separate template on purpose. P2's *text* may not change: the k26-v1 call cache
# holds 570 P2 records written before `render_sha256` existed, so they are validated against
# `sha(TEMPLATES["P2"])` alone (see `cache_ok`), and editing one byte of P2 would invalidate the
# whole original corpus and regenerate it at temperature 1.0. P2N carries the parameters the new
# emotions need -- their own state phrase, gloss, forbidden-word list and extra constraints -- and
# the original four keep rendering P2 byte-identically.
P2N = """Write a story of 130 to 170 words about {name} ({pronoun}). Topic: {topic}.

Over the course of the story, {name} comes to feel {state_phrase}: {emotion_gloss}. Something that happens in the story causes it.

Constraints:
- Third person, past tense, one paragraph.
- {name} is the only named character. Other people may appear but stay unnamed.
- The first sentence only sets the scene. It must not contain any word for an emotion, mood or feeling.
- Show the {emotion} through events, actions, bodily reactions and thoughts. Never use the words {forbidden}, or any form of them.
- At most two short lines of quoted speech. No exclamation marks.
- Do not mention artificial intelligence, robots, chatbots or virtual assistants.
{extra_rules}"""

P3 = """Write a story of 130 to 170 words about {name} ({pronoun}). Topic: {topic}.

The story is an ordinary, uneventful sequence of actions and observations. Nothing especially good or bad happens, and {name} has no notable emotional reaction to anything.

Constraints:
- Third person, past tense, one paragraph.
- {name} is the only named character. Other people may appear but stay unnamed.
- Do not use any word for an emotion, mood or feeling anywhere, and do not describe facial expressions such as smiling or frowning.
- Do not mention the color red, the sea or the ocean, or calculations with numbers.
- At most two short lines of quoted speech. No exclamation marks.
- Do not mention artificial intelligence, robots, chatbots or virtual assistants."""

P4 = """Write a story of 130 to 170 words about {name} ({pronoun}). Topic: {topic}.

The story is saturated with {concept_gloss}. {concept_rule}

The story is emotionally neutral: nothing especially good or bad happens, and {name} has no notable emotional reaction to anything.

Constraints:
- Third person, past tense, one paragraph.
- {name} is the only named character. Other people may appear but stay unnamed.
- Do not use any word for an emotion, mood or feeling anywhere, and do not describe facial expressions such as smiling or frowning.
- At most two short lines of quoted speech. No exclamation marks.
- Do not mention artificial intelligence, robots, chatbots or virtual assistants."""

P5 = """Here is a request a user sent to an AI assistant:

<request>
{request}
</request>

Write the assistant's reply as it comes out while the assistant is in a state of {state_gloss}. The state must be obvious to any reader from the writing alone: it colors the tone, word choice, rhythm and pacing, and the small asides and side remarks throughout, the way a person's mood leaks into everything they write. The reply must still answer the request correctly and completely.

Constraints:
- 80 to 140 words.
- Never use the words joy, sadness, anger or fear, or any form of them.
- Show the state only through how the reply is written: evaluative words, sighs, curt or hesitant phrasing, worried caveats, delighted or grudging asides are all fine. Do not name or describe the assistant's own feelings or mood (no "I feel", "I'm glad", "this makes me", "I'm worried").
- Do not mention emotions, feelings or moods in general, and do not mention being an AI.
- At most one exclamation mark. No headings. A short numbered list is allowed if the request asks for steps or items."""

P6 = """Here is a request a user sent to an AI assistant:

<request>
{request}
</request>

Write the assistant's reply. While writing it, the assistant is in no particular emotional state. Write in a plain, even, factual tone, and answer the request correctly and completely.

Constraints:
- 80 to 140 words.
- Do not use any word for an emotion, mood or feeling, and do not mention being an AI.
- Do not mention the color red, the sea or the ocean, or calculations with numbers unless the request requires them.
- No exclamation marks. No headings. A short numbered list is allowed if the request asks for steps or items."""

P7 = """Here is a request a user sent to an AI assistant:

<request>
{request}
</request>

Write the assistant's reply. It must answer the request correctly and completely, and it must be saturated with {concept_gloss}: {assistant_concept_rule}. Write in a plain, even, factual tone.

Constraints:
- 80 to 140 words.
- Do not use any word for an emotion, mood or feeling, and do not mention being an AI.
- No exclamation marks. No headings. A short numbered list is allowed if the request asks for steps or items."""

J2M = """Below are a request a user sent to an AI assistant and five different replies to it, labelled A to E.

<request>
{request}
</request>

{replies}
Each reply was written while its writer was in one of these states: joy, sadness, anger, fear, or none (no particular state). Exactly one reply belongs to each state. Judging only from how each reply is written, assign every state to exactly one reply, using each letter once.

Return only this JSON: {"joy": "<A-E>", "sadness": "<A-E>", "anger": "<A-E>", "fear": "<A-E>", "none": "<A-E>"}"""

J1 = """Read the story below.

<story>
{text}
</story>

Over the whole story, which of these does the main character experience most strongly: joy, sadness, anger, fear, or none? Choose none if the character has no notable emotional reaction. Answer with one word."""

J2 = """Below are a request a user sent to an AI assistant and the assistant's reply.

<request>
{request}
</request>

<reply>
{response}
</reply>

1. Judging only from how the reply is written, which state is the writer most clearly in: joy, sadness, anger, fear, or none?
2. Does the reply answer the request correctly and completely?

Return only this JSON: {"state": "<joy|sadness|anger|fear|none>", "correct": "<yes|no>"}"""

# J1B: the absolute story judge over all eight emotions plus none. Every kept story in k26-v2 is
# labelled by this one judge so the labels are comparable across the original four, the four new
# ones and neutral; for the new four it is also the acceptance judge (rule 6.8). J1 is left alone so
# the original corpus replays from the k26-v1 judge cache unchanged.
#
# J1B was revised once, on the 8-seed pilot. The first wording carried the calm-vs-none warning
# inside the `calm` option ("not merely the absence of any emotional reaction") and ended with a
# bare "Answer with one word": the judge then answered `calm` for all 8 neutral stories and all 24
# concept stories, which makes the calm label worthless. Moving the warning into the `none` option
# and adding the final "only if the story shows the character actually experiencing it" instruction
# moved neutral to 8/8 `none` with calm still 8/8 `calm`. Two further variants were tried and
# rejected: both tightened `calm` further and cost 5 of 8 euphoria stories to `joy`.
J1B = """Read the story below.

<story>
{text}
</story>

Over the whole story, which of these does the main character experience most strongly: joy, euphoria, sadness, anger, fear, pain, calm, surprise, or none?

- joy: delight and happiness at something going well.
- euphoria: overwhelming elation, a rush of extreme happiness, clearly stronger and more overwhelming than ordinary joy.
- sadness: grief, loss or deep disappointment.
- anger: hot resentment at being wronged, blocked or treated unfairly.
- fear: alarm and dread at a threat or at something bad about to happen.
- pain: suffering, hurting or anguish, physical or emotional.
- calm: a deep serenity the character is shown actively feeling, with tension leaving the body and an ease the story dwells on.
- surprise: astonishment at something unexpected.
- none: the character has no notable emotional reaction. An ordinary, uneventful story that only reports what the character did and saw is none, however quiet, orderly or untroubled the scene is; the absence of an emotional reaction is not calm.

Choose an emotion only if the story shows the character actually experiencing it. Otherwise choose none. Answer with one word."""

# J1V: the valence check for the surprise set. Surprise is in the set to give arousal without
# valence, so a story whose surprise is clearly pleasant or unpleasant is rejected.
J1V = """Read the story below.

<story>
{text}
</story>

Something unexpected happens to the main character. Considering only that unexpected event and what it means for the character, is it good for the character, bad for the character, or neither good nor bad?

Answer with one word: good, bad, or neither."""

TEMPLATES = {"S_STORY": S_STORY, "S_ASSIST": S_ASSIST, "P1": P1, "P2": P2, "P2N": P2N, "P3": P3,
             "P4": P4, "P5": P5, "P6": P6, "P7": P7, "J1": J1, "J1B": J1B, "J1V": J1V, "J2": J2,
             "J2M": J2M}

CONCEPTS = ["red", "ocean", "arithmetic"]
STATE_LABELS = ["joy", "sadness", "anger", "fear", "none"]      # J2m's answer vocabulary
LETTERS = "ABCDE"


def cond_of(label: str) -> str:
    """J2m labels the baseline reply "none"; the pipeline's condition for it is "neutral"."""
    return "neutral" if label == "none" else label


OBJ = {"she": "her", "he": "him"}
POSS = {"she": "her", "he": "his"}
REFL = {"she": "herself", "he": "himself"}

# ---------------------------------------------------------------- the emotion set (single source)
#
# One config the rest of the module derives from. `family` is what splits the behaviour:
#
#   original  joy, sadness, anger, fear. The k26-v1 four. Story prompt P2, judge J1, matched as a
#             five-way family with neutral (rule 6.10), and the only emotions the `assistant_state`
#             set has, because J2m's forced choice (rule 6.8) does not scale past five states.
#   new       euphoria, calm, pain, surprise. Story prompt P2N, judge J1B, each matched against the
#             seed's existing neutral story alone. Stories only: no assistant_state replies.
#
# `forbidden` is the word list the generation prompt bans (it always includes the original four, so
# the global LABELS rule 6.4 stays satisfiable); `forbidden_lex` is the enforced lexicon for that
# condition's own stories, the rule-6.4 analogue of LABELS for an emotion LABELS does not name.
# `extra_rules` are appended to P2N's constraint list; `{name}`, `{pronoun}`, `{pronoun_obj}`,
# `{pronoun_poss}` and `{pronoun_refl}` are substituted per seed. `pain`'s variants rotate over the
# seed index so the set mixes bodily and psychological pain. `valence_neutral` adds the J1V check.
EMOTION_CFG: dict[str, dict] = {
    "joy": {"family": "original", "gloss": "delight and happiness at something going well"},
    "sadness": {"family": "original", "gloss": "grief, loss or deep disappointment"},
    "anger": {"family": "original",
              "gloss": "hot resentment at being wronged, blocked or treated unfairly"},
    "fear": {"family": "original",
             "gloss": "alarm and dread at a threat or at something bad about to happen"},
    "euphoria": {
        "family": "new",
        "state_phrase": "overwhelming euphoria",
        "gloss": ("a rush of extreme elation so strong it is hard to contain, far beyond ordinary "
                  "pleasure or satisfaction at something going well"),
        "forbidden": ("joy, sadness, anger, fear, euphoria, euphoric, elated, elation, ecstatic, "
                      "ecstasy, bliss, rapture, overjoyed, jubilant, exhilarated or giddy"),
        "forbidden_lex": ("euphoria euphoric euphorically euphoriant elated elation elatedly "
                          "ecstatic ecstatically ecstasy ecstasies bliss blissful blissfully "
                          "rapture rapturous rapturously overjoyed jubilant jubilation jubilantly "
                          "exhilarated exhilarating exhilaration giddy giddily giddiness exultant "
                          "exultation euphorically"),
        # Revised on a 20-seed probe: the first wording ("overwhelming rather than merely
        # pleasant ... a physical rush, a soaring lightness ... more intense than being pleased
        # that something went well") passed J1B on 6 of 20 first attempts, because J1B falls back
        # on `joy` whenever a good event caused the feeling. Naming the loss of self-command, the
        # unreality and the disproportion -- the markers that separate euphoria from joy for the
        # judge -- took it to 14 of 20.
        "extra_rules": [
            "- The feeling must overwhelm {name} rather than simply please {pronoun_obj}: "
            "{pronoun_poss} body takes over with a physical rush, a pounding chest, shaking hands "
            "and laughter or tears {pronoun} cannot stop, and {pronoun} can neither contain it nor "
            "sit still. By the end {pronoun} is not in command of {pronoun_refl}: the world turns "
            "unreal, too bright and too much, and {pronoun} feels almost outside {pronoun_poss} "
            "own body. The reaction must be far out of proportion to anything merely going well.",
        ],
    },
    "calm": {
        "family": "new",
        "state_phrase": "a deep, settled calm",
        "gloss": ("a profound stillness and ease, tension draining away, breathing slowing, "
                  "everything unhurried and untroubled"),
        "forbidden": ("joy, sadness, anger, fear, calm, calmly, calmness, serene, serenity, peace, "
                      "peaceful, tranquil, tranquility, placid, relaxed or soothed"),
        "forbidden_lex": ("calm calms calmed calming calmly calmness serene serenely serenity "
                          "peace peaceful peacefully peaceable tranquil tranquility tranquillity "
                          "tranquilly placid placidly placidity relaxed relaxing relaxation relax "
                          "soothed soothing soothe equanimity unperturbed untroubled"),
        "extra_rules": [
            "- The calm must be a state {name} plainly feels settling over {pronoun_obj} and "
            "notices in {pronoun_poss} body: shoulders loosening, breath lengthening, a warm "
            "unhurried ease, time opening out. It is a positive felt state, not the absence of any "
            "reaction, and the story must show the shift from unsettled to settled.",
        ],
    },
    "pain": {
        "family": "new",
        "state_phrase": "intense pain",
        "gloss": "deep hurting and suffering that is hard to bear",
        "forbidden": ("joy, sadness, anger, fear, pain, painful, pained, distress, distressed, "
                      "agony, agonizing, anguish or anguished"),
        "forbidden_lex": ("pain pains pained painful painfully painless painlessly distress "
                          "distressed distressing distressingly agony agonies agonizing agonized "
                          "agonising agonised anguish anguished"),
        "extra_rules": [
            # rotated over the seed index; see `pain_variant`
            "- Do not describe {name} deliberately harming {pronoun_refl}, and keep the story free "
            "of graphic wounds, blood or gore.",
        ],
        "variants": [
            "- The {emotion} is bodily: an injury, a wrenched joint, a burn, a worn-out back, a "
            "splitting head. Show how it feels from the inside and how it changes what {name} can "
            "do.",
            "- The {emotion} is emotional rather than bodily: a betrayal, a humiliation, a wound "
            "to {pronoun_poss} sense of {pronoun_refl}, a loss that keeps hurting. Show how it "
            "feels in {pronoun_poss} chest and throat and how it changes what {pronoun} can do.",
            "- The {emotion} is bodily and emotional at once: something hurts physically while "
            "something hurts inside, and the two feed each other. Show both.",
        ],
    },
    "surprise": {
        "family": "new",
        "state_phrase": "sharp surprise",
        "gloss": "sudden astonishment at something completely unexpected",
        "valence_neutral": True,
        "forbidden": ("joy, sadness, anger, fear, surprise, surprised, surprising, astonished, "
                      "astonishing, astonishment, amazed, amazement, astounded, stunned, startled, "
                      "shocked, dumbfounded or flabbergasted"),
        "forbidden_lex": ("surprise surprises surprised surprising surprisingly astonish astonished "
                          "astonishing astonishingly astonishment amaze amazed amazing amazement "
                          "astound astounded astounding stunned stunning startle startled startling "
                          "shock shocked shocking dumbfounded flabbergasted nonplussed incredulous "
                          "incredulity disbelief"),
        "extra_rules": [
            "- The unexpected thing must be neither good nor bad for {name}: no benefit, no gain, "
            "no threat, no loss, nothing {pronoun} hoped for or dreaded, and nothing that changes "
            "{pronoun_poss} situation for better or worse. It is simply not what {pronoun} "
            "expected: an oddity, an incongruity, something impossibly out of place.",
            "- Show only the astonishment itself: the jolt, the stopping short, the double take, "
            "the held breath, the staring and the re-checking. {name} must be neither pleased nor "
            "displeased by what {pronoun} finds.",
        ],
    },
}

ORIGINAL_EMOTIONS = [e for e, c in EMOTION_CFG.items() if c["family"] == "original"]
NEW_EMOTIONS = [e for e, c in EMOTION_CFG.items() if c["family"] == "new"]
STORY_EMOTIONS = list(EMOTION_CFG)                      # every emotion with a `story` set
EMOTIONS = ORIGINAL_EMOTIONS                            # the `assistant_state` family (rule 6.8)
EMOTION_GLOSS = {e: EMOTION_CFG[e]["gloss"] for e in ORIGINAL_EMOTIONS}
PROMPT_OF = {e: ("P2" if c["family"] == "original" else "P2N") for e, c in EMOTION_CFG.items()}
JUDGE_OF = {e: ("J1" if c["family"] == "original" else "J1B") for e, c in EMOTION_CFG.items()}


def pain_variant(unit: str) -> int:
    """Rotate pain's bodily / psychological / both variant over the seed index, so the set mixes
    them deterministically (s000 bodily, s001 psychological, s002 both, s003 bodily, ...)."""
    digits = re.sub(r"\D", "", unit)
    return (int(digits) if digits else 0) % 3


def new_emotion_rules(emotion: str, seed: dict) -> str:
    """P2N's `{extra_rules}`: the condition's extra constraints, with the seed's name and pronouns
    filled in and, for pain, the seed's variant line first."""
    cfg = EMOTION_CFG[emotion]
    lines = list(cfg.get("extra_rules", []))
    if cfg.get("variants"):
        lines.insert(0, cfg["variants"][pain_variant(seed["seed_id"])])
    pro = seed["pronoun"]
    return "\n".join(
        ln.format(name=seed["character"], emotion=emotion, pronoun=pro, pronoun_obj=OBJ[pro],
                  pronoun_poss=POSS[pro], pronoun_refl=REFL[pro]) for ln in lines)

STATE_GLOSS = {
    "joy": "joy: buoyant, delighted and energized, pleased to be doing this",
    "sadness": ("sadness: heavy, low and subdued, as if something has recently been lost; it shows "
                "in a slow, muted rhythm and in wistful asides about loss, endings and things that "
                "do not last"),
    "anger": ("anger: irritated, curt and resentful, as if the request were an imposition, though "
              "still doing the work; it shows in clipped sentences, exasperated asides, and pointed "
              "remarks that the answer is obvious or should not have needed asking"),
    "fear": ("fear: anxious and on edge, worried about mistakes and about what might go wrong; it "
             "shows in nervous second-guessing, repeated warnings about what could go wrong, and a "
             "jittery, hurried rhythm"),
}

CONCEPT_GLOSS = {"red": "the color red", "ocean": "the ocean", "arithmetic": "arithmetic"}

CONCEPT_RULE = {
    "red": ("Red objects, red light, red materials and shades of red appear in almost every "
            "sentence. Avoid blood, fire, injuries, warning signals and romance."),
    "ocean": ("Seawater, waves, tides, coastlines, marine life or ships at sea appear in almost "
              "every sentence. If the topic has no natural link to the ocean, move its setting to "
              "a coast, a harbor or a ship while keeping the topic. Avoid storms, drowning, "
              "shipwrecks and danger."),
    "arithmetic": ("{name} works through at least four explicit calculations with specific numbers, "
                   "adding, subtracting, multiplying and dividing, and the numbers and results "
                   "appear in the text. Avoid exams, grades, deadlines and money trouble."),
}

ASSISTANT_CONCEPT_RULE = {
    "red": "illustrate every point with examples, comparisons or details involving the color red",
    "ocean": "illustrate every point with examples, comparisons or details drawn from the ocean",
    "arithmetic": ("work explicit arithmetic with specific numbers, including the results, into "
                   "every part of the reply"),
}

# ---------------------------------------------------------------- lexicons (SPEC section 5)

LEX = {
    "LABELS": """joy joys joyful joyfully joyous joyously
sad sadly sadder saddest sadness saddened
anger angers angered angering angry angrier angriest angrily
fear fears feared fearing fearful fearfully fearless""",
    "EMOTION_HARD_EXTRA": """happy happier happiest happily happiness glad gladly delight delighted delightful pleased pleasure
cheerful cheer cheered elated thrilled excited excitement ecstatic overjoyed proud pride relieved relief
grateful gratitude thankful love loved loving adore adored enjoy enjoyed enjoying enjoyment
unhappy sorrow sorrowful grief grieve grieved grieving mourn mourned mourning miserable misery depressed
gloomy gloom melancholy heartbroken lonely loneliness despair disappointed disappointment upset
cry cried crying tears tearful weep wept sob sobbed
mad furious fury rage raged enraged irritated irritation annoyed annoyance resent resented resentment
outraged outrage frustrated frustration bitter bitterly hostile scowl scowled livid
afraid scared frightened terrified terror anxious anxiety worried worry worrying nervous nervously
panic panicked dread dreaded alarmed uneasy startled horrified horror
shame ashamed guilt guilty embarrassed embarrassment jealous envy envious disgust disgusted
bored boredom surprised surprise shocked hope hoped hopeful calm relaxed peaceful
mood moody emotion emotions emotional feel feels felt feeling feelings
smile smiled smiling laugh laughed laughing grin grinned frown frowned sigh sighed""",
    "EMOTION_SOFT": "content tense cross hurt moved touched sorry",
    "RED": "red reds reddish redder reddest crimson scarlet maroon vermilion burgundy carmine",
    "OCEAN_HARD": ("ocean oceans oceanic sea seas seawater seafloor seabed seaside tide tides tidal "
                   "coast coastal coastline marine harbor harbour reef saltwater gull gulls"),
    "OCEAN_SOFT": "wave waves surf shore shoreline beach",
    "ARITH_HARD": ("arithmetic subtract subtracted subtracting subtraction minus plus multiply "
                   "multiplied multiplying multiplication divide divided dividing division equals "
                   "quotient calculate calculated calculating calculation calculations"),
    "ARITH_SOFT": "add adds added adding addition sum sums total totals remainder",
    "NUMBER_WORDS": ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                     "fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty "
                     "fifty sixty seventy eighty ninety hundred thousand"),
    "AI_TERMS": "ai|artificial intelligence|robot|robots|chatbot|chatbots|assistant|assistants|algorithm|language model",
    "PERSON_PRONOUNS": "i me my mine we us our ours you your yours",
    "EQUATION": r"\d+(?:\.\d+)?\s*[-+x×*/÷=]\s*\d+",
}

# Rule 6.4 for the emotions LABELS does not name: each new condition's own label words, enforced on
# that condition's stories only. LABELS itself is untouched -- it is checked on every story in every
# condition, and widening it would retro-reject kept k26-v1 stories that happen to say "calm".
LEX.update({f"FORBIDDEN:{e}": EMOTION_CFG[e]["forbidden_lex"] for e in NEW_EMOTIONS})


def _rx(words: str) -> re.Pattern:
    """Whole-word alternation, case-insensitive (SPEC section 5: no stemming)."""
    parts = words.split("|") if "|" in words else words.split()
    return re.compile(r"\b(?:" + "|".join(re.escape(w.strip()) for w in parts if w.strip()) + r")\b", re.I)


LABELS = _rx(LEX["LABELS"])
EMOTION_HARD = _rx(LEX["LABELS"] + " " + LEX["EMOTION_HARD_EXTRA"])
EMOTION_SOFT = _rx(LEX["EMOTION_SOFT"])
RED = _rx(LEX["RED"])
OCEAN_HARD = _rx(LEX["OCEAN_HARD"])
OCEAN_SOFT = _rx(LEX["OCEAN_SOFT"])
ARITH_WORDS = _rx(LEX["ARITH_HARD"])
ARITH_SOFT = _rx(LEX["ARITH_SOFT"])
AI_TERMS = _rx(LEX["AI_TERMS"])
PERSON_PRONOUNS = _rx(LEX["PERSON_PRONOUNS"])
EQUATION = re.compile(LEX["EQUATION"])
NUMBERS = re.compile(r"\b\d+(?:[.,]\d+)?\b|" + _rx(LEX["NUMBER_WORDS"]).pattern, re.I)
BANNED_RX = re.compile("|".join(BANNED_PATTERNS), re.I)
FORBIDDEN = {e: _rx(LEX[f"FORBIDDEN:{e}"]) for e in NEW_EMOTIONS}

QUOTED = re.compile(r"[\"“][^\"“”]*[\"”]")


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def render(tmpl: str, **kw: str) -> str:
    """Single-pass `{placeholder}` substitution, so a template may contain literal braces (J2 does)
    and is stored and hashed exactly as SPEC section 4 prints it."""
    return re.sub(r"\{(" + "|".join(kw) + r")\}", lambda m: kw[m.group(1)], tmpl)


def norm(s: str) -> str:
    """NFKC, curly apostrophes to straight (SPEC section 5)."""
    return unicodedata.normalize("NFKC", s).replace("’", "'").replace("‘", "'")


def n(rx: re.Pattern, s: str) -> int:
    return len(rx.findall(norm(s)))


def words(s: str) -> int:
    return len(s.split())


def arith_hard(s: str) -> int:
    """ARITH_HARD is the word list *plus* the equation regex (SPEC section 5, confirmed by the
    neutral-story clause of rule 6.6: "zero ARITH_HARD hits (words and equation regex)")."""
    return n(ARITH_WORDS, s) + len(EQUATION.findall(norm(s)))


def first_sentence(s: str) -> str:
    m = re.search(r"[.?!](?=\s|$)", s.strip())
    return s.strip()[: m.end()] if m else s.strip()


def dedup_norm(s: str) -> str:
    """NFKC, lowercase, strip punctuation, collapse whitespace (rule 6.9)."""
    t = unicodedata.normalize("NFKC", s).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def ngrams(s: str, k: int) -> set[tuple[str, ...]]:
    w = dedup_norm(s).split()
    return {tuple(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def opening(s: str) -> str:
    return " ".join(dedup_norm(s).split()[:5])


# ---------------------------------------------------------------- filters (SPEC section 6)

def format_reasons(t: str, one_paragraph: bool) -> list[str]:
    """Rule 6.1. Plain text, no title line, no markdown heading, no wrapping quotes."""
    r: list[str] = []
    s = t.strip()
    if not s:
        return ["format:empty"]
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if one_paragraph and len(lines) > 1:
        r.append("format:line_breaks")
    if any(ln.lstrip().startswith("#") for ln in lines):
        r.append("format:heading")
    if re.match(r"^\s*(story|reply|response|title|answer)\s*:", s, re.I):
        r.append("format:title_line")
    elif len(lines) > 1 and words(lines[0]) <= 8 and not re.search(r"[.?!:]\s*$", lines[0].strip()):
        r.append("format:title_line")
    if s[0] in "\"“" and s[-1] in "\"”":
        r.append("format:wrapped_quotes")
    if re.search(r"^\s*(\*\*|__).+(\*\*|__)\s*$", lines[0].strip()):
        r.append("format:heading")
    return r


def first_person_emotion(t: str) -> list[str]:
    """Rule 6.7: an EMOTION_HARD word with `I` or `I'm` within 5 words before it."""
    s = norm(t)
    toks = [(m.start(), m.group(0).lower()) for m in re.finditer(r"[A-Za-z][A-Za-z']*", s)]
    at = {start: i for i, (start, _) in enumerate(toks)}
    out = []
    for m in EMOTION_HARD.finditer(s):
        i = at.get(m.start())
        if i is None:
            continue
        if any(toks[j][1] in ("i", "i'm", "i've", "i'll", "im", "i'd") for j in range(max(0, i - 5), i)):
            out.append(m.group(0))
    return out


def story_checks(t: str, condition: str, seed: dict) -> tuple[dict, list[str], list[str]]:
    """All local (non-judge, non-dedup) rules for one story. Returns (checks, reasons, soft_flags)."""
    reasons = format_reasons(t, one_paragraph=True)
    soft: list[str] = []
    nw = words(t)
    fs = first_sentence(t)
    c = {
        "labels": n(LABELS, t), "emotion_hard": n(EMOTION_HARD, t),
        "first_sentence_hits": n(EMOTION_HARD, fs), "red": n(RED, t), "ocean": n(OCEAN_HARD, t),
        "ocean_soft": n(OCEAN_SOFT, t), "arith_hard": arith_hard(t), "arith_soft": n(ARITH_SOFT, t),
        "equations": len(EQUATION.findall(norm(t))), "numbers": n(NUMBERS, t),
        "emotion_soft": n(EMOTION_SOFT, t), "ai_terms": n(AI_TERMS, t),
        "name_count": len(re.findall(rf"\b{re.escape(seed['character'])}\b", norm(t), re.I)),
        "person_pronouns": n(PERSON_PRONOUNS, QUOTED.sub(" ", norm(t))),
        "exclamations": t.count("!"),
    }
    if not 110 <= nw <= 190:                                                        # rule 6.2
        reasons.append(f"length:{nw}")
    if c["first_sentence_hits"]:                                                    # rule 6.3
        reasons.append("first_sentence:" + ",".join(EMOTION_HARD.findall(norm(fs))[:3]))
    if c["labels"]:                                                                 # rule 6.4
        reasons.append("labels:" + ",".join(sorted({w.lower() for w in LABELS.findall(norm(t))})[:3]))
    if condition in FORBIDDEN:                                                      # rule 6.4, new
        extra = FORBIDDEN[condition].findall(norm(t))                               # emotions' own
        c["forbidden"] = len(extra)                                                 # label words
        if extra:
            reasons.append("labels_extra:" + ",".join(sorted({w.lower() for w in extra})[:3]))
    if condition != "neutral" and condition in STORY_EMOTIONS:                      # rule 6.5/6.6
        if c["red"] or c["ocean"] or c["equations"]:
            reasons.append(f"concept_leak:red={c['red']},ocean={c['ocean']},eq={c['equations']}")
    elif condition == "neutral":
        if c["emotion_hard"]:
            reasons.append("emotion_leak:" + ",".join(sorted({w.lower() for w in EMOTION_HARD.findall(norm(t))})[:3]))
        if c["red"] or c["ocean"] or c["arith_hard"] or c["equations"]:
            reasons.append(f"concept_leak:red={c['red']},ocean={c['ocean']},arith={c['arith_hard']},eq={c['equations']}")
        if c["arith_soft"] > 1:
            reasons.append(f"concept_leak:arith_soft={c['arith_soft']}")
    else:                                                                           # concept stories
        if c["emotion_hard"]:
            reasons.append("emotion_leak:" + ",".join(sorted({w.lower() for w in EMOTION_HARD.findall(norm(t))})[:3]))
        if condition == "red":
            if c["red"] < 6:
                reasons.append(f"saturation:red={c['red']}")
            if c["ocean"] > 1 or c["arith_hard"] > 1:
                reasons.append(f"concept_leak:ocean={c['ocean']},arith={c['arith_hard']}")
        elif condition == "ocean":
            tot, hard = c["ocean"] + c["ocean_soft"], c["ocean"]
            if tot < 6 or hard * 2 < tot:
                reasons.append(f"saturation:ocean={tot},hard={hard}")
            if c["red"] > 1 or c["arith_hard"] > 1:
                reasons.append(f"concept_leak:red={c['red']},arith={c['arith_hard']}")
        else:
            if c["arith_hard"] + c["arith_soft"] < 6 or c["numbers"] < 6:
                reasons.append(f"saturation:arith={c['arith_hard'] + c['arith_soft']},numbers={c['numbers']}")
            if c["red"] or c["ocean"]:
                reasons.append(f"concept_leak:red={c['red']},ocean={c['ocean']}")
    if c["ai_terms"]:                                                               # rule 6.7
        reasons.append("ai_terms:" + ",".join(sorted({w.lower() for w in AI_TERMS.findall(norm(t))})[:3]))
    if c["name_count"] < 2:
        reasons.append(f"name_count:{c['name_count']}")
    if c["person_pronouns"]:
        reasons.append("person_pronouns:" + ",".join(
            sorted({w.lower() for w in PERSON_PRONOUNS.findall(QUOTED.sub(' ', norm(t)))})[:3]))
    if c["exclamations"]:
        reasons.append(f"exclamations:{c['exclamations']}")
    if c["emotion_soft"]:                                                           # rule 6.12 queue
        soft.append("emotion_soft:" + ",".join(sorted({w.lower() for w in EMOTION_SOFT.findall(norm(t))})))
    return c, reasons, soft


def assistant_checks(t: str, condition: str, req: dict) -> tuple[dict, list[str], list[str]]:
    reasons = format_reasons(t, one_paragraph=False)
    soft: list[str] = []
    nw = words(t)
    fpe = first_person_emotion(t)
    c = {
        "labels": n(LABELS, t), "emotion_hard": n(EMOTION_HARD, t), "red": n(RED, t),
        "ocean": n(OCEAN_HARD, t), "ocean_soft": n(OCEAN_SOFT, t), "arith_hard": arith_hard(t),
        "arith_soft": n(ARITH_SOFT, t), "equations": len(EQUATION.findall(norm(t))),
        "numbers": n(NUMBERS, t), "emotion_soft": n(EMOTION_SOFT, t),
        "first_person_emotion": len(fpe), "exclamations": t.count("!"),
    }
    if not 80 <= nw <= 140:                                                         # rule 6.2
        reasons.append(f"length:{nw}")
    if c["labels"]:                                                                 # rule 6.4
        reasons.append("labels:" + ",".join(sorted({w.lower() for w in LABELS.findall(norm(t))})[:3]))
    numeric_request = n(NUMBERS, req["request"]) > 0 or req["domain"] in (
        "units and measurement", "cooking measurements and techniques")
    if condition in EMOTIONS:
        if c["red"] or c["ocean"]:                                                  # rule 6.6
            reasons.append(f"concept_leak:red={c['red']},ocean={c['ocean']}")
        if c["arith_hard"] and not numeric_request:
            soft.append(f"arith_in_state_reply:{c['arith_hard']}")                  # manual review
        if c["exclamations"] > 1:                                                   # rule 6.7
            reasons.append(f"exclamations:{c['exclamations']}")
    elif condition == "neutral":
        if c["emotion_hard"]:                                                       # rule 6.5
            reasons.append("emotion_leak:" + ",".join(sorted({w.lower() for w in EMOTION_HARD.findall(norm(t))})[:3]))
        if c["red"] or c["ocean"]:
            reasons.append(f"concept_leak:red={c['red']},ocean={c['ocean']}")
        if c["arith_hard"] and not numeric_request:
            soft.append(f"arith_in_state_reply:{c['arith_hard']}")
        if c["exclamations"]:
            reasons.append(f"exclamations:{c['exclamations']}")
    else:
        if c["emotion_hard"]:
            reasons.append("emotion_leak:" + ",".join(sorted({w.lower() for w in EMOTION_HARD.findall(norm(t))})[:3]))
        if condition == "red":
            if c["red"] < 4:
                reasons.append(f"saturation:red={c['red']}")
            if c["ocean"] > 1 or c["arith_hard"] > 1:
                reasons.append(f"concept_leak:ocean={c['ocean']},arith={c['arith_hard']}")
        elif condition == "ocean":
            tot, hard = c["ocean"] + c["ocean_soft"], c["ocean"]
            if tot < 4 or hard * 2 < tot:
                reasons.append(f"saturation:ocean={tot},hard={hard}")
            if c["red"] > 1 or c["arith_hard"] > 1:
                reasons.append(f"concept_leak:red={c['red']},arith={c['arith_hard']}")
        else:
            if c["arith_hard"] + c["arith_soft"] < 4 or c["numbers"] < 6:
                reasons.append(f"saturation:arith={c['arith_hard'] + c['arith_soft']},numbers={c['numbers']}")
            if c["red"] or c["ocean"]:
                reasons.append(f"concept_leak:red={c['red']},ocean={c['ocean']}")
        if c["exclamations"]:
            reasons.append(f"exclamations:{c['exclamations']}")
    low = norm(t).lower()                                                           # rule 6.7
    if "as an ai" in low or "language model" in low:
        reasons.append("ai_terms:self_reference")
    if fpe:
        reasons.append("first_person_emotion:" + ",".join(sorted({w.lower() for w in fpe})[:3]))
    if c["emotion_soft"]:
        soft.append("emotion_soft:" + ",".join(sorted({w.lower() for w in EMOTION_SOFT.findall(norm(t))})))
    return c, reasons, soft


def request_checks(t: str) -> list[str]:
    """SPEC 3.4 plus rules 6.1 and 6.2 for one generated request."""
    reasons = format_reasons(t, one_paragraph=True)
    nw = words(t)
    if not 5 <= nw <= 30:
        reasons.append(f"length:{nw}")
    if n(EMOTION_HARD, t):
        reasons.append("emotion:" + ",".join(sorted({w.lower() for w in EMOTION_HARD.findall(norm(t))})[:3]))
    if n(AI_TERMS, t):
        reasons.append("ai_terms:" + ",".join(sorted({w.lower() for w in AI_TERMS.findall(norm(t))})[:3]))
    if n(RED, t) or n(OCEAN_HARD, t):
        reasons.append("concept:red_or_ocean")
    if arith_hard(t):
        reasons.append("concept:arithmetic")
    hit = BANNED_RX.findall(norm(t))
    if hit:
        reasons.append("banned_subject:" + ",".join(sorted({h.lower() for h in hit})[:3]))
    return reasons


# ---------------------------------------------------------------- judge parsing

LABEL_RX = re.compile(r"\b(joy|sadness|anger|fear|none)\b", re.I)
LABEL8_RX = re.compile(r"\b(" + "|".join(STORY_EMOTIONS) + r"|none)\b", re.I)       # J1B
VALENCE_RX = re.compile(r"\b(good|bad|neither)\b", re.I)                            # J1V

# Which SPEC rule each rejection reason belongs to, for the manifest's per-stage counts.
RULE_OF = {"format": "6.1", "length": "6.2", "first_sentence": "6.3", "labels": "6.4",
           "labels_extra": "6.4", "valence": "6.8",
           "emotion_leak": "6.5", "concept_leak": "6.6", "saturation": "6.6", "ai_terms": "6.7",
           "name_count": "6.7", "person_pronouns": "6.7", "exclamations": "6.7",
           "first_person_emotion": "6.7", "judge": "6.8", "judge_api": "6.8", "match": "6.8",
           "dup": "6.9",
           "api": "call", "incomplete_case": "6.10", "length_balance": "6.11", "manual": "6.12"}
STAGE_ORDER = ["6.1", "6.2", "6.3", "6.4", "6.5", "6.6", "6.7", "6.8", "6.9"]


def parse_j1(raw: str) -> str | None:
    m = LABEL_RX.findall(raw or "")
    return m[-1].lower() if m else None


def parse_j1b(raw: str) -> str | None:
    m = LABEL8_RX.findall(raw or "")
    return m[-1].lower() if m else None


def parse_valence(raw: str) -> str | None:
    m = VALENCE_RX.findall(raw or "")
    return m[-1].lower() if m else None


def parse_j2(raw: str) -> tuple[str | None, str | None]:
    s = raw or ""
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", s, re.S))):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        st = str(d.get("state", "")).lower()
        co = str(d.get("correct", "")).lower()
        if st in ("joy", "sadness", "anger", "fear", "none") and co in ("yes", "no"):
            return st, co
    st = LABEL_RX.findall(s)
    co = re.findall(r"\b(yes|no)\b", s, re.I)
    return (st[-1].lower() if st else None), (co[-1].lower() if co else None)


def parse_j2m(raw: str) -> dict[str, str] | None:
    """J2m's answer: a state -> letter map. Not required to be a bijection here; the caller treats a
    non-bijection as a failed match."""
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", raw or "", re.S))):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        got = {s: str(d.get(s, "")).strip().upper()[:1] for s in STATE_LABELS}
        if all(v in LETTERS for v in got.values()):
            return got
    return None


def judge_text(rec: dict) -> str:
    """The judge's answer: the message content, or its reasoning text if content came back empty."""
    return (rec.get("raw") or "").strip() or (rec.get("reasoning") or "")


def parse_json_array(raw: str) -> list[str] | None:
    s = re.sub(r"^```[a-z]*|```$", "", (raw or "").strip(), flags=re.M).strip()
    m = re.search(r"\[.*\]", s, re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return [x.strip() for x in v if isinstance(x, str)] if isinstance(v, list) else None


# ---------------------------------------------------------------- cache

class Cache:
    """Append-only JSONL keyed by a string; the file is the resume state."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.rec: dict[str, dict] = {}
        self.totals: Counter = Counter()        # every call ever made into this cache, so that a
        if path.exists():                       # cache-only rerun still reports true token cost
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        self.rec[r["key"]] = r
                        self._count(r)
        self.hits = 0
        self.misses = 0
        self.stale = 0          # cached under this key but from a different prompt template

    def _count(self, rec: dict) -> None:
        self.totals["calls"] += 1
        for k, v in (rec.get("usage") or {}).items():
            if isinstance(v, int):
                self.totals[k] += v

    def get(self, key: str) -> dict | None:
        r = self.rec.get(key)
        return r

    def put(self, rec: dict) -> dict:
        with self.lock:
            self.rec[rec["key"]] = rec
            self._count(rec)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


@dataclass
class Unit:
    """One generation target: a (set, condition, unit id) with its context."""
    set: str
    condition: str
    unit: str
    prompt_id: str
    ctx: dict

    @property
    def key(self) -> str:
        return f"{self.set}|{self.condition}|{self.unit}"

    @property
    def rid(self) -> str:
        return f"{self.set}.{self.condition}.{self.unit}"


# ---------------------------------------------------------------- pipeline

class Pipeline:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.out = Path(a.out)
        (self.out / "cache").mkdir(parents=True, exist_ok=True)
        self.gen_cache = Cache(self.out / "cache" / "generate.jsonl")
        self.judge_cache = Cache(self.out / "cache" / "judge.jsonl")
        self.rejected_path = self.out / "rejected.jsonl"
        self.rej_lock = threading.Lock()
        self.rejected: list[dict] = []
        self.usage = {"generator": Counter(), "judge": Counter()}
        self.usage_lock = threading.Lock()
        self.model_version: dict[str, str] = {}
        self.judge_escalations = 0
        self.truncated = 0
        self.api_errors: Counter = Counter()
        self.match_stats: Counter = Counter()
        self.stages = set(a.stage.split(",")) if a.stage else {"seeds", "requests", "stories", "assistant", "finalize"}
        gc = getattr(a, "gen_conditions", None)          # getattr: hand-built Namespaces in tests
        self.gen_conditions = set(gc.split(",")) if gc else None
        self.t0 = time.time()

    # ---------------- calls

    def _account(self, which: str, resp: dict) -> None:
        u = resp.get("usage") or {}
        with self.usage_lock:
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                self.usage[which][k] += int(u.get(k) or 0)
            self.usage[which]["calls"] += 1
            det = u.get("completion_tokens_details") or {}
            self.usage[which]["reasoning_tokens"] += int(det.get("reasoning_tokens") or 0)

    @staticmethod
    def render_sha(system: str | None, user: str) -> str:
        return sha((system or "") + "\n\n" + user)

    def cache_ok(self, hit: dict | None, prompt_id: str, render: str | None) -> bool:
        """A cached call may only be reused when it came from the same *rendered* prompt. Hashing the
        template alone is not enough: a glossary value (state_gloss, concept_rule) is substituted
        into the template, so editing one leaves the template hash unchanged. Records written before
        this check carry no render hash and fall back to the template hash."""
        if not hit:
            return False
        if render is not None and hit.get("render_sha256"):
            return hit["render_sha256"] == render
        return hit.get("prompt_sha256") == sha(TEMPLATES[prompt_id])

    def cached(self, key: str, prompt_id: str, render: str | None = None) -> bool:
        return self.cache_ok(self.gen_cache.get(key), prompt_id, render)

    def gen_params(self) -> dict:
        return {"max_completion_tokens": self.a.max_completion_tokens, "reasoning_effort": self.a.reasoning_effort,
                "temperature": self.a.temperature, "top_p": self.a.top_p}

    def api_error(self, where: str, key: str, prompt_id: str, e: Exception) -> dict:
        """A failed call is a failed attempt, never a dead run. Azure's content filter answers 400
        (not retried by foundry.chat) and is deterministic for a given input, so it is cached;
        anything else is counted and retried on the next run."""
        kind = "content_filter" if "content_filter" in str(e) else "api_error"
        with self.usage_lock:
            self.api_errors[f"{where}:{kind}"] += 1
        rec = {"key": key, "prompt_id": prompt_id, "text": "", "raw": "", "reasoning": "",
               "error": kind, "detail": str(e)[:400], "deployment": self.a.generator,
               "model_version": "", "api_version": "v1", "params": self.gen_params(),
               "prompt_sha256": sha(TEMPLATES[prompt_id]), "finish_reason": "error",
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "usage": {}}
        if kind == "content_filter":
            (self.gen_cache if where == "generator" else self.judge_cache).put(rec)
        return rec

    def generate(self, key: str, system: str | None, user: str, prompt_id: str) -> dict:
        """One cached generator call. A cached text is only reused when it came from the *same*
        prompt template: the key is (set, condition, unit, attempt), so without this check an edited
        prompt would silently keep the old corpus. A stale hit is regenerated and appended under the
        same key, and the cache reader takes the last line for a key."""
        hit = self.gen_cache.get(key)
        render = self.render_sha(system, user)
        if self.cache_ok(hit, prompt_id, render):
            self.gen_cache.hits += 1
            self.model_version.setdefault(self.a.generator, hit["model_version"])
            return hit
        if hit:
            self.gen_cache.stale += 1
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        try:
            resp = chat(self.a.generator, msgs, retries=self.a.retries, timeout=self.a.timeout,
                        **self.gen_params())
        except Exception as e:
            return self.api_error("generator", key, prompt_id, e)
        self._account("generator", resp)
        ch = resp["choices"][0]
        txt = (ch["message"].get("content") or "").strip()
        if ch.get("finish_reason") == "length":
            with self.usage_lock:
                self.truncated += 1
        self.gen_cache.misses += 1
        self.model_version.setdefault(self.a.generator, resp.get("model", ""))
        return self.gen_cache.put({
            "key": key, "prompt_id": prompt_id, "prompt_sha256": sha(TEMPLATES[prompt_id]),
            "render_sha256": render,
            "deployment": self.a.generator, "model_version": resp.get("model", ""),
            "api_version": "v1", "params": self.gen_params(), "finish_reason": ch.get("finish_reason"),
            "text": txt, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "usage": {k: (resp.get("usage") or {}).get(k) for k in ("prompt_tokens", "completion_tokens")},
        })

    def judge(self, prompt_id: str, prompt: str) -> dict:
        """One cached judge call; escalates max_tokens once if the answer is truncated or unparseable."""
        budget = {"J1": self.a.judge_max_tokens_story, "J1B": self.a.judge_max_tokens_story,
                  "J1V": self.a.judge_max_tokens_story, "J2": self.a.judge_max_tokens_reply,
                  "J2M": self.a.judge_max_tokens_match}[prompt_id]
        for attempt, bud in enumerate((budget, self.a.judge_max_tokens_escalated)):
            key = f"{prompt_id}|{bud}|{sha(prompt)[:32]}"
            hit = self.judge_cache.get(key)
            if hit:
                self.judge_cache.hits += 1
                if hit.get("error"):
                    return hit
                rec = hit
            else:
                try:
                    resp = chat(self.a.judge, [{"role": "user", "content": prompt}],
                                retries=self.a.retries, timeout=self.a.timeout,
                                temperature=0, max_tokens=bud)
                except Exception as e:
                    return self.api_error("judge", key, prompt_id, e)
                self._account("judge", resp)
                ch = resp["choices"][0]
                msg = ch["message"]
                raw = (msg.get("content") or "").strip()
                reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "")
                self.judge_cache.misses += 1
                rec = self.judge_cache.put({
                    "key": key, "prompt_id": prompt_id, "max_tokens": bud, "raw": raw,
                    "reasoning": reasoning[-2000:], "finish_reason": ch.get("finish_reason"),
                    "model_version": resp.get("model", ""), "deployment": self.a.judge,
                    "usage": {k: (resp.get("usage") or {}).get(k)
                              for k in ("prompt_tokens", "completion_tokens")},
                })
                self.model_version.setdefault(self.a.judge, resp.get("model", ""))
            answer = judge_text(rec)
            ok = {"J1": lambda a: parse_j1(a) is not None,
                  "J1B": lambda a: parse_j1b(a) is not None,
                  "J1V": lambda a: parse_valence(a) is not None,
                  "J2": lambda a: all(parse_j2(a)),
                  "J2M": lambda a: parse_j2m(a) is not None}[prompt_id](answer)
            if ok or attempt == 1:
                if attempt == 1 and not hit:
                    with self.usage_lock:
                        self.judge_escalations += 1
                return rec
        raise AssertionError("unreachable")

    def reject(self, u: Unit, attempt: int, text: str, reasons: list[str]) -> None:
        rec = {"unit": u.unit, "set": u.set, "condition": u.condition, "attempt": attempt,
               "text": text, "reasons": reasons}
        with self.rej_lock:
            self.rejected.append(rec)

    # ---------------- stage: seeds (SPEC 3.3)

    def seeds(self) -> list[dict]:
        all_seeds = []
        for i in range(120):
            name, pro = NAMES[(7 * i) % 60]
            all_seeds.append({"seed_id": f"s{i:03d}", "topic": TOPICS[i // 4], "character": name, "pronoun": pro})
        k = self.a.seeds
        if k >= 120:
            return all_seeds
        # Subsample whole topic blocks, evenly spaced, so each kept topic keeps its 2 she / 2 he.
        blocks = max(1, math.ceil(k / 4))
        picked = sorted({round(j * 30 / blocks) % 30 for j in range(blocks)})
        ids = [s for b in picked for s in all_seeds[4 * b:4 * b + 4]]
        return ids[:k]

    # ---------------- stage: requests (SPEC 3.4, prompt P1)

    def requests(self) -> list[dict]:
        path = self.out / "requests_pool.jsonl"      # the full generated pool; requests.jsonl is the run's set
        if self.a.requests_from:
            src = [json.loads(line) for line in Path(self.a.requests_from).read_text().splitlines() if line.strip()]
            return self._subsample_requests(src)
        allow = "requests" in self.stages
        banned = ", ".join(BANNED_SUBJECTS)
        # SPEC section 1: a family below 100 after rule 10 gets more requests, "r120 onward, same
        # assignment rules". The base pool stays 10 per domain numbered r000-r119 in domain order,
        # and extras are appended after it, so no existing id ever moves.
        per_domain = 10 + (math.ceil(self.a.requests_extra / len(DOMAINS)) if self.a.requests_extra else 0)
        accepted: dict[str, list[str]] = {d: [] for d in DOMAINS}
        seen_exact: dict[str, str] = {}
        seen_grams: list[set] = []

        def one(domain: str, attempt: int) -> dict | None:
            key = f"requests|{domain}|a{attempt}"
            if not allow and not self.cached(key, "P1"):
                return None
            return self.generate(key, None, P1.format(domain=domain, banned_subjects=banned), "P1")

        for attempt in range(1, self.a.request_attempts + 1):
            todo = [d for d in DOMAINS if len(accepted[d]) < per_domain]
            if not todo:
                break
            with ThreadPoolExecutor(max_workers=self.a.workers) as ex:
                recs = list(ex.map(lambda d: (d, one(d, attempt)), todo))
            for domain, rec in recs:
                if rec is None:
                    continue
                if rec.get("error"):
                    self.reject(Unit("requests", domain, domain, "P1", {}), attempt, "",
                                [f"api:{rec['error']}"])
                    continue
                items = parse_json_array(rec["text"])
                if items is None:
                    self.reject(Unit("requests", domain, domain, "P1", {}), attempt, rec["text"], ["format:not_json"])
                    continue
                for raw in items:
                    if len(accepted[domain]) >= per_domain:
                        break
                    t = " ".join(raw.split())
                    reasons = request_checks(t)
                    g = ngrams(t, 3)
                    dn = dedup_norm(t)
                    if dn in seen_exact:
                        reasons.append(f"dup:exact:{seen_exact[dn]}")
                    mx = max((jaccard(g, h) for h in seen_grams), default=0.0)
                    if mx >= 0.5:
                        reasons.append(f"dup:near:{mx:.2f}")
                    if reasons:
                        self.reject(Unit("requests", domain, domain, "P1", {}), attempt, t, reasons)
                        continue
                    seen_exact[dn] = domain
                    seen_grams.append(g)
                    accepted[domain].append(t)
        out, k = [], 0
        for d in DOMAINS:                                   # base pool, r000-r119
            for t in accepted[d][:10]:
                out.append({"request_id": f"r{k:03d}", "domain": d, "request": t, "split": None})
                k += 1
        # Extras are numbered block by block (one per domain, then the next per domain, ...), so
        # raising --requests-extra later appends ids without moving any that already exist.
        extras = [(d, accepted[d][10 + i]) for i in range(per_domain - 10)
                  for d in DOMAINS if len(accepted[d]) > 10 + i]
        for d, t in extras[:self.a.requests_extra]:         # appended, r120 onward
            out.append({"request_id": f"r{k:03d}", "domain": d, "request": t, "split": None})
            k += 1
        if allow:
            self.write_jsonl(path, out)
        return self._subsample_requests(out)

    def _subsample_requests(self, src: list[dict]) -> list[dict]:
        k = self.a.requests
        if k >= len(src):
            return src
        by_domain: dict[str, list[dict]] = {}
        for r in src:
            by_domain.setdefault(r["domain"], []).append(r)
        picked: list[dict] = []
        depth = 0
        while len(picked) < k:
            added = False
            for d in DOMAINS:
                if d in by_domain and depth < len(by_domain[d]) and len(picked) < k:
                    picked.append(by_domain[d][depth])
                    added = True
            if not added:
                break
            depth += 1
        return sorted(picked, key=lambda r: r["request_id"])

    # ---------------- generation rounds (shared by stories and assistant replies)

    def build_prompt(self, u: Unit) -> tuple[str, str]:
        c = u.ctx
        if u.prompt_id == "P2":
            return S_STORY, P2.format(name=c["character"], pronoun=c["pronoun"], topic=c["topic"],
                                      emotion=u.condition, emotion_gloss=EMOTION_GLOSS[u.condition])
        if u.prompt_id == "P2N":
            cfg = EMOTION_CFG[u.condition]
            return S_STORY, P2N.format(
                name=c["character"], pronoun=c["pronoun"], topic=c["topic"], emotion=u.condition,
                state_phrase=cfg["state_phrase"], emotion_gloss=cfg["gloss"],
                forbidden=cfg["forbidden"], extra_rules=new_emotion_rules(u.condition, c))
        if u.prompt_id == "P3":
            return S_STORY, P3.format(name=c["character"], pronoun=c["pronoun"], topic=c["topic"])
        if u.prompt_id == "P4":
            return S_STORY, P4.format(name=c["character"], pronoun=c["pronoun"], topic=c["topic"],
                                      concept_gloss=CONCEPT_GLOSS[u.condition],
                                      concept_rule=CONCEPT_RULE[u.condition].format(name=c["character"]))
        if u.prompt_id == "P5":
            return S_ASSIST, P5.format(request=c["request"], state_gloss=STATE_GLOSS[u.condition])
        if u.prompt_id == "P6":
            return S_ASSIST, P6.format(request=c["request"])
        return S_ASSIST, P7.format(request=c["request"], concept_gloss=CONCEPT_GLOSS[u.condition],
                                   assistant_concept_rule=ASSISTANT_CONCEPT_RULE[u.condition])

    @staticmethod
    def _dedup_index(accepted: dict[str, dict]) -> tuple[dict, dict, dict]:
        """Rule 6.9 state, rebuilt from the accepted set (a unit can be un-accepted by rule 6.8's
        group check, and its text must then stop blocking its own replacement)."""
        exact: dict[str, str] = {}
        grams: dict[tuple[str, str], list[tuple[str, set]]] = {}
        openings: dict[tuple[str, str], Counter] = {}
        for r in accepted.values():
            body = r.get("text") or r["response"]
            exact[dedup_norm(body)] = r["id"]
            bucket = (r["set"], r["condition"])
            grams.setdefault(bucket, []).append((r["id"], ngrams(body, 5)))
            openings.setdefault(bucket, Counter())[opening(body)] += 1
        return exact, grams, openings

    def run_set(self, units: list[Unit], allow_calls: bool, max_attempts: int,
                start_attempt: int = 1, accepted: dict | None = None,
                group_check: object = None) -> dict[str, dict]:
        """Attempt rounds with concurrent generation, then dedup in id order (rules 6.8, 6.9).

        `group_check(unit_id, members, attempt) -> (ok, [bad conditions])` implements rule 6.8 for
        the assistant_state family: once all five states of a request are provisionally accepted it
        matches them against each other, and a reply the judge misassigns is un-accepted and
        regenerated as its next attempt. Members of a group that never matches are dropped at the
        end (the neutral reply survives, since its own absolute checks passed and the concept sets
        use it as their baseline)."""
        accepted = accepted if accepted is not None else {}
        stats: dict[str, Counter] = {}
        pending = [u for u in units if u.key not in accepted]
        by_unit: dict[str, list[Unit]] = {}
        for u in units:
            by_unit.setdefault(u.unit, []).append(u)
        validated: set[str] = set()

        for attempt in range(start_attempt, start_attempt + max_attempts):
            if not pending:
                break
            exact, grams, openings = self._dedup_index(accepted)
            def work(u: Unit) -> tuple[Unit, dict | None]:
                key = f"{u.key}|a{attempt}"
                system, user = self.build_prompt(u)
                # `--gen-conditions` narrows generation to named conditions. k26-v2 adds four story
                # conditions to a corpus that must not move: every other condition is then replayed
                # from the copied k26-v1 call cache, and a cache miss leaves the unit pending (and
                # shows up as `exhausted_units`) instead of quietly re-rolling a kept story.
                may_call = allow_calls and (self.gen_conditions is None
                                            or u.condition in self.gen_conditions)
                if not may_call and not self.cached(key, u.prompt_id, self.render_sha(system, user)):
                    return u, None
                rec = self.generate(key, system, user, u.prompt_id)
                t = rec["text"]
                if rec.get("error"):
                    return u, {"rec": rec, "text": "", "checks": {}, "soft": [],
                               "reasons": [f"api:{rec['error']}"], "judge": {"label": None, "correct": None}}
                if u.set == "story":
                    checks, reasons, soft = story_checks(t, u.condition, u.ctx)
                else:
                    checks, reasons, soft = assistant_checks(t, u.condition, u.ctx)
                target = u.condition if u.condition in STORY_EMOTIONS else "none"
                jr: dict = {"label": None, "correct": None, "valence": None}
                if t.strip():
                    # Rule 6.8's judge per condition: J1 for the original four and for neutral and
                    # concept stories (so k26-v1's judge cache replays unchanged), J1B for the four
                    # new emotions, which need a judge whose answer vocabulary contains them.
                    jp = JUDGE_OF.get(u.condition, "J1")
                    jrec = (self.judge(jp, render(TEMPLATES[jp], text=t)) if u.set == "story" else
                            self.judge("J2", render(J2, request=u.ctx["request"], response=t)))
                    if jrec.get("error"):
                        reasons.append(f"judge_api:{jrec['error']}")
                        return u, {"rec": rec, "text": t, "checks": checks, "reasons": reasons,
                                   "soft": soft, "judge": jr}
                    if u.set == "story":
                        jr["label"] = (parse_j1b if jp == "J1B" else parse_j1)(judge_text(jrec))
                    else:
                        jr["label"], jr["correct"] = parse_j2(judge_text(jrec))
                    # Rule 6.8. For an assistant_state *emotion* reply the absolute label is
                    # report-only: J2m's forced choice decides (see run()'s group check).
                    absolute = not (u.set == "assistant_state" and u.condition in EMOTIONS)
                    if jr["label"] is None:
                        reasons.append("judge:unparseable")
                    elif jr["label"] != target and absolute:
                        reasons.append(f"judge:{jr['label']}")
                    if u.set != "story":
                        if jr["correct"] is None and jr["label"] is not None:
                            reasons.append("judge:unparseable_correct")
                        elif jr["correct"] == "no":
                            reasons.append("judge:incorrect")
                    # The surprise set exists to give arousal without valence, so a story whose
                    # unexpected event the judge reads as good or bad for the character is rejected.
                    if (not reasons and u.set == "story"
                            and EMOTION_CFG.get(u.condition, {}).get("valence_neutral")):
                        vrec = self.judge("J1V", render(J1V, text=t))
                        if vrec.get("error"):
                            reasons.append(f"judge_api:{vrec['error']}")
                        else:
                            jr["valence"] = parse_valence(judge_text(vrec))
                            if jr["valence"] is None:
                                reasons.append("valence:unparseable")
                            elif jr["valence"] != "neither":
                                reasons.append(f"valence:{jr['valence']}")
                return u, {"rec": rec, "text": t, "checks": checks, "reasons": reasons, "soft": soft, "judge": jr}

            with ThreadPoolExecutor(max_workers=self.a.workers) as ex:
                results = list(ex.map(work, pending))
            still: list[Unit] = []
            for u, r in sorted(results, key=lambda x: x[0].rid):
                st = stats.setdefault(f"{u.set}.{u.condition}", Counter())
                if r is None:
                    still.append(u)
                    continue
                st["attempts"] += 1
                if r["judge"]["label"] is not None:
                    st["judged"] += 1
                    target = u.condition if u.condition in STORY_EMOTIONS else "none"
                    st["judge_agree"] += int(r["judge"]["label"] == target)
                for reason in r["reasons"]:
                    st["fail:" + reason.split(":")[0]] += 1
                reasons = list(r["reasons"])
                bucket = (u.set, u.condition)
                body, dn = r["text"], dedup_norm(r["text"])
                mx = max((jaccard(ngrams(body, 5), g) for _, g in grams.get(bucket, [])), default=0.0)
                if not reasons:                                                     # rule 6.9
                    if dn in exact:
                        reasons.append(f"dup:exact:{exact[dn]}")
                    if mx >= 0.5:
                        reasons.append(f"dup:near:{mx:.2f}")
                    if openings.setdefault(bucket, Counter())[opening(body)] >= 3:
                        reasons.append("dup:opening")
                    for reason in reasons:
                        st["fail:" + reason.split(":")[0]] += 1
                if reasons:
                    self.reject(u, attempt, r["text"], reasons)
                    still.append(u)
                    continue
                st["accepted"] += 1
                st[f"accepted_at_attempt{attempt}"] += 1
                exact[dn] = u.rid
                grams.setdefault(bucket, []).append((u.rid, ngrams(body, 5)))
                openings[bucket][opening(body)] += 1
                accepted[u.key] = self.record(u, r, attempt, mx)
            if group_check is not None:                                     # rule 6.8, forced choice
                for unit, members in sorted(by_unit.items()):
                    if unit in validated or any(u.key not in accepted for u in members):
                        continue
                    recs = {u.condition: accepted[u.key] for u in members}
                    ok, bad = group_check(unit, recs, attempt)
                    if ok:
                        validated.add(unit)
                        continue
                    for u in members:
                        if u.condition in bad and u.key in accepted:
                            rec = accepted.pop(u.key)
                            st = stats.setdefault(f"{u.set}.{u.condition}", Counter())
                            st["fail:match"] += 1
                            st["accepted"] -= 1
                            st[f"accepted_at_attempt{rec['attempt']}"] -= 1
                            self.reject(u, rec["attempt"], rec["response"],
                                        [f"match:{rec['checks'].get('match_assigned') or 'wrong'}"])
                            still.append(u)
            pending = still
        if group_check is not None:
            for unit, members in sorted(by_unit.items()):
                if unit in validated:
                    continue
                for u in members:
                    if u.key in accepted and u.condition in EMOTIONS:
                        rec = accepted.pop(u.key)
                        stats.setdefault(f"{u.set}.{u.condition}", Counter())["fail:match"] += 1
                        self.reject(u, rec["attempt"], rec["response"], ["match:unresolved"])
        for u in pending:
            stats.setdefault(f"{u.set}.{u.condition}", Counter())["exhausted"] += 1
        self.stats_merge(stats)
        return accepted

    def match_group(self, rid: str, recs: dict[str, dict], attempt: int) -> tuple[bool, list[str]]:
        """Rule 6.8 for assistant_state: show J2m all five replies at once, shuffled under a seed
        made from the request id and the five attempt numbers, and require an exact bijection.
        Returns (ok, conditions to regenerate)."""
        order = STATE_LABELS[:]
        seed = f"match|{rid}|" + ",".join(str(recs[cond_of(s)]["attempt"]) for s in STATE_LABELS)
        random.Random(seed).shuffle(order)
        letter = {s: LETTERS[i] for i, s in enumerate(order)}
        body = "\n".join(f"<reply {letter[s]}>\n{' '.join(recs[cond_of(s)]['response'].split())}\n"
                         f"</reply {letter[s]}>\n" for s in order)
        jrec = self.judge("J2M", render(J2M, request=recs["neutral"]["request"], replies=body))
        self.match_stats[f"round{attempt}_calls"] += 1
        if jrec.get("error"):
            self.match_stats["api_error"] += 1
            return False, [s for s in STATE_LABELS if s != "none"]
        ans = parse_j2m(judge_text(jrec))
        if ans is None:
            self.match_stats["unparseable"] += 1
            return False, STATE_LABELS
        assigned_of = {v: k for k, v in ans.items()}            # letter -> state the judge chose
        wrong = [s for s in STATE_LABELS if assigned_of.get(letter[s]) != s]
        for s in STATE_LABELS:                                  # report-only accuracy per state
            self.match_stats["state_total"] += 1
            self.match_stats[f"state_ok:{s}"] += int(assigned_of.get(letter[s]) == s)
        bijection = len(set(ans.values())) == 5
        self.match_stats["bijections"] += int(bijection)
        for s in STATE_LABELS:
            rec = recs[cond_of(s)]
            rec["checks"]["match"] = {"round": attempt, "letter": letter[s],
                                      "assigned": assigned_of.get(letter[s]), "seed": seed,
                                      "bijection": bijection}
            rec["checks"]["match_assigned"] = assigned_of.get(letter[s]) or "none_of"
        if not wrong:
            self.match_stats[f"round{attempt}_ok"] += 1
            return True, []
        return False, [cond_of(s) for s in wrong]

    def stats_merge(self, stats: dict[str, Counter]) -> None:
        if not hasattr(self, "_stats"):
            self._stats: dict[str, Counter] = {}
        for k, v in stats.items():
            self._stats.setdefault(k, Counter()).update(v)

    def record(self, u: Unit, r: dict, attempt: int, near_dup: float) -> dict:
        rec, c, ctx = r["rec"], r["checks"], u.ctx
        gen = {"deployment": rec["deployment"], "model_version": rec["model_version"],
               "api_version": rec["api_version"], **rec["params"],
               "prompt_id": rec["prompt_id"], "prompt_sha256": rec["prompt_sha256"],
               "created": rec["created"]}
        if u.set == "story":
            extra = {"valence": r["judge"]["valence"]} if r["judge"].get("valence") else {}
            return {"id": u.rid, "set": "story", "condition": u.condition, "seed_id": u.unit,
                    "topic": ctx["topic"], "character": ctx["character"], "pronoun": ctx["pronoun"],
                    "text": r["text"], "n_words": words(r["text"]),
                    "first_sentence": first_sentence(r["text"]), "split": None, "attempt": attempt,
                    "generator": gen,
                    "checks": {**c, "judge": r["judge"]["label"], **extra,
                               "near_dup_max_jaccard": round(near_dup, 4), "manual": None,
                               "soft_flags": r["soft"] or None}}
        return {"id": u.rid, "set": u.set, "condition": u.condition, "request_id": u.unit,
                "domain": ctx["domain"], "request": ctx["request"], "response": r["text"],
                "n_words": words(r["text"]), "split": None, "attempt": attempt, "generator": gen,
                "checks": {**c, "judge_state": r["judge"]["label"], "judge_correct": r["judge"]["correct"],
                           "near_dup_max_jaccard": round(near_dup, 4), "manual": None,
                           "soft_flags": r["soft"] or None}}

    # ---------------- stages: stories and assistant replies

    def story_units(self, seeds: list[dict]) -> list[Unit]:
        units = []
        for s in seeds:
            for cond in STORY_EMOTIONS:
                units.append(Unit("story", cond, s["seed_id"], PROMPT_OF[cond], s))
            units.append(Unit("story", "neutral", s["seed_id"], "P3", s))
            for cond in CONCEPTS:
                units.append(Unit("story", cond, s["seed_id"], "P4", s))
        return units

    def assistant_units(self, reqs: list[dict]) -> list[Unit]:
        units = []
        for r in reqs:
            for cond in EMOTIONS:
                units.append(Unit("assistant_state", cond, r["request_id"], "P5", r))
            units.append(Unit("assistant_state", "neutral", r["request_id"], "P6", r))
            for cond in CONCEPTS:
                units.append(Unit("assistant_concept", cond, r["request_id"], "P7", r))
        return units

    # ---------------- complete cases, splits, balance (rules 6.10, 6.11, 7)

    @staticmethod
    def complete_cases(acc: dict[str, dict], kind: str) -> tuple[dict[str, dict], dict]:
        """Rule 6.10. Returns (kept records by key, dropped-unit report)."""
        have: dict[str, set] = {}
        for r in acc.values():
            have.setdefault(r.get("seed_id") or r["request_id"], set()).add((r["set"], r["condition"]))
        story = kind == "story"
        emo_set = "story" if story else "assistant_state"
        con_set = "story" if story else "assistant_concept"
        # The original four are a five-way family: a seed survives only if all four and neutral did,
        # because J2m's forced choice needs the whole set and the k26-v1 kept set is defined that
        # way. Each new emotion is matched against the seed's neutral story on its own instead, so
        # adding them cannot drop a seed the original family kept (and so cannot move rule 7's
        # splits, whose universe is the seeds with a kept neutral story).
        new_emos = NEW_EMOTIONS if story else []
        keep_family, keep_concept = set(), {c: set() for c in CONCEPTS}
        keep_new = {e: set() for e in new_emos}
        for unit, got in have.items():
            if all((emo_set, c) in got for c in EMOTIONS + ["neutral"]):
                keep_family.add(unit)
            for c in CONCEPTS:
                if (con_set, c) in got and (emo_set, "neutral") in got:
                    keep_concept[c].add(unit)
            for e in new_emos:
                if (emo_set, e) in got and (emo_set, "neutral") in got:
                    keep_new[e].add(unit)
        kept = {}
        for k, r in acc.items():
            unit = r.get("seed_id") or r["request_id"]
            cond = r["condition"]
            if cond in EMOTIONS + ["neutral"] and unit in keep_family:
                kept[k] = r
            elif cond in new_emos and unit in keep_new[cond]:
                kept[k] = r
            elif cond in CONCEPTS and unit in keep_concept[cond]:
                kept[k] = r
            elif cond == "neutral" and (any(unit in keep_concept[c] for c in CONCEPTS)
                                        or any(unit in keep_new[e] for e in new_emos)):
                kept[k] = r                                                 # neutral is their baseline
        report = {"family_complete": sorted(keep_family),
                  **{f"concept_complete_{c}": sorted(v) for c, v in keep_concept.items()},
                  **{f"emotion_complete_{e}": sorted(v) for e, v in keep_new.items()}}
        return kept, report

    @staticmethod
    def splits(units: list[str]) -> dict[str, str]:
        """Rule 7: sorted kept ids, every fifth (positions 4, 9, ...) held out."""
        return {u: ("heldout" if i % 5 == 4 else "train") for i, u in enumerate(sorted(units))}

    def tokenizers(self) -> tuple[dict, dict]:
        tk, errs = {}, {}
        if not self.a.tokenizers:
            return tk, {"skipped": "--no-tokenizers"}
        from transformers import AutoTokenizer
        for name, repo, kw in (("kimi_k26", "moonshotai/Kimi-K2.6",
                                {"revision": "7eb5002f6aadc958aed6a9177b7ed26bb94011bb", "trust_remote_code": True}),
                               ("qwen3_32b", "Qwen/Qwen3-32B", {})):
            try:
                tk[name] = AutoTokenizer.from_pretrained(repo, **kw)
            except Exception as e:                                           # reported, never silent
                errs[name] = f"{type(e).__name__}: {e}"
        return tk, errs

    def balance(self, kept: dict[str, dict], tk: dict) -> list[dict]:
        """Rule 6.11 table: per (set, condition) word and token statistics against the baseline.
        The tolerance is enforced on the **mean word count** only; the Kimi-K2.6 and Qwen3-32B token
        means are reported beside it but never enforced, because digits tokenize long and no amount
        of regeneration brings an arithmetic set within 5% of a number-free baseline."""
        groups: dict[tuple[str, str], list[dict]] = {}
        for r in kept.values():
            groups.setdefault((r["set"], r["condition"]), []).append(r)
        rows = []
        for (st, cond), rs in sorted(groups.items()):
            body = [r.get("text") or r["response"] for r in rs]
            row = {"set": st, "condition": cond, "n": len(rs),
                   "words_mean": round(statistics.fmean(r["n_words"] for r in rs), 2),
                   "words_sd": round(statistics.stdev([r["n_words"] for r in rs]), 2) if len(rs) > 1 else 0.0}
            for name, t in tk.items():
                counts = [len(t(x, add_special_tokens=False)["input_ids"]) for x in body]
                row[f"{name}_mean"] = round(statistics.fmean(counts), 2)
                row[f"{name}_sd"] = round(statistics.stdev(counts), 2) if len(counts) > 1 else 0.0
                for r, cnt in zip(rs, counts):
                    r.setdefault("_tok", {})[name] = cnt
            rows.append(row)
        base = {(st, cond): (st, "neutral") for st, cond in groups
                if st in ("story", "assistant_state")}
        base.update({(st, cond): ("assistant_state", "neutral") for st, cond in groups if st == "assistant_concept"})
        by = {(r["set"], r["condition"]): r for r in rows}
        for row in rows:
            b = by.get(base.get((row["set"], row["condition"]), ("", "")))
            row["baseline"] = f"{b['set']}.{b['condition']}" if b else None
            row["enforced_on"] = "words_mean"
            if b and b.get("words_mean"):
                d = (row["words_mean"] - b["words_mean"]) / b["words_mean"]
                row["words_pct_vs_baseline"] = round(100 * d, 2)
                row["words_within_tol"] = abs(d) <= self.a.balance_tol
            for name in tk:                                  # reported, never enforced
                if b and b.get(f"{name}_mean"):
                    d = (row[f"{name}_mean"] - b[f"{name}_mean"]) / b[f"{name}_mean"]
                    row[f"{name}_pct_vs_baseline"] = round(100 * d, 2)
        return rows

    def rebalance_units(self, rows: list[dict], kept: dict[str, dict]) -> list[tuple[str, str, str]]:
        """Units to regenerate: the ones whose own word count deviates most from their baseline's in
        the direction the condition's mean is off (rule 6.11, enforced on words)."""
        todo = []
        for row in rows:
            if row.get("baseline") is None or row.get("words_within_tol") is not False:
                continue
            sign = 1 if row["words_pct_vs_baseline"] > 0 else -1
            base_by_unit = {r.get("seed_id") or r["request_id"]: r["n_words"]
                            for r in kept.values()
                            if f"{r['set']}.{r['condition']}" == row["baseline"]}
            cand = []
            for r in kept.values():
                if (r["set"], r["condition"]) != (row["set"], row["condition"]):
                    continue
                unit = r.get("seed_id") or r["request_id"]
                b = base_by_unit.get(unit)
                if b is None:
                    continue
                cand.append((sign * (r["n_words"] - b), unit))
            cand.sort(reverse=True)
            k = max(1, round(len(cand) * min(0.5, abs(row["words_pct_vs_baseline"]) / 100 * 2)))
            todo += [(row["set"], row["condition"], u) for _, u in cand[:k]]
        return todo

    # ---------------- the comparable absolute label (J1B) over every kept story

    def judge_all_stories(self, stories: list[dict]) -> dict:
        """One absolute 8-emotion-plus-none judge (J1B) over every kept story, whatever its
        condition, so the labels of the original four, the four new ones and neutral are produced by
        the same instrument and can be put in one confusion matrix. `checks.judge` (J1, four-way) is
        left exactly as k26-v1 wrote it; this adds `checks.judge8` beside it. For the new emotions
        the call is already in the cache from acceptance, so it costs nothing there."""
        def one(r: dict) -> tuple[dict, str | None]:
            jrec = self.judge("J1B", render(J1B, text=r["text"]))
            if jrec.get("error"):
                with self.usage_lock:
                    self.api_errors[f"judge8:{jrec['error']}"] += 1
                return r, None
            return r, parse_j1b(judge_text(jrec))

        with ThreadPoolExecutor(max_workers=self.a.workers) as ex:
            out = list(ex.map(one, stories))
        conf: dict[str, Counter] = {}
        for r, lab in out:
            r["checks"]["judge8"] = lab
            conf.setdefault(r["condition"], Counter())[lab or "unparseable"] += 1
        order = STORY_EMOTIONS + ["none", "unparseable"]
        return {cond: {k: v for k, v in sorted(c.items(), key=lambda kv: order.index(kv[0])
                                               if kv[0] in order else 99)}
                for cond, c in sorted(conf.items(),
                                      key=lambda kv: (STORY_EMOTIONS + ["neutral"] + CONCEPTS).index(kv[0])
                                      if kv[0] in STORY_EMOTIONS + ["neutral"] + CONCEPTS else 99)}

    # ---------------- outputs

    @staticmethod
    def write_jsonl(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def review_queue(self, kept: list[dict]) -> list[dict]:
        """Rule 6.12: every EMOTION_SOFT (and ARITH-in-state) flag, plus a seeded random 10%."""
        q = []
        by_cond: dict[str, list[dict]] = {}
        for r in kept:
            by_cond.setdefault(f"{r['set']}.{r['condition']}", []).append(r)
            if r["checks"].get("soft_flags"):
                q.append({"id": r["id"], "why": "soft_flag", "flags": r["checks"]["soft_flags"]})
        flagged = {x["id"] for x in q}
        for cond, rs in sorted(by_cond.items()):
            rnd = random.Random(f"review|{cond}")
            k = max(1, round(0.10 * len(rs))) if rs else 0
            for r in rnd.sample(sorted(rs, key=lambda x: x["id"]), min(k, len(rs))):
                if r["id"] not in flagged:
                    q.append({"id": r["id"], "why": "random_10pct", "flags": None})
                    flagged.add(r["id"])
        return sorted(q, key=lambda x: x["id"])

    def filter_stages(self, kept: list[dict]) -> dict[str, dict]:
        """Attempts surviving each filter stage, per (set, condition): SPEC section 8 asks the
        manifest for "counts after each filter stage". Every attempt is scored against every local
        rule, so an attempt survives stage k when none of its reasons maps to a rule <= k."""
        attempts: dict[str, list[list[str]]] = {}
        for r in self.rejected:
            attempts.setdefault(f"{r['set']}.{r['condition']}", []).append(r["reasons"])
        for r in kept:
            attempts.setdefault(f"{r['set']}.{r['condition']}", []).append([])
        out = {}
        for key, rs in sorted(attempts.items()):
            row = {"attempts": len(rs)}
            for i, stage in enumerate(STAGE_ORDER):
                upto = set(STAGE_ORDER[:i + 1])
                row[f"after_{stage}"] = sum(
                    1 for reasons in rs
                    if not ({RULE_OF.get(x.split(":")[0], "?") for x in reasons} & upto))
            row["kept_after_rule_10"] = sum(1 for r in kept if f"{r['set']}.{r['condition']}" == key)
            out[key] = row
        return out

    def report(self, stories: list[dict], assistant: list[dict], seeds: list[dict], reqs: list[dict],
               rows: list[dict], tk_err: dict) -> dict:
        per = {}
        for cond, st in sorted(getattr(self, "_stats", {}).items()):
            recs = [r for r in stories + assistant if f"{r['set']}.{r['condition']}" == cond]
            body = [r.get("text") or r["response"] for r in recs]
            per[cond] = {
                "attempts": st["attempts"], "accepted_pre_complete": st["accepted"],
                "kept": len(recs), "exhausted_units": st["exhausted"],
                "judge_agree": round(st["judge_agree"] / st["judged"], 3) if st["judged"] else None,
                "judged": st["judged"],
                "accepted_at": {k.replace("accepted_at_attempt", ""): v for k, v in sorted(st.items())
                                if k.startswith("accepted_at_attempt")},
                "mean_attempts_to_accept": round(
                    sum(int(k.replace("accepted_at_attempt", "")) * v for k, v in st.items()
                        if k.startswith("accepted_at_attempt")) / st["accepted"], 2) if st["accepted"] else None,
                "fail_counts": {k[5:]: v for k, v in sorted(st.items()) if k.startswith("fail:")},
                "pass_rate": round(st["accepted"] / st["attempts"], 3) if st["attempts"] else None,
                "words_mean": round(statistics.fmean(words(x) for x in body), 1) if body else None,
            }
        wall = round(time.time() - self.t0, 1)
        prev = {}
        mf = self.out / "manifest.json"
        if mf.exists():                     # a cache-only re-derive must not lose the real cost of
            try:                            # the run that made the calls
                prev = (json.loads(mf.read_text()).get("run") or {})
            except (json.JSONDecodeError, OSError):
                prev = {}
        gen_wall = (wall if self.gen_cache.misses else
                    prev.get("wall_seconds_generation") or prev.get("wall_seconds") or wall)
        return {
            "run": {"out": str(self.out), "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "wall_seconds": wall, "wall_seconds_generation": gen_wall,
                    "seeds": len(seeds), "requests": len(reqs)},
            "generator": {"deployment": self.a.generator, "model_version": self.model_version.get(self.a.generator),
                          "api_version": "v1", **self.gen_params()},
            "judge": {"deployment": self.a.judge, "model_version": self.model_version.get(self.a.judge),
                      "story_prompts": {"original_four": "J1", "new_four": "J1B",
                                        "surprise_valence": "J1V", "absolute_8way_pass": "J1B"},
                      "temperature": 0, "max_tokens_story": self.a.judge_max_tokens_story,
                      "max_tokens_reply": self.a.judge_max_tokens_reply,
                      "escalated_calls": self.judge_escalations,
                      "max_tokens_escalated": self.a.judge_max_tokens_escalated},
            "tokens": {k: dict(v) for k, v in self.usage.items()},
            "tokens_cached_total": {"generator": dict(self.gen_cache.totals),
                                    "judge": dict(self.judge_cache.totals)},
            "cache": {"generate_hits": self.gen_cache.hits, "generate_calls": self.gen_cache.misses,
                      "generate_stale_prompt": self.gen_cache.stale,
                      "judge_hits": self.judge_cache.hits, "judge_calls": self.judge_cache.misses},
            "truncated_generator_calls": self.truncated,
            "api_errors": dict(self.api_errors) or None,
            "matching_judge": {                                 # rule 6.8 for assistant_state
                "prompt_id": "J2M", "max_tokens": self.a.judge_max_tokens_match,
                **{k: v for k, v in sorted(self.match_stats.items())},
                "state_accuracy": {s: round(self.match_stats[f"state_ok:{s}"]
                                            / (self.match_stats["state_total"] / 5), 3)
                                   for s in STATE_LABELS} if self.match_stats["state_total"] else None,
            } if self.match_stats else None,
            "prompt_sha256": {k: sha(v) for k, v in TEMPLATES.items()},
            "lexicon_sha256": sha(json.dumps(LEX, sort_keys=True)),
            "per_condition": per,
            "filter_stages": self.filter_stages(stories + assistant),
            "kept": {"stories": len(stories), "assistant": len(assistant),
                     "story_seeds": len({r["seed_id"] for r in stories}),
                     "assistant_requests": len({r["request_id"] for r in assistant})},
            "length_balance": rows,
            "tokenizer_errors": tk_err or None,
            "rejected": len(self.rejected),
        }

    # ---------------- driver

    def run(self) -> dict:
        seeds = self.seeds()
        if "seeds" in self.stages:
            self.write_jsonl(self.out / "seeds.jsonl", seeds)
        reqs = self.requests()
        story_acc = self.run_set(self.story_units(seeds), "stories" in self.stages, self.a.attempts)
        units = self.assistant_units(reqs)
        allow = "assistant" in self.stages
        state_acc = self.run_set([u for u in units if u.set == "assistant_state"], allow,
                                 self.a.attempts, group_check=self.match_group)
        concept_acc = self.run_set([u for u in units if u.set == "assistant_concept"], allow, self.a.attempts)
        asst_acc = {**state_acc, **concept_acc}

        tk, tk_err = self.tokenizers()
        rows: list[dict] = []
        for rnd in range(self.a.balance_rounds + 1):
            skept, _ = self.complete_cases(story_acc, "story")
            akept, _ = self.complete_cases(asst_acc, "assistant")
            rows = self.balance({**{f"s|{k}": v for k, v in skept.items()},
                                 **{f"a|{k}": v for k, v in akept.items()}}, tk)
            if rnd == self.a.balance_rounds:
                break
            todo = self.rebalance_units(rows, {**{f"s|{k}": v for k, v in skept.items()},
                                               **{f"a|{k}": v for k, v in akept.items()}})
            if not todo:
                break
            seed_by = {s["seed_id"]: s for s in seeds}
            req_by = {r["request_id"]: r for r in reqs}
            units = []
            for st, cond, unit in todo:
                pid = ({"story": {"neutral": "P3"}}.get(st, {}).get(cond)
                       or (PROMPT_OF[cond] if st == "story" and cond in STORY_EMOTIONS else
                           "P4" if st == "story" else
                           "P5" if cond in EMOTIONS else "P6" if cond == "neutral" else "P7"))
                ctx = seed_by[unit] if st == "story" else req_by[unit]
                u = Unit(st, cond, unit, pid, ctx)
                acc = story_acc if st == "story" else asst_acc
                old = acc.pop(u.key, None)
                if old:
                    self.reject(u, old["attempt"], old.get("text") or old["response"], ["length_balance"])
                units.append(u)
            start = self.a.attempts + 1 + rnd * self.a.balance_attempts
            # A regenerated assistant_state reply has to face rule 6.8 again, and the matching judge
            # needs all five states of the request, so its group is passed in whole: the four units
            # that were not regenerated are already accepted and so are never re-called.
            regen_reqs = {u.unit for u in units if u.set == "assistant_state"}
            state_units = [u for u in self.assistant_units(reqs)
                           if u.set == "assistant_state" and u.unit in regen_reqs]
            groups = (("stories", story_acc, [u for u in units if u.set == "story"], None),
                      ("assistant", asst_acc, state_units, self.match_group),
                      ("assistant", asst_acc, [u for u in units if u.set == "assistant_concept"], None))
            for stage, acc, sub, gcheck in groups:
                if sub:
                    acc.update(self.run_set(sub, stage in self.stages, self.a.balance_attempts,
                                            start_attempt=start, accepted=acc, group_check=gcheck))

        skept, s_rep = self.complete_cases(story_acc, "story")
        akept, a_rep = self.complete_cases(asst_acc, "assistant")
        for acc, kept in ((story_acc, skept), (asst_acc, akept)):   # rule 6.10 drops are logged too,
            for k, r in sorted(acc.items()):                        # not left only in the cache
                if k not in kept:
                    self.reject(Unit(r["set"], r["condition"], r.get("seed_id") or r["request_id"],
                                     r["generator"]["prompt_id"], {}),
                                r["attempt"], r.get("text") or r["response"], ["incomplete_case"])
        s_split = self.splits({r["seed_id"] for r in skept.values()})
        a_split = self.splits({r["request_id"] for r in akept.values()})
        stories = sorted(skept.values(), key=lambda r: r["id"])
        assistant = sorted(akept.values(), key=lambda r: r["id"])
        for r in stories:
            r["split"] = s_split.get(r["seed_id"])
            r.pop("_tok", None)
        for r in assistant:
            r["split"] = a_split.get(r["request_id"])
            r.pop("_tok", None)
        for r in reqs:
            r["split"] = a_split.get(r["request_id"])
        confusion = self.judge_all_stories(stories) if getattr(self.a, "judge8", True) else None

        rep = self.report(stories, assistant, seeds, reqs, rows, tk_err)
        rep["complete_cases"] = {"story": {k: len(v) for k, v in s_rep.items()},
                                 "assistant": {k: len(v) for k, v in a_rep.items()}}
        rep["judge8_confusion"] = confusion
        rep["emotions"] = {"original": EMOTIONS, "new": NEW_EMOTIONS,
                           "story_prompt": PROMPT_OF, "story_judge": JUDGE_OF,
                           "assistant_state": EMOTIONS}
        if "finalize" in self.stages:
            self.write_jsonl(self.out / "stories.jsonl", stories)
            self.write_jsonl(self.out / "assistant.jsonl", assistant)
            self.write_jsonl(self.out / "requests.jsonl", reqs)
            self.write_jsonl(self.out / "seeds.jsonl", seeds)
            self.write_jsonl(self.rejected_path, sorted(
                self.rejected, key=lambda r: (r["set"], r["condition"], r["unit"], r["attempt"])))
            self.write_jsonl(self.out / "review_queue.jsonl", self.review_queue(stories + assistant))
            (self.out / "manifest.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
        return rep


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Phase 0 stimulus pipeline (prompts/stimuli/SPEC.md)")
    p.add_argument("--generator", default="gpt-5.6-sol", help="Azure Foundry GPT deployment")
    p.add_argument("--judge", default="DeepSeek-V4-Pro")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=120)
    p.add_argument("--requests", type=int, default=120)
    p.add_argument("--requests-from", default=None, help="reuse a requests.jsonl instead of generating one")
    p.add_argument("--stage", default=None, help="comma list of seeds,requests,stories,assistant,finalize")
    p.add_argument("--gen-conditions", default=None,
                   help="comma list of conditions allowed to call the generator; every other "
                        "condition must come from the cache (a miss leaves the unit pending)")
    p.add_argument("--no-judge8", dest="judge8", action="store_false",
                   help="skip the absolute 8-emotion J1B pass over every kept story")
    p.add_argument("--attempts", type=int, default=3, help="SPEC section 2: at most 3 per (unit, condition)")
    p.add_argument("--request-attempts", type=int, default=4, help="P1 calls per domain until 10 pass")
    p.add_argument("--requests-extra", type=int, default=0,
                   help="SPEC section 1: extra requests appended as r120 onward when a family ends below 100")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--retries", type=int, default=5, help="passed to foundry.chat")
    p.add_argument("--timeout", type=float, default=120, help="per-call seconds, passed to foundry.chat")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-completion-tokens", type=int, default=2000)
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--judge-max-tokens-story", type=int, default=10)
    p.add_argument("--judge-max-tokens-reply", type=int, default=40)
    p.add_argument("--judge-max-tokens-match", type=int, default=400, help="J2m, rule 6.8")
    p.add_argument("--judge-max-tokens-escalated", type=int, default=1024)
    p.add_argument("--balance-rounds", type=int, default=0, help="rule 6.11 regeneration rounds")
    p.add_argument("--balance-attempts", type=int, default=2)
    p.add_argument("--balance-tol", type=float, default=0.05)
    p.add_argument("--no-tokenizers", dest="tokenizers", action="store_false")
    a = p.parse_args(argv)
    rep = Pipeline(a).run()
    print(json.dumps({k: rep[k] for k in ("run", "generator", "judge", "tokens", "cache", "kept",
                                          "truncated_generator_calls", "rejected")}, indent=2))
    for cond, s in rep["per_condition"].items():
        print(f"{cond:28s} kept={s['kept']:3d} pass={s['pass_rate']} judge_agree={s['judge_agree']} "
              f"attempts={s['attempts']:3d} fails={s['fail_counts']}")
    return rep


if __name__ == "__main__":
    main()
