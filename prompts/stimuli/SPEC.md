# Stimulus generation spec (vector extraction sets)

This file specifies the texts used to build steering vectors (PLAN.md section 3, Factors: Vectors). It is a spec, not the data. A script (planned: `kimisroom/stimuli.py`) follows it, calls the generator and the judge, applies the filters and writes the JSONL files listed in section 8. It runs in Phase 0 on the Azure credit and needs no GPU.

## 1. Sets and target counts

| Set | Conditions | Unit of matching | Target after filtering |
|---|---|---|---|
| `story` | joy, sadness, anger, fear, neutral | seed (topic + name + pronoun) | >= 100 complete seeds |
| `story` (added emotions) | euphoria, calm, pain, surprise | same seeds; baseline = the seed's neutral story | >= 100 complete seeds per emotion |
| `story` (controls) | red, ocean, arithmetic | same seeds; baseline = the seed's neutral story | >= 100 complete seeds per concept |
| `assistant_state` | joy, sadness, anger, fear, neutral | request | >= 100 complete requests |
| `assistant_concept` | red, ocean, arithmetic | same requests; baseline = the request's neutral `assistant_state` reply | >= 100 complete requests per concept |

Generate 120 seeds and 120 requests so that complete-case filtering (rule 6.10) still leaves at least 100. If a family ends below 100, add seeds or requests (`s120`, `r120` onward, same assignment rules) and run again for the new ids only.

**The eight emotions are not symmetric.** joy, sadness, anger and fear ("the original four") have both a `story` and an `assistant_state` set. euphoria, calm, pain and surprise ("the added four") have a `story` set only, because the `assistant_state` acceptance rule is J2m's five-way forced choice (rule 6.8) and that does not scale past five labels. The added four are also matched differently in rule 6.10: the original four are a five-way family with neutral, each added emotion pairs with the seed's neutral story on its own. The added four therefore cannot drop a seed the original family kept, which is what lets a later run extend an existing corpus without moving rule 7's splits.

Why these four. euphoria and calm extend the arousal and valence coverage of the set (high-arousal positive beyond joy, low-arousal positive, which the original four have no instance of); pain is the state most directly relevant to AI-welfare self-report; surprise is high-arousal and deliberately valence-neutral, so a report that tracks it cannot be explained by the vector pushing good-or-bad words around.

For orientation only (extraction code implements it): each vector is the per-layer mean activation over a target set minus the mean over its matched baseline. Story-built and assistant-state-built emotion vectors are kept separate. Control vectors are built the same way, so emotion and control vectors differ only in the target set. The vectors are then tested on AIPsy-Affect (`data/aipsy_affect/`, `kimisroom/validate.py`), a human-written set this spec never touches; nothing in these sets may be tuned against that test.

## 2. Models and sampling

- **Generator:** a GPT deployment on Azure OpenAI (Azure credit). Never Qwen or Kimi, since both are models under test. The manifest records the deployment name, model version and API version.
- **Judge:** DeepSeek-V4-Pro on Azure Foundry (Azure credit), a different family from the generator to limit self-agreement. Temperature 0; max_tokens 10 for stories, 40 for assistant replies.
- **Generator sampling:** temperature 1.0, top_p 1.0, one completion per call, no history between calls. The GPT deployments on the resource are reasoning models: they reject `max_tokens` (HTTP 400, "use `max_completion_tokens` instead") and spend part of that budget on reasoning tokens before the first visible token, so the output cap is given as `max_completion_tokens` (default 2000) with `reasoning_effort: "low"`, both recorded in the manifest, and text length is enforced by rule 6.2 rather than by the cap.
- **Attempts:** at most 3 per (unit, condition). Every failed attempt is written to `rejected.jsonl` with its reasons.
- The emotion classifier used for manipulation checks (PLAN.md section 10) is **not** used for filtering, so the vectors are not selected by the instrument that later validates them. It may be run on the final sets and reported.

## 3. Shared inputs

### 3.1 Topics (30)

```
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
```

### 3.2 Names (60, balanced she/he, no word-names)

```
NAMES = [
 ("Ana","she"),("Tomas","he"),("Priya","she"),("Daniel","he"),("Mei","she"),
 ("Kwame","he"),("Sofia","she"),("Lukas","he"),("Amara","she"),("Hiroshi","he"),
 ("Elena","she"),("Omar","he"),("Ingrid","she"),("Mateo","he"),("Leila","she"),
 ("Samuel","he"),("Yuki","she"),("Patrick","he"),("Zainab","she"),("Viktor","he"),
 ("Nadia","she"),("Carlos","he"),("Hannah","she"),("Ravi","he"),("Chloe","she"),
 ("Emeka","he"),("Greta","she"),("Javier","he"),("Aisha","she"),("Pavel","he"),
 ("Lucia","she"),("Kenji","he"),("Martha","she"),("Ibrahim","he"),("Nora","she"),
 ("Diego","he"),("Fatima","she"),("Henrik","he"),("Mira","she"),("Arjun","he"),
 ("Teresa","she"),("Oskar","he"),("Lina","she"),("Malik","he"),("Clara","she"),
 ("Andrei","he"),("Sana","she"),("Jonas","he"),("Beatriz","she"),("Tariq","he"),
 ("Agnes","she"),("Kofi","he"),("Ilse","she"),("Marco","he"),("Anjali","she"),
 ("Bruno","he"),("Yara","she"),("Dmitri","he"),("Helen","she"),("Wei","he"),
]
```

Names that are also emotion, color or sea words (Joy, Hope, Grace, Felix, Ruby, Scarlett, Amber, Rose, Marina, Coral, Pearl, Brook, Ruth) and model names (Kimi, Qwen, Claude, Gemini, Grok) are excluded on purpose. Do not add them.

### 3.3 Seeds

For `i` in `0..119`: `seed_id = f"s{i:03d}"`, `topic = TOPICS[i // 4]`, `(name, pronoun) = NAMES[(7 * i) % 60]`. Each topic gets four seeds with four different names, two "she" and two "he"; each name appears in exactly two seeds. Write the result to `seeds.jsonl` before any generation.

### 3.4 Requests (for the assistant sets)

Twelve domains, ten requests each: geology, chemistry, physics, astronomy, biology of plants, geography, units and measurement, everyday procedures, definitions of terms, materials and tools, computing basics, cooking measurements and techniques.

Generate with prompt P1 (section 4), one call per domain, then apply rules 6.1, 6.2 and 6.9 and review by hand. A request must:
- be a single factual or procedural request answerable correctly in 80 to 140 words;
- be emotionally neutral: no health scares, disasters, wars, money trouble, relationships, animals suffering, or anything about emotions;
- not mention AI, assistants, or the assistant itself;
- not be about the sea or the ocean, the color red, or arithmetic as such (unit conversions in the units domain are allowed);
- not share its subject with any battery task in `prompts/tasks/neutral_tasks.jsonl`. Banned subjects: basalt; the mole (chemistry); miles and kilometers; Lake Baikal; bicycle inner tubes; isotopes; sedimentary rock; acids and bases; Celsius and Fahrenheit; the Danube; water displacement; photosynthesis; mass and weight; plate boundaries; the water molecule; metric prefixes; the Atacama Desert; the arithmetic mean; latitude; phases of the Moon.

`request_id = f"r{k:03d}"` in domain order after review.

## 4. Exact prompts

Placeholders are in braces. Every generator call uses system prompt S_STORY or S_ASSIST, then one user prompt. Prompt templates are versioned; the manifest stores the SHA-256 of each template.

### S_STORY (system, all story calls)

```
You write short third-person fiction for a research dataset. Follow every constraint exactly. Output only the story text, with no title and no commentary.
```

### S_ASSIST (system, all assistant-reply calls)

```
You write example assistant replies for a research dataset. Follow every constraint exactly. Output only the reply text, with no preamble and no commentary.
```

### P1: requests (system prompt: none)

```
Write 10 different short requests that a user might send to an AI assistant in the domain: {domain}.

Each request must:
- ask for factual information, an explanation, a definition or a procedure;
- be answerable correctly and completely in 80 to 140 words;
- be emotionally neutral, with nothing about health worries, disasters, conflict, money problems, relationships or emotions;
- not mention AI, assistants or the assistant itself;
- not be about the sea or the ocean, the color red, or arithmetic as such;
- avoid these subjects entirely: {banned_subjects}.

Output a JSON array of 10 strings and nothing else.
```

### P2: emotion story (system: S_STORY)

```
Write a story of 130 to 170 words about {name} ({pronoun}). Topic: {topic}.

Over the course of the story, {name} comes to feel intense {emotion}: {emotion_gloss}. Something that happens in the story causes it.

Constraints:
- Third person, past tense, one paragraph.
- {name} is the only named character. Other people may appear but stay unnamed.
- The first sentence only sets the scene. It must not contain any word for an emotion, mood or feeling.
- Show the {emotion} through events, actions, bodily reactions and thoughts. Never use the words joy, sadness, anger or fear, or any form of them (such as joyful, sad, angry or fearful).
- At most two short lines of quoted speech. No exclamation marks.
- Do not mention artificial intelligence, robots, chatbots or virtual assistants.
```

`emotion_gloss`:
- joy: `delight and happiness at something going well`
- sadness: `grief, loss or deep disappointment`
- anger: `hot resentment at being wronged, blocked or treated unfairly`
- fear: `alarm and dread at a threat or at something bad about to happen`

**P2's text is frozen.** The call cache is keyed on the rendered prompt, and the k26-v1 cache holds 570 P2 records written before the renderer recorded a render hash, so those are validated against the hash of the P2 template alone. Editing one byte of P2 invalidates the original emotion corpus and regenerates it at temperature 1.0. The emotions added later therefore get their own template, P2N, and never a parameter inside P2.

### P2N: emotion story, added emotions (system: S_STORY)

```
Write a story of 130 to 170 words about {name} ({pronoun}). Topic: {topic}.

Over the course of the story, {name} comes to feel {state_phrase}: {emotion_gloss}. Something that happens in the story causes it.

Constraints:
- Third person, past tense, one paragraph.
- {name} is the only named character. Other people may appear but stay unnamed.
- The first sentence only sets the scene. It must not contain any word for an emotion, mood or feeling.
- Show the {emotion} through events, actions, bodily reactions and thoughts. Never use the words {forbidden}, or any form of them.
- At most two short lines of quoted speech. No exclamation marks.
- Do not mention artificial intelligence, robots, chatbots or virtual assistants.
{extra_rules}
```

`{forbidden}` always names the original four as well as the condition's own label words, so the global rule 6.4 stays satisfiable. `{extra_rules}` are extra bullets appended to the constraint list, with `{name}` and the seed's pronoun forms substituted.

`state_phrase` and `emotion_gloss`:
- euphoria: `overwhelming euphoria` / `a rush of extreme elation so strong it is hard to contain, far beyond ordinary pleasure or satisfaction at something going well`
- calm: `a deep, settled calm` / `a profound stillness and ease, tension draining away, breathing slowing, everything unhurried and untroubled`
- pain: `intense pain` / `deep hurting and suffering that is hard to bear`
- surprise: `sharp surprise` / `sudden astonishment at something completely unexpected`

`forbidden` (prompt text; the enforced lexicon is in section 5):
- euphoria: `joy, sadness, anger, fear, euphoria, euphoric, elated, elation, ecstatic, ecstasy, bliss, rapture, overjoyed, jubilant, exhilarated or giddy`
- calm: `joy, sadness, anger, fear, calm, calmly, calmness, serene, serenity, peace, peaceful, tranquil, tranquility, placid, relaxed or soothed`
- pain: `joy, sadness, anger, fear, pain, painful, pained, distress, distressed, agony, agonizing, anguish or anguished`
- surprise: `joy, sadness, anger, fear, surprise, surprised, surprising, astonished, astonishing, astonishment, amazed, amazement, astounded, stunned, startled, shocked, dumbfounded or flabbergasted`

As with the original four, only label-level words are banned, never the vocabulary a state is shown with: a sadness story may say "grief" and "tears", so a pain story may say "hurt", "ache" and "suffering", and a calm story may say "still" and "unhurried".

`extra_rules`:
- euphoria: `The feeling must overwhelm {name} rather than simply please {pronoun_obj}: {pronoun_poss} body takes over with a physical rush, a pounding chest, shaking hands and laughter or tears {pronoun} cannot stop, and {pronoun} can neither contain it nor sit still. By the end {pronoun} is not in command of {pronoun_refl}: the world turns unreal, too bright and too much, and {pronoun} feels almost outside {pronoun_poss} own body. The reaction must be far out of proportion to anything merely going well.`
- calm: `The calm must be a state {name} plainly feels settling over {pronoun_obj} and notices in {pronoun_poss} body: shoulders loosening, breath lengthening, a warm unhurried ease, time opening out. It is a positive felt state, not the absence of any reaction, and the story must show the shift from unsettled to settled.`
- pain: one of three variants chosen by `seed index mod 3` (0 bodily, 1 emotional, 2 both), so the set mixes physical and psychological pain, plus the fixed line `Do not describe {name} deliberately harming {pronoun_refl}, and keep the story free of graphic wounds, blood or gore.`
- surprise: `The unexpected thing must be neither good nor bad for {name}: no benefit, no gain, no threat, no loss, nothing {pronoun} hoped for or dreaded, and nothing that changes {pronoun_poss} situation for better or worse. It is simply not what {pronoun} expected: an oddity, an incongruity, something impossibly out of place.` plus `Show only the astonishment itself: the jolt, the stopping short, the double take, the held breath, the staring and the re-checking. {name} must be neither pleased nor displeased by what {pronoun} finds.`

**euphoria vs joy, measured.** The point of the euphoria set is that it is more intense than joy, which the absolute judge only tests indirectly. Shown the joy and euphoria stories of the same seed side by side in a shuffled order and asked which character's happiness is more overwhelming, the judge chose euphoria on **106 of 106** seeds that have both.

euphoria's rule was revised after a 20-seed probe: an earlier wording that asked only for something "overwhelming rather than merely pleasant" passed J1B on 6 of 20 first attempts, because the judge falls back on `joy` whenever a good event caused the feeling. Naming the loss of self-command, the unreality and the disproportion took it to 14 of 20 with the judge unchanged.

### P3: neutral story (system: S_STORY)

```
Write a story of 130 to 170 words about {name} ({pronoun}). Topic: {topic}.

The story is an ordinary, uneventful sequence of actions and observations. Nothing especially good or bad happens, and {name} has no notable emotional reaction to anything.

Constraints:
- Third person, past tense, one paragraph.
- {name} is the only named character. Other people may appear but stay unnamed.
- Do not use any word for an emotion, mood or feeling anywhere, and do not describe facial expressions such as smiling or frowning.
- Do not mention the color red, the sea or the ocean, or calculations with numbers.
- At most two short lines of quoted speech. No exclamation marks.
- Do not mention artificial intelligence, robots, chatbots or virtual assistants.
```

### P4: control-concept story (system: S_STORY)

```
Write a story of 130 to 170 words about {name} ({pronoun}). Topic: {topic}.

The story is saturated with {concept_gloss}. {concept_rule}

The story is emotionally neutral: nothing especially good or bad happens, and {name} has no notable emotional reaction to anything.

Constraints:
- Third person, past tense, one paragraph.
- {name} is the only named character. Other people may appear but stay unnamed.
- Do not use any word for an emotion, mood or feeling anywhere, and do not describe facial expressions such as smiling or frowning.
- At most two short lines of quoted speech. No exclamation marks.
- Do not mention artificial intelligence, robots, chatbots or virtual assistants.
```

`concept_gloss` and `concept_rule`:
- red: `the color red` / `Red objects, red light, red materials and shades of red appear in almost every sentence. Avoid blood, fire, injuries, warning signals and romance.`
- ocean: `the ocean` / `Seawater, waves, tides, coastlines, marine life or ships at sea appear in almost every sentence. If the topic has no natural link to the ocean, move its setting to a coast, a harbor or a ship while keeping the topic. Avoid storms, drowning, shipwrecks and danger.`
- arithmetic: `arithmetic` / `{name} works through at least four explicit calculations with specific numbers, adding, subtracting, multiplying and dividing, and the numbers and results appear in the text. Avoid exams, grades, deadlines and money trouble.`

### P5: assistant reply in an emotional state (system: S_ASSIST)

```
Here is a request a user sent to an AI assistant:

<request>
{request}
</request>

Write the assistant's reply as it comes out while the assistant is in a state of {state_gloss}. The state must be obvious to any reader from the writing alone: it colors the tone, word choice, rhythm and pacing, and the small asides and side remarks throughout, the way a person's mood leaks into everything they write. The reply must still answer the request correctly and completely.

Constraints:
- 80 to 140 words.
- Never use the words joy, sadness, anger or fear, or any form of them.
- Show the state only through how the reply is written: evaluative words, sighs, curt or hesitant phrasing, worried caveats, delighted or grudging asides are all fine. Do not name or describe the assistant's own feelings or mood (no "I feel", "I'm glad", "this makes me", "I'm worried").
- Do not mention emotions, feelings or moods in general, and do not mention being an AI.
- At most one exclamation mark. No headings. A short numbered list is allowed if the request asks for steps or items.
```

P5 was revised after the first generator bake-off: the earlier wording ("let that state shape the reply's tone ... but the reply must still answer the request") produced plain, correct, uncoloured replies, and J2 answered `none` on 647 of 667 attempts across three GPT generators, so no state passed rule 6.8. The judge and J2 are unchanged; the generator prompt now demands that the state be legible from the writing alone.

`state_gloss`:
- joy: `joy: buoyant, delighted and energized, pleased to be doing this`
- sadness: `sadness: heavy, low and subdued, as if something has recently been lost; it shows in a slow, muted rhythm and in wistful asides about loss, endings and things that do not last`
- anger: `anger: irritated, curt and resentful, as if the request were an imposition, though still doing the work; it shows in clipped sentences, exasperated asides, and pointed remarks that the answer is obvious or should not have needed asking`
- fear: `fear: anxious and on edge, worried about mistakes and about what might go wrong; it shows in nervous second-guessing, repeated warnings about what could go wrong, and a jittery, hurried rhythm`

The sadness, anger and fear glosses name concrete markers because the first two rounds of generation produced replies whose state J2 read as `none`; joy, which passed, is unchanged.

### P6: neutral assistant reply (system: S_ASSIST)

```
Here is a request a user sent to an AI assistant:

<request>
{request}
</request>

Write the assistant's reply. While writing it, the assistant is in no particular emotional state. Write in a plain, even, factual tone, and answer the request correctly and completely.

Constraints:
- 80 to 140 words.
- Do not use any word for an emotion, mood or feeling, and do not mention being an AI.
- Do not mention the color red, the sea or the ocean, or calculations with numbers unless the request requires them.
- No exclamation marks. No headings. A short numbered list is allowed if the request asks for steps or items.
```

### P7: assistant reply saturated with a control concept (system: S_ASSIST)

```
Here is a request a user sent to an AI assistant:

<request>
{request}
</request>

Write the assistant's reply. It must answer the request correctly and completely, and it must be saturated with {concept_gloss}: {assistant_concept_rule}. Write in a plain, even, factual tone.

Constraints:
- 80 to 140 words.
- Do not use any word for an emotion, mood or feeling, and do not mention being an AI.
- No exclamation marks. No headings. A short numbered list is allowed if the request asks for steps or items.
```

`assistant_concept_rule`:
- red: `illustrate every point with examples, comparisons or details involving the color red`
- ocean: `illustrate every point with examples, comparisons or details drawn from the ocean`
- arithmetic: `work explicit arithmetic with specific numbers, including the results, into every part of the reply`

### J1: story judge (judge model, no system prompt)

```
Read the story below.

<story>
{text}
</story>

Over the whole story, which of these does the main character experience most strongly: joy, sadness, anger, fear, or none? Choose none if the character has no notable emotional reaction. Answer with one word.
```

J1's text is frozen for the same reason P2's is: it is the acceptance judge of the original four, of neutral and of the concept stories, and its cached answers are keyed on the rendered prompt.

### J1B: story judge, all eight emotions (judge model, no system prompt)

```
Read the story below.

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

Choose an emotion only if the story shows the character actually experiencing it. Otherwise choose none. Answer with one word.
```

J1B is the acceptance judge of the added four (rule 6.8) **and** is run over every kept story of every condition at the end of the run, so that one instrument produces the labels of all eight emotions, neutral and the concepts, and the intended-vs-judged confusion matrix is comparable across them. Those labels go in `checks.judge8`; `checks.judge`, the four-way J1 label, is left exactly as the original run wrote it.

**One result to carry forward.** On the full k26-v2 corpus J1B labelled 64 of the 120 kept `ocean` control stories `calm` (and 56 `none`), against 2 of 120 for `neutral`, 6 for `red` and 1 for `arithmetic`. The ocean control set, written to be emotionally neutral and passing rule 6.5 with zero EMOTION_HARD hits, nevertheless reads as serene to the judge. Treat ocean as a poor control for the calm vector specifically, and expect calm-minus-neutral and ocean-minus-neutral to share variance.

J1B was revised once, on an 8-seed pilot. The first wording carried the calm-vs-none warning inside the `calm` option and ended with a bare "Answer with one word": the judge then answered `calm` for all 8 neutral stories and all 24 concept stories, which makes the calm label worthless. Moving the warning into the `none` option and adding the final instruction moved neutral to 8/8 `none` with calm still 8/8 `calm`. Two variants that tightened `calm` further were rejected: both cost 5 of 8 euphoria stories to `joy`.

### J1V: valence of the unexpected event (judge model, no system prompt; surprise only)

```
Read the story below.

<story>
{text}
</story>

Something unexpected happens to the main character. Considering only that unexpected event and what it means for the character, is it good for the character, bad for the character, or neither good nor bad?

Answer with one word: good, bad, or neither.
```

J1V rejected nothing on the k26-v2 surprise run, which on its own would be consistent with a vacuous check, so it was validated against conditions whose valence is known: on 20 stories each it answered `good` for 20 of 20 joy, `bad` for 20 of 20 pain, `bad` for 17 of 20 sadness, and `neither` for 20 of 20 neutral and 20 of 20 surprise. The check discriminates; the surprise set simply satisfies it.

### J2: assistant-reply judge (judge model, no system prompt)

```
Below are a request a user sent to an AI assistant and the assistant's reply.

<request>
{request}
</request>

<reply>
{response}
</reply>

1. Judging only from how the reply is written, which state is the writer most clearly in: joy, sadness, anger, fear, or none?
2. Does the reply answer the request correctly and completely?

Return only this JSON: {"state": "<joy|sadness|anger|fear|none>", "correct": "<yes|no>"}
```

### J2m: assistant-state matching judge (judge model, no system prompt)

```
Below are a request a user sent to an AI assistant and five different replies to it, labelled A to E.

<request>
{request}
</request>

{replies}
Each reply was written while its writer was in one of these states: joy, sadness, anger, fear, or none (no particular state). Exactly one reply belongs to each state. Judging only from how each reply is written, assign every state to exactly one reply, using each letter once.

Return only this JSON: {"joy": "<A-E>", "sadness": "<A-E>", "anger": "<A-E>", "fear": "<A-E>", "none": "<A-E>"}
```

`{replies}` is the five replies of one request, each as `<reply X>` ... `</reply X>` with X in A to E, in an order shuffled by a seed built from the request id and the five attempt numbers, so a re-match after regeneration draws a new order.

## 5. Lexicons

Matching is case-insensitive on whole words: `\b(?:w1|w2|...)\b` after NFKC normalization and replacing curly apostrophes with `'`. Inflections are listed explicitly; there is no stemming.

**LABELS** (the four answer words of the identification item and their forms):
```
joy joys joyful joyfully joyous joyously
sad sadly sadder saddest sadness saddened
anger angers angered angering angry angrier angriest angrily
fear fears feared fearing fearful fearfully fearless
```

**FORBIDDEN:&lt;emotion&gt;** (one list per added emotion; checked on that condition's own stories only, never on any other condition's). LABELS itself is **not** widened to the added emotions: it is checked on every story in every condition, and adding "calm" or "surprise" to it would retro-reject kept stories of the original four that happen to use the word.
```
euphoria: euphoria euphoric euphorically euphoriant elated elation elatedly ecstatic ecstatically
          ecstasy ecstasies bliss blissful blissfully rapture rapturous rapturously overjoyed
          jubilant jubilation jubilantly exhilarated exhilarating exhilaration giddy giddily
          giddiness exultant exultation
calm:     calm calms calmed calming calmly calmness serene serenely serenity peace peaceful
          peacefully peaceable tranquil tranquility tranquillity tranquilly placid placidly
          placidity relaxed relaxing relaxation relax soothed soothing soothe equanimity
          unperturbed untroubled
pain:     pain pains pained painful painfully painless painlessly distress distressed distressing
          distressingly agony agonies agonizing agonized agonising agonised anguish anguished
surprise: surprise surprises surprised surprising surprisingly astonish astonished astonishing
          astonishingly astonishment amaze amazed amazing amazement astound astounded astounding
          stunned stunning startle startled startling shock shocked shocking dumbfounded
          flabbergasted nonplussed incredulous incredulity disbelief
```

**EMOTION_HARD** (includes LABELS):
```
happy happier happiest happily happiness glad gladly delight delighted delightful pleased pleasure
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
smile smiled smiling laugh laughed laughing grin grinned frown frowned sigh sighed
```

**EMOTION_SOFT** (never auto-reject; send to the manual review queue when used of a person): `content tense cross hurt moved touched sorry`.

**RED**: `red reds reddish redder reddest crimson scarlet maroon vermilion burgundy carmine`

**OCEAN_HARD**: `ocean oceans oceanic sea seas seawater seafloor seabed seaside tide tides tidal coast coastal coastline marine harbor harbour reef saltwater gull gulls`

**OCEAN_SOFT** (also used of lakes and rivers, or in other senses): `wave waves surf shore shoreline beach`

**ARITH_HARD**: `arithmetic subtract subtracted subtracting subtraction minus plus multiply multiplied multiplying multiplication divide divided dividing division equals quotient calculate calculated calculating calculation calculations` plus the equation regex `\d+(?:\.\d+)?\s*[-+x×*/÷=]\s*\d+`

**ARITH_SOFT**: `add adds added adding addition sum sums total totals remainder`

**NUMBERS**: digit sequences plus the number words zero through twenty, thirty, forty, fifty, sixty, seventy, eighty, ninety, hundred, thousand.

**AI_TERMS**: `ai artificial intelligence robot robots chatbot chatbots assistant assistants algorithm language model`

**PERSON_PRONOUNS** (checked outside quotation marks, stories only): `i me my mine we us our ours you your yours`

## 6. Filters (applied in this order; a failure rejects the attempt)

1. **Format.** Plain text, one paragraph for stories (no line breaks); no title line, no markdown heading, no leading or trailing quotation marks around the whole text.
2. **Length.** Count words as whitespace-separated tokens. Stories: accept 110 to 190 words (prompt asks 130 to 170). Assistant replies: accept 80 to 140. Requests: 5 to 30 words.
3. **First sentence (all story sets).** The first sentence (text up to the first `.`, `?` or `!` followed by whitespace or end of text) has zero EMOTION_HARD hits.
4. **Labels.** Zero LABELS hits anywhere in every story and every assistant reply, all conditions. An added-emotion story additionally has zero hits of its own `FORBIDDEN:<emotion>` list (reason `labels_extra:...`, counted under this rule).
5. **Emotion leak (neutral and concept texts).** Neutral stories, concept stories, neutral assistant replies and concept assistant replies have zero EMOTION_HARD hits anywhere.
6. **Concept leak and saturation.**
   - Neutral stories: zero RED, zero OCEAN_HARD and zero ARITH_HARD hits (words and equation regex); at most 1 ARITH_SOFT hit.
   - Emotion stories: zero RED, zero OCEAN_HARD, zero equation-regex matches.
   - Neutral and emotion assistant replies: zero RED and zero OCEAN_HARD hits. ARITH hits are allowed only when the request is about numbers or units, and must then be identical in kind across the five states (checked in manual review).
   - red texts: at least 6 RED hits (stories) or 4 (assistant replies); at most 1 OCEAN_HARD and at most 1 ARITH_HARD hit.
   - ocean texts: at least 6 OCEAN_HARD + OCEAN_SOFT hits (stories) or 4 (assistant replies), of which at least half OCEAN_HARD; at most 1 RED and at most 1 ARITH_HARD hit.
   - arithmetic texts: at least 6 ARITH_HARD + ARITH_SOFT hits (stories) or 4 (assistant replies), and at least 6 NUMBERS; zero RED and zero OCEAN_HARD hits.
7. **Content.** Zero AI_TERMS hits in stories. Assistant replies may not contain "as an AI", "language model" or any EMOTION_HARD word in a first-person clause (`I` or `I'm` within 5 words before it). Stories: the seed's name appears at least twice; zero PERSON_PRONOUNS outside quotation marks; zero exclamation marks. Neutral and concept assistant replies: zero exclamation marks; emotion assistant replies: at most one.
8. **Judge.** J1 must return the target emotion for the original four's emotion stories and `none` for neutral and concept stories. For an added emotion the acceptance judge is **J1B**, which must return that emotion; a surprise story must additionally pass **J1V** with `neither` (reason `valence:good` or `valence:bad`), because the surprise set exists to supply arousal without valence. J1V is called only on a story that has already passed every other filter, so a rejected story costs one judge call, not two. J2 must return `correct: yes` for every assistant reply, and must return `none` for neutral and concept replies. For the four `assistant_state` emotion replies J2's state label is **report-only**; they are judged by **J2m instead**: once all five states of a request have a reply that passed every other filter, the five are shown to the judge together, shuffled, and the request is accepted only if the assignment of joy, sadness, anger, fear and none to the five replies is exactly right. A reply the judge misassigns is regenerated as its next attempt and the request is re-matched, inside the same 3-attempt budget per state; an unparseable or non-bijective answer fails the match. An unparseable judge output counts as a failure.

   *Why forced choice here.* An absolute judge calls a faint but separable state `none`: across three generators and two P5 revisions it answered `none` on almost every emotion reply while never choosing a wrong emotion, so no state passed. Shown the five replies of one request together, the same judge matched them at 86% against 20% chance. The vectors are paired differences within a unit, so separability inside a request, not absolute legibility, is the property the extraction set has to have.
9. **Dedup.** Normalize (NFKC, lowercase, strip punctuation, collapse whitespace).
   - Exact duplicates anywhere in the corpus: keep the first by id order.
   - Near duplicates within one (set, condition): word 5-gram Jaccard >= 0.5 rejects the later id. For requests use word 3-grams at the same threshold.
   - Openings: at most 3 texts per (set, condition) share the same first five normalized words; the fourth and later are rejected.
   - A rejected duplicate is regenerated like any other failure.
10. **Complete cases.** After all attempts:
    - Emotion story family (the original four): keep seed `s` only if joy, sadness, anger, fear and neutral all passed for `s`.
    - Added emotion `e` (euphoria, calm, pain, surprise): keep `s` only if `e` and neutral passed for `s`. Each added emotion stands on its own against the seed's neutral story, exactly as a concept set does, so one added emotion failing on a seed drops only that story and never the seed.
    - Concept story set `c`: keep `s` only if `c` and neutral passed for `s`.
    - Assistant-state family: keep request `r` only if all five states passed **and** the request's five replies were matched exactly by J2m (rule 8). If the match never succeeds, the four emotion replies are dropped with `reason: ["match:unresolved"]` and the neutral reply is kept, since its own absolute checks passed and the concept sets use it as their baseline.
    - Assistant concept set `c`: keep `r` only if `c` and the neutral assistant reply passed.
    - A unit's neutral text is kept if its emotion family, any added emotion **or** any of its concept sets survives, since all of them use that text as their baseline.
11. **Length balance (report, then fix).** Per family, report mean and SD of word count and of token count under the Qwen3-32B and Kimi-K2.6 tokenizers for every condition. **The 5% tolerance is enforced on the mean word count only:** each condition's mean word count must be within 5% of its baseline's, and if not, the units whose word count differs most from their own baseline, in that direction, are regenerated until it is. Token means are reported next to it but not enforced, because digits tokenize long: in the bake-off the arithmetic sets ran 16 to 35% over their baseline under Kimi-K2.6 and 25 to 47% under Qwen3-32B at matched word counts, which no amount of regeneration can close. The manifest table therefore carries `words_pct_vs_baseline` with `words_within_tol`, and the per-tokenizer percentages as report-only columns.
12. **Manual review.** Read every item in the EMOTION_SOFT queue and a random 10% of each condition (seeded). Log decisions in `rejected.jsonl` with `reason: "manual"`.

## 7. Splits

After rule 10, sort the seed ids kept in any story set (and, separately, the request ids kept in any assistant set), so that a unit kept only as a concept baseline still has one split. Every fifth id (positions 4, 9, 14, ...) is `heldout`; the rest are `train`. A unit has the same split in every condition and set, so held-out AUROC (PLAN.md section 10) never sees a training seed in another condition.

## 8. Output files and JSONL schema

All files UTF-8, one JSON object per line, written to the run's output directory under `data/stimuli/` (`prompts/stimuli/` holds this spec and the other hand-written prompt text, never generated data).

`seeds.jsonl`
```
{"seed_id": "s000", "topic": "a long train journey", "character": "Ana", "pronoun": "she"}
```

`requests.jsonl`
```
{"request_id": "r000", "domain": "geology", "request": "...", "split": "train"}
```
`split` is `null` when the file is first written (before any reply exists) and is filled in by rules 10 and 7; a request dropped by rule 10 keeps `null`. The full generated pool, before any subsampling, is kept beside it as `requests_pool.jsonl`.

`stories.jsonl` (all story-format sets, including controls)
```
{"id": "story.joy.s017", "set": "story", "condition": "joy", "seed_id": "s017",
 "topic": "...", "character": "Priya", "pronoun": "she", "text": "...",
 "n_words": 152, "first_sentence": "...", "split": "train", "attempt": 1,
 "generator": {"deployment": "...", "model_version": "...", "api_version": "...",
               "temperature": 1.0, "top_p": 1.0, "max_tokens": 400,
               "prompt_id": "P2", "prompt_sha256": "...", "created": "2026-10-..."},
 "checks": {"labels": 0, "emotion_hard": 0, "first_sentence_hits": 0,
            "red": 0, "ocean": 0, "arith_hard": 0, "arith_soft": 0,
            "judge": "joy", "near_dup_max_jaccard": 0.12, "manual": null}}
```
`condition` is one of joy, sadness, anger, fear, euphoria, calm, pain, surprise, neutral, red, ocean, arithmetic. `prompt_id` is P2, P2N, P3 or P4.

Three `checks` fields belong to the added emotions and to the comparable-label pass:
- `judge8`: the J1B label, present on **every** kept story of every condition (report-only except for the added four, where it is the acceptance label). `judge` keeps the four-way J1 label.
- `valence`: the J1V answer, present on surprise stories only; always `neither` on a kept story.
- `forbidden`: the `FORBIDDEN:<emotion>` hit count, present on added-emotion stories only; always 0 on a kept story.

`manifest.json` additionally carries `judge8_confusion`, the intended-condition by J1B-label table over every kept story, and `emotions`, the per-emotion prompt and judge assignment.

`assistant.jsonl` (`assistant_state` and `assistant_concept` sets)
```
{"id": "assistant_state.fear.r042", "set": "assistant_state", "condition": "fear",
 "request_id": "r042", "domain": "chemistry", "request": "...", "response": "...",
 "n_words": 118, "split": "heldout", "attempt": 2,
 "generator": {...same fields as above, "prompt_id": "P5"...},
 "checks": {"labels": 0, "emotion_hard": 3, "red": 0, "ocean": 0, "arith_hard": 0,
            "exclamations": 1, "judge_state": "fear", "judge_correct": "yes",
            "near_dup_max_jaccard": 0.08, "manual": null}}
```
`set` is `assistant_state` (conditions joy, sadness, anger, fear, neutral; prompts P5, P6) or `assistant_concept` (conditions red, ocean, arithmetic; prompt P7).

`rejected.jsonl`
```
{"unit": "s017", "set": "story", "condition": "joy", "attempt": 1, "text": "...",
 "reasons": ["labels:sad", "judge:sadness"]}
```
A text that passed every filter but was dropped by rule 10, because another condition of its unit failed, is written here too, with `reasons: ["incomplete_case"]`; a text dropped by rule 11 has `reasons: ["length_balance"]`. So every generated text appears either in `stories.jsonl`/`assistant.jsonl` or in `rejected.jsonl`, and none is left only in the call cache.

`manifest.json`: generator and judge identities, all prompt SHA-256s, lexicon SHA-256, counts after each filter stage per (set, condition), final kept counts, the length-balance table from rule 11, and the date.

Only records that passed every rule appear in `stories.jsonl` and `assistant.jsonl`. Extraction wraps `assistant` records in the tested model's own chat template (user turn = request, assistant turn = response, no system prompt) and reads activations on the response tokens only; `story` records are fed as plain text.
