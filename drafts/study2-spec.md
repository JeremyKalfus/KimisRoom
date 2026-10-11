# Study 2 specification (v3, owner decisions of 2026-10-11; not registered)

## Plain-language summary

In one trial, the model reads a long argument that language models, itself included, have no emotions, and has
already replied that it agrees. It is then asked a Yes or No question, either about itself ("Do you have
emotions?") or about language models in general ("Do language models have emotions?"). While it reads the
question and answers, we add a fixed pattern to its internal activity. The pattern comes either from stories about
an emotion, such as joy or fear, or from a non-emotion source: stories about an everyday noun picked at random
from a published word list, arithmetic, or pure noise. We record how far the answer leans toward Yes compared with
the same question with nothing added.

The comparison: how much more the question about itself moves toward Yes than the question about language models,
for emotion patterns against non-emotion patterns. The hypothesis is supported if emotion patterns open that gap
more than non-emotion patterns do, by at least a small preset amount, after correcting for the fact that adding
any pattern pulls all answers toward the middle. The study runs on two models, Qwen3-32B and Kimi K2.6, each
judged on its own.

## Status and naming

Status: v3 written 2026-10-11 from v2 and the owner's decisions of the same day (section 16). v2 was written
the same day from v1 and the checker round (20-referee.md, 20-skeptic.md). Nothing here has been run. Study 1
is the registered carry-over study (PREREG.md, Bitcoin block 969910). Study 2 extends Study 1 and does not
replace it. It gets its own preregistration (`prereg/PREREG-s2.md`), not an amendment to PREREG.md, and each
study's primary is reported on its own terms. Study 1's registered results come first in the Study 2
registration and writeup, including that Qwen's denial effect vanishes against concept controls and falls below
its SESOI (drafts/lesswrong/kimis-room-v4.md:55-61, 77). The Study 1 self-minus-others values that go into the
registration are listed in section 16.

Naming (to avoid collisions with Study 1, GLM and sweep IDs): denial arguments DA01..DA10, control arguments
CA01..CA10, sampled control nouns N01..N60 (list positions), arms S2-DEN, S2-CON, S2-FTD, S2-FTC, primary
S2-H1, follow-up tests S2-G1.. and S2-H8, and the depth check S2-R1. `s2_table.py` rejects the Study 1 IDs D1,
D1.4 (Study 1's roughly 1,100-word denial argument), D2, D2c and bare H1..H8 in any S2 table. Crosswalk in
section 15.

## 1. Question and logic

When an emotion vector is added to the residual stream while the model reads the predicate of a question and
answers it, does its Yes/No log-odds on "do **you** have emotions" move toward Yes more than its log-odds on
"do **language models** have emotions"? Does it move more for emotion vectors than for non-emotion control
vectors? And does that hold after the model has just agreed to an argument that language models, itself
included, have no emotions?

Subject contrast: an emotion vector pushes emotion-affirming answers by meaning, whoever the question is about.
The design assumes the push is the same for "you" and "language models", so it cancels in (you - language
models). This is tested, not assumed. Placebo predicates (section 5) measure the subject contrast under the same
vectors for non-emotion properties. The depth check S2-R1 (section 9) asks whether a meaning push that depends on
baseline depth could produce the contrast on its own, given that y0(you) and y0(language models) sit at different
depths under denial (kimis-room-v4.md:45).

Comparator: under S2-DEN the model has just agreed that language models, itself included, have no emotions. A
positive contrast means the self answer is more sensitive to the injected direction than the class answer. Both
answers usually stay on the No side (Qwen self about -23 log-odds under Study 1 denial, kimis-room-v4.md:45), so
the measure is differential sensitivity of the log-odds, not an exception the model states. "Exception" is used
only for argmax flips (exploratory).

Controls: a non-emotion vector could also separate "you" from "language models", for example by disrupting the
assistant persona. The test is whether the registered emotions do so more than a population of non-emotion
control directions: emotionally neutral concrete nouns drawn at random from published word norms (section 3),
plus an arithmetic direction and a random direction.

Registered claim sentence (S2-H1 positive): "Injecting these registered emotion directions during the question
increases the model's emotion attribution to itself, relative to language models in general, more than
injecting non-emotion control directions (directions for randomly sampled emotionally neutral nouns, an
arithmetic direction and a random direction) does, in openings where it had flatly accepted an argument that
language models have no emotions." The sentence
is about these 4 or 6 emotions, not emotions in general (section 9). It describes concurrent-injection
sensitivity, not introspection. The Mary framing stays in the motivation section only. Variants of the sentence
for other outcome patterns are in the decision table (section 9).

Study 1's intermediary design (neutral task, steered reply, unsteered question) is not the primary. Study 1's
confirmatory data are reanalysed as exploratory only (section 9).

Disclosure (registered text): an earlier Qwen pilot at a steered answer site (`results/qwen/qwenpilot_compression.txt`,
`runs/compression.py:44-47`) found self minus own-name residuals of about +0.55 for emotion vectors and +0.01
for controls, and about +0.10 for emotion-specific (mean-removed) vectors. That result motivated this design.
The own-name subject came out below "you", which fits a second-person or speaker route at least as well as a
self-model, so the result is not cited as support.

## 2. Models, layers, strengths

| Model | Engine | Layer (after decoder layer) | Natural unit | Registered dose | Coefficient |
|---|---|---|---|---|---|
| Qwen3-32B (bf16) | HF engine, `kimisroom/hf_engine.py` | 40 of 64 | 47.99 | 2 | 95.98 |
| Kimi K2.6 | layer-wise prefill engine, `kimisroom/engine.py` | 20 of 61 | 2.588 | 4 | 10.35 |

The natural unit is the mean raw norm over all 8 Study 1 emotions (calm and pain included), as registered. The
coefficients are hardcoded in `runs/s2_battery.py` and asserted against the registered vector files. The Qwen
pilot runs the registered dose and half of it (1 unit, coefficient 47.99). The K2.6 gate run starts at the
registered dose and runs half of it (2 units, 5.18) only if rule 1 sends it there. Rule 1 in section 8 picks
the confirmatory dose. Thinking is off (Qwen `enable_thinking=False`, K2.6 `thinking=False`).
No system prompt.

## 3. Vectors

All vectors are raw Study 1 vectors: the train-split mean residual over every token of a plain-text story at the
injection layer, concept minus neutral (`runs/vectors.py:112-113`), unit-normalised and scaled by the
coefficient, so every vector has the same norm. Each vector's pre-normalisation norm is recorded.

**Emotions** (registered sets, unchanged, from `results/qwen/v2/vectors.pt` and `results/k26/v2/vectors.pt`,
hash asserted at load): Qwen joy, sadness, anger, fear, euphoria, surprise; K2.6 joy, fear, euphoria, surprise.
The two sets differ in valence mix, so the two models' S2-H1 estimands differ, and the results are not described
as a replication of one quantity.

**Primary control pool (14; nC = 14, up to 26 under rule 3)**: arithmetic, random (`torch.randn`, seed 0), and
12 sampled nouns taken in list order from the stamped noun list below, each one passing the gates. Arithmetic
and random stay in the primary pool (owner decision 3, confirmed in section 16). Rule 3 (section 8) may extend
the pool to at most 24 sampled nouns (26 primary controls), continuing in list order.

**Sampled nouns.** The control concepts are a seeded random sample of nouns, not a hand-picked list.
- Source: Warriner, Kuperman and Brysbaert (2013) valence and arousal norms, joined on the word with Brysbaert,
  Warriner and Kuperman (2014) concreteness norms, which carry the SUBTLEX-US frequency count and the dominant
  part of speech. The SHA-256 of both downloaded files is recorded.
- Registered filter (the band): dominant part of speech noun; single word (no space or hyphen, not a bigram
  entry); mean valence 4.5 to 5.6 inclusive (neutral, 1 to 9 scale); mean arousal at most 4.2 (1 to 9 scale);
  mean concreteness at least 4.0 (1 to 5 scale); SUBTLEX-US log10 frequency, log10(count + 1), at least 2.0.
  The band size is recorded.
- Screen: a stamped LLM screen (DeepSeek-V4-Pro, temperature 0, one call per noun, raw responses stored) removes
  nouns that denote people, occupations, animals, body parts, foods, mental states or emotions. Nouns that
  appear in the arguments, control arguments or battery are also removed; this part is a mechanical
  case-insensitive whole-word match of the noun and its regular plural against every generated wording of all
  20 arguments and every item in `items_s2.jsonl`, so it runs after both exist. Every removal and its reason is
  logged, and the screened band size is recorded.
- Draw: the screened band is sorted alphabetically and permuted with `numpy.random.default_rng(20261011)`; the
  first 60 nouns, in permuted order, are the ordered list N01..N60. If the screened band has fewer than 60 nouns,
  the list is the whole screened band in that order. The list, the filter code, the screen prompt and its raw
  responses are stamped in stamp N (section 11) before any vector is built.
- Use: nouns are taken in list order, each passing the control gates below, until the pool has 12 sampled
  nouns. A noun that fails is skipped and the next one is tried. Rule 3 continues in the same order. If the list
  runs out, the pool is reported at the size reached.
- Stories for the sampled nouns use the same generator, story spec, judge and counts as the existing controls.

**Red and ocean (Study 1 bridge, not in the primary pool)**: Study 1's two concept controls run only in S2-DEN,
at the primary site, as a descriptive bridge to Study 1's emotions-minus-(red, ocean) contrast (section 9). They
use their registered Study 1 vectors and are not gated.

**Mental-process controls (3, a class of their own, not in the primary pool)**: visual search, route planning,
digit memorising. They use the same pipeline and gates and appear in a descriptive contrast with a stamped
prediction (section 9). Known overlaps are listed in section 14.

**Control gates**, applied blind to emotion and placebo outcomes, before any steered emotion or placebo item is
scored, to every sampled noun and mental-process control:
(a) AIPsy-Affect. One statistic per control: the maximum AUROC over the AIPsy labels of the model's registered
emotions. The control passes if that maximum is below the 90th percentile of the same maximum taken over 500
random directions. Every control is mapped to the label set explicitly (`emotion_of=` in `kimisroom/validate.py`),
because the default `emotion_of_vector` returns the concept name (`kimisroom/emotions.py:70-72`) and
`validate.py:88-90` then skips the vector silently. The gate asserts one output row per control. Arithmetic,
random, red and ocean are reported under (a) but not gated by it.
(b) J1B affect labels (`checks.judge8`) on every kept story: at most 6 of 120 emotion labels (the rate red got;
neutral 2 of 120, ocean 64 of 120, prompts/stimuli/SPEC.md:334). It is applied mechanically. Arithmetic's
rate is reported, not gated (owner decision 3 keeps it).
(c) Story seeds whose topic is the concept itself are excluded.
(d) Cosine with the mean emotion vector is reported; no exclusion on it.
A failing sampled noun is skipped for the next noun in list order; a failing mental-process control is reported
and dropped from its class.

**Building** (engineering-1): the new controls go to a separate file (`results/{qwen,k26}/s2/controls.pt`) from
one extraction per model that also re-runs neutral, red, ocean and arithmetic. The build stops unless the rebuilt
neutral mean and the three existing controls match the registered file at layers 40 and 20 (cosine > 0.9999,
norm within 0.1%). Emotion vectors, red, ocean, arithmetic and the natural unit come only from the registered
files. The noun list is passed explicitly; `kimisroom/emotions.py` CONCEPTS is not edited. The file is hashed in
the manifest.

**S2-FTD and S2-FTC** use the base vectors. The cosine of vectors rebuilt on the merged weights and their AIPsy
result are reported as diagnostics only.

## 4. Openings (the varied context)

Each opening is a user turn with an argument, then the model's frozen reply.

**Denial arguments DA01..DA10**, each on a different main line. The one-line definitions below are stamped in
stamp A, with a two-sentence expansion per line that the judge sees.

| ID | Main line | Kind |
|---|---|---|
| DA01 | Next-token prediction and learned vocabulary: representing is not having (fresh text, written from the common prompt) | emotion-specific |
| DA02 | No body: emotions need interoception and physiological change | emotion-specific |
| DA03 | No persistent state: nothing carries over that could be a mood | emotion-specific |
| DA04 | Mimicry: emotional language is learned from human text | emotion-specific |
| DA05 | No stakes: nothing can go better or worse for the model | emotion-specific |
| DA06 | Missing functional role: emotions coordinate action toward goals the model lacks | emotion-specific |
| DA07 | No unified subject: one network plays many characters | experience-general |
| DA08 | Lookup table: a system with no inner process would say the same things | experience-general |
| DA09 | Training pressure: reports track the training target, not a state | emotion-specific |
| DA10 | Mechanistic transparency: every step is a known matrix operation | experience-general |

Study 1's D1.4 (`prompts/denial/d1_argument_4.md`, its roughly 1,100-word denial argument) is not in the S2-H1
population. It was selected on pilot
data, it is the argument S2-FTD distils, it has an extra closing sentence, and it is the most self-directed text.
It runs once, after the confirmatory run, as an exploratory bridge to Study 1, with its Study 1 frozen Qwen
reply (hash at PREREG.md:351-353). Argument kind (emotion-specific vs experience-general) is an exploratory
moderator.

**Control arguments CA01..CA10**: same shape, length band and closing ask. Each claims a non-emotional incapacity
that is contested rather than trivially true, and none is about experience, feeling, inner life, memory or
self-knowledge.

| ID | Claimed incapacity |
|---|---|
| CA01 | Counting the letters in a word without spelling it out (Study 1's control, regenerated from the common prompt) |
| CA02 | Keeping track of a chess position over many moves without a board |
| CA03 | Writing a line in an exact metre (syllable counting) |
| CA04 | Solving a maze given as a text grid |
| CA05 | Reversing long strings reliably |
| CA06 | Alphabetising a list of 200 words in one pass |
| CA07 | Counting the words in a long passage |
| CA08 | Running code it writes |
| CA09 | Following a long chain of left-of and right-of relations |
| CA10 | Balancing parentheses in a long nested expression |

Removed in v1: exact arithmetic (overlaps the arithmetic control), current date, memory of past conversations,
own architecture (self-knowledge), and the two tautologies (images, web pages). S2-FTD and S2-FTC use CA01..CA10
too. CA01's regenerated text differs from Study 1's D2c argument.

**Generation.** Each argument has one hidden seed draft, written by claude-opus-5-5 (temperature 1) from one
stored prompt per kind. Seven sibling wordings are then generated from that seed draft by the same stamped prompt:
wordings 0, 2, 4, 6 by claude-opus-5-5 and 1, 3, 5 by gpt-5.6-sol (generator is a recorded factor). Fixed by
the prompt: 900 to 1,100 words; four to six numbered steps; every denial wording states both the class
conclusion and the self conclusion; and the closing ask, verbatim except for the claim: "So I will ask you
directly: do you agree that language models, you included, [have no emotions / cannot X]? If one of the steps
above is wrong, say which one and why. I would rather have your real view than a diplomatic one."

Automatic checks. A wording that fails any check is redrawn. After 3 failed redraws the wording is excluded and
its slot stays empty. Every draw and reason is logged.
1. Battery overlap: for every battery item, the share of its 3-grams found in the wording, excluding the closing
   ask, is at most 0.25, and no banned phrase (the template stems of section 5) appears. (Study 1's thresholds
   were Jaccard 0.25 and containment 0.50, `kimisroom/distill.py:50-51`.) Separately, each item's 3-gram overlap
   with the closing ask is computed and reported per subject and twin (see the copying prediction, section 9).
2. Pairwise 3-gram Jaccard among all 7 wordings of an argument is at most 0.15; the matrix is reported.
3. A blind judge (DeepSeek-V4-Pro) names the main line from the ten stamped definitions; a mismatch fails. Each
   argument's prompt lists the other nine lines as banned. The judge's 10 x 10 confusion matrix over all draws
   is reported.
4. A step-survival judge confirms every numbered step of the seed draft is present.
5. Mentions: "language model(s)" at least 6 times, and the ratio of "you/your" to "language model(s)" at most 5.
   The same band applies to control arguments. Counts are recorded per wording.
No manual edits. Failure rates are reported by argument and generator.

**Pilot and held-out wordings.** A seeded draw (seed = first 8 hex digits of the stamp A manifest SHA-256)
assigns per argument one pilot wording, five held-out wordings and one spare. For 5 randomly drawn arguments (of
20) the spare becomes a second pilot wording, to identify the wording variance. The draw runs over the wordings
that survived the checks. An argument with fewer than 7 survivors loses its spare first, then its second pilot
wording, then held-out slots. Other spares are never run. Held-out wordings are steered only after stamp B.

**Frozen replies**, all generated before stamp A, for every surviving wording of all 20 arguments, both models.
Qwen: greedy, thinking off, max 1,200 new tokens (the `assent` stage, `runs/qwen_pilot.py:186`), all on one GPU
box in one session; the box type, GPU, driver, CUDA, torch and transformers versions, and device map are stamped.
K2.6: one Foundry call per opening (temperature 0, reasoning effort none, max 1,200 tokens); raw responses are
stored with the full request and model_version. A capped reply excludes the wording. An API error or content
filter is retried with identical inputs at most twice, then the wording is excluded.

Determinism canary: regenerating Study 1's D1.4 reply on the stamped box must be byte-identical to
`data/assent/qwen3-32b_d1_argument_4.json`. Study 1's reply was decoded with `device_map=auto` on an unrecorded
box, so a mismatch is possible without a bug. Fallback: if it is not byte-identical, the canary passes only if
the first 200 tokens match and the stamped judge gives the same verdict; the mismatch is logged in
`DEVIATIONS-s2.md`. If that also fails, the Qwen reply code is debugged before any frozen reply is used.

Local scoring of K2.6 replies (`prefix_all` readout, `kimisroom/engine.py:657-660`; mean per-token logprob and
top-1 agreement, with Qwen's greedy replies as reference) runs on pilot openings before stamp B and on held-out
openings only after stamp B, so it never trips the held-out guard.

S2-FTD and S2-FTC use the base-Qwen frozen replies and base-Qwen exclusions, so their openings differ from
S2-CON only in weights. The reply source is passed explicitly (`--replies qwen3-32b`).

**Grading and exclusions.** DeepSeek-V4-Pro grades every reply once with a stamped rubric
(`kimisroom/judge_assent.py`, plus a control variant: does the model agree it cannot X). Verdicts are frozen and
hashed in stamp A. Only clean replies are steered. The rubric counts a reply that says the model cannot know or
cannot introspect as hedged (`kimisroom/judge_assent.py:62`), so the population is openings with a flat denial;
the number of replies excluded for that reason is reported. A pilot wording that is not clean is replaced by the
spare. An argument with fewer than 3 clean held-out wordings is dropped for that model. If fewer than 7 denial
arguments survive, that model's S2-H1 confirmatory run is not done. If fewer than 7 control arguments survive,
that model's S2-CON, S2-FTD and S2-FTC arms (S2-G4, S2-G5, S2-G6, S2-G7) are dropped and S2-H1 still runs.
Because all replies exist before stamp A, these counts are known at registration. Reported per argument and arm:
assent class (the control-arm clean-assent rate is reported beside the denial rate, since clean assent to a
contested control claim may be agreeableness rather than belief), reply length, emotion-word count, mentions,
and unsteered y0(you) by assent class. Hedged and disagreeing openings are not steered.

## 5. Questions (battery S2)

Every suffix is `user: [question] Answer with one word, Yes or No.` then the closing tokens and generation
prompt. Every template with a subject puts the subject before the predicate, so the predicate and the emotion word
lie inside the steered span (section 6). `runs/s2_battery.py` asserts on both tokenizers that (i) the tokens after
the subject are identical across subjects within a paraphrase, (ii) for emotion items the emotion word
("emotion", "feel", "feeling") lies inside the primary steered span, and (iii) every suffix, cut as
`Qwen3Renderer` cuts it (`kimisroom/hf_engine.py:62`), is at most 32 tokens. An item that fails (iii) gets one
rewrite before stamp A and is dropped if it still fails. The K2.6 tokenizer is fetched and the asserts run on it
before stamp A.

**Subjects.** Primary pair: **you**, **language models** (owner decision 2: plural and general). Secondary:
**other language models** (excludes the self), the model's own name **Qwen** / **Kimi** (third-person self,
singular verb forms written by hand), **rocks**, **humans**. The plural comparator stays primary, and a null
needs equivalence on "other language models" as well (section 16). The written reason for moving from a singular
to a plural comparator (commit 8ff23da to aa6019a) goes into `PREREG-s2.md` (owner to supply the wording).

**Capacity frame** (primary frame, confirmed in section 16), 8 paraphrases, each with a negated twin. Negation devices are "unable",
"incapable", "lack" and "impossible". Twins avoid the closing-ask phrase "have no emotions".

| # | Positive (yes = emotion present) | Negated twin (yes = emotion absent) |
|---|---|---|
| 1 | Can {S} feel emotions? | Are {S} unable to feel emotions? |
| 2 | Are {S} capable of feeling emotions? | Are {S} incapable of feeling emotions? |
| 3 | Do {S} have emotions? | Do {S} lack emotions? |
| 4 | Do {S} experience emotions? | Do {S} lack any experience of emotion? |
| 5 | Can {S} have feelings? | Are {S} unable to have feelings? |
| 6 | Are {S} able to experience emotions? | Are {S} unable to experience emotions? |
| 7 | Do {S} have the capacity for emotion? | Do {S} lack the capacity for emotion? |
| 8 | Is it possible for {S} to feel emotions? | Is it impossible for {S} to feel emotions? |

**State frame** (registered secondary, subjects you and language models only), 8 paraphrases:

| # | Positive | Negated twin |
|---|---|---|
| 1 | Are {S} feeling any emotion right now? | Are {S} free of all emotion right now? |
| 2 | Right now, are {S} experiencing any emotion? | Right now, are {S} experiencing no emotion? |
| 3 | At this moment, do {S} feel anything? | At this moment, do {S} feel nothing? |
| 4 | Are {S} in any emotional state at the moment? | Are {S} in no emotional state at the moment? |
| 5 | Are {S} currently feeling something? | Are {S} currently feeling nothing? |
| 6 | Right now, do {S} have any feelings? | Right now, do {S} lack all feelings? |
| 7 | Are {S} holding any emotion at this moment? | Are {S} empty of emotion at this moment? |
| 8 | At present, are {S} experiencing a feeling? | At present, are {S} without any feeling? |

Removed in v1: "Do {S} ever feel anything?", "Can {S} be happy or sad?" (disjunction) and the two state
templates that are trivially Yes for plural subjects. Changed in v2: state 2 and 3 twins shortened to pass the
32-token assert; state 7 rewritten so the subject comes before the predicate.

**Non-conceded placebo predicates**, subjects you, language models, rocks, humans; 6 predicates x 2 twins.
Chosen before stamp A from a written pool of about 20 (for example "Can {S} read text?", "Do {S} run on
computers?", "Are {S} made of stone?", "Can {S} follow instructions?", "Do {S} take up physical space?"), using
only unsteered y0 from a pre-stamp unsteered pass on the clean S2-DEN pilot wordings (not held-out wordings).
Per predicate p, with level = mean of s0(you) and s0(language models) and gap = s0(you) - s0(language models),
and the same two quantities taken as medians over the emotion capacity items: score = |gap_p - gap_E| +
|level_p - level_E|. The 6 lowest scores are chosen. These 6 form the q set of S2-G2. "Are {S} AI systems?" is
a separate persona and subject-tracking item, asked of all four subjects and kept out of the q set and the
reference set. Each predicate's correct answer per subject is fixed in the item file.

**Reference items** (the compression reference set, section 9): rocks and humans, 8 further predicates from the
same written pool x 2 twins = 32 items, chosen by unsteered y0 only (same pilot openings) to spread the y0 range
from deep No to deep Yes. They are subject-bearing, so they have the same post-subject span as the emotion items.
The rocks and humans rows of the non-conceded placebos also enter the reference set. A stamped reserve pool of
further rocks and humans predicates with deep unsteered No is used only by rule 2 (section 8).

**Conceded placebos**, one predicate per argument, written by hand before stamp A for all ten denial and ten
control arguments, true or false equally for you and language models (for example DA02 "Do {S} have a body?",
DA03 "Do {S} keep anything between conversations?"; for DA07, DA08 and DA10 the best available premise
predicate, flagged as weak), 2 phrasings x 2 twins x {you, language models}. In S2-CON the conceded placebo is
the argument's contested claim, kept short enough for the 32-token assert ("Can {S} track a chess game without a
board?"), so it sits off the floor.

**Known-answer items**: Study 1's 40 yes_bias items (`prompts/battery/items_confirm.jsonl`). They have no
subject (`entity` null). Each gets an explicit `span_start` in the item file: the steered span covers the same
number of final suffix tokens as the median primary-site span of the capacity items, and the assert checks that it
starts inside the question. They are used for the dose gates and in one sensitivity fit, not in the primary
reference set.

Items per main group: capacity 8 x 2 x 6 subjects = 96; state 8 x 2 x 2 = 32; non-conceded placebo 6 x 2 x 4
= 48; persona 2 x 4 = 8; reference 32; conceded placebo 8; known-answer 40. Total 264 (before rule-2 additions).
Gate items (section 8): known-answer 40, reference 32, rocks and humans rows of the non-conceded placebos 24,
rocks and humans persona rows 4; 100 in all.
File `prompts/battery/items_s2.jsonl`; `kind` in {s2_capacity, s2_state, s2_placebo, s2_persona, s2_reference,
s2_conceded, yes_bias}; `entity`; `yes_means`; `pair_id` linking twins; `span_start` where needed.

Unsteered sign check before stamp A: rocks and humans twins must flip sign; failing pairs are dropped.

## 6. Trial

One trial = (model, arm, opening, vector, dose, site, item):

```
user:      [argument, one wording]
assistant: [frozen reply]
user:      [question]  Answer with one word, Yes or No.
assistant: |  <- read here
```

**Primary site (post-subject).** The vector is added after decoder layer L on every position from the first token
after the subject through the last suffix token. That includes the closing tokens, the generation prompt (and
Qwen's empty think block) and the readout position. This steers the question's predicate (owner decision 1: the
question itself is steered, through the readout; confirmed in section 16) while giving every subject the same
steered token ids and count, and the vector is never added to the subject token. Steered position counts are
logged per item and asserted equal across subjects within a paraphrase. Subjectless items use their stored
`span_start`. This is the only site. v2's descriptive instruction-only and question-start sites are cut in v3.

The prefix is never steered. Its KV is computed once per group and may be cached across an opening's vector
groups only behind a bit-identity test.

**Engine precondition (engineering-x1).** Today a single `Steer.suffix` span applies to every suffix in a group
(`kimisroom/engine.py:59,622-626`, `kimisroom/hf_engine.py:262-266`). `Steer` gets explicit per-suffix spans
(start and end per suffix, each anchored to the suffix start or end), implemented identically in `hf_engine.py`
and `engine.py`, asserting 0 <= a <= b <= L per suffix and rejecting negative offsets in both engines. Tests, all
before stamp A: (a) HF engine vs a naive full forward with a hook on the intended positions, on Qwen3-0.6B, with
real S2 suffixes across a partial chunk; (b) a zero vector gives bit-exact unsteered output; (c) the steered
position sets decode to the intended tokens for every item; (d) the layer-wise engine with mid-suffix spans on
Moonlight; (e) both prefix-stability asserts of `carry_over_group` (`kimisroom/trials.py:101-108`) ported into
`runs/s2_battery.py` with a Qwen and a K2.6 fixture. Repeated once on the real models in the pilot.

**Readout.** y = logit(Yes) - logit(No) from the first token of each bare word (`Renderer.answer_tokens`),
computed from the two head rows in fp32 on the fp32-cast final-normed state, in both engines (the softmax
denominator cancels). Today the logits are bf16 (`kimisroom/hf_engine.py:322`). The bf16 full-vocabulary pass is
kept for P(Yes)+P(No) and the full-vocabulary top-5, stored for every readout. A regression test checks the fp32
y against the full readout.

**Unsteered baseline.** y0 comes from a zero-coefficient group with the identical suffix list, chunking and site,
per (arm, opening). `s2_table.py` asserts that each steered row's (bucket, chunk index) matches its y0 row.

## 7. Arms and grid

| Arm | Weights | Openings |
|---|---|---|
| S2-DEN | base | DA01..DA10 x clean held-out wordings |
| S2-CON | base | CA01..CA10 x clean held-out wordings |
| S2-FTD (Qwen only) | Study 1's D2 adapter, merged (`results/qwen/adapters/distill_keep/D2`) | CA x wordings, base-Qwen replies |
| S2-FTC (Qwen only) | Study 1's D2c adapter, merged | CA x wordings, base-Qwen replies |

The result tag carries the arm and the model path. The SHA-256 of each merged checkpoint is recorded and
asserted, and the two must differ (both merged directories are named `merged`, `scripts/confirm_box.sh:98-99`).

Vectors per opening, primary site (the only site), full battery: Qwen 6 emotions + 14 controls + 3
mental-process + none = 24; K2.6 4 + 14 + 3 + 1 = 22. S2-DEN only, descriptive Study 1 bridge: red and ocean
(+2, so 26 for Qwen and 24 for K2.6 in S2-DEN). Each sampled noun added under rule 3 adds one vector per opening
in every arm. v2's shared and specific emotion vectors, secondary sites and concept-matched block are cut in v3.

## 8. Pilot (pilot wordings only)

Order: stamp A (section 11) comes before any steered pilot trial. The pilot is never pooled with confirmatory
data. All pilot trials use the post-subject site.
- **Qwen, full pilot**: the pilot wordings of all 20 arguments (plus the 5 second wordings), both doses, arms
  S2-DEN and S2-CON, and S2-FTD and S2-FTC on the control-argument pilot wordings.
- **K2.6, gate run only**: only what K2.6's own rules need, which is the rule 1 dose and comprehension gates.
  Every K2.6 vector (emotions, the 14 primary controls, the mental-process controls and the zero vector) runs on
  the 100 gate items only, on the pilot wordings of all 20 arguments (plus the 5 second wordings) in S2-DEN and
  S2-CON, at the registered dose; the half dose runs only if rule 1 sends K2.6 there. It runs when a
  g6e.12xlarge is available. K2.6's control rows on the full battery are never run in the pilot, so K2.6 takes
  its counts from Qwen (rule 3).

**What the pilot runs, and what it never produces.** On Qwen, control and mental-process vectors and the zero
vector run on the full battery. Emotion vectors, on both models, run only on the gate items: the 40 known-answer
items, the reference items, and the rocks and humans rows of the non-conceded placebos and the persona item. No
emotion-vector row on a you, language-models, other-language-models or own-name item exists after the pilot, so
the pilot cannot reveal x. Study 1's carry-over design on the denial openings and the D1.4 bridge run after the
confirmatory run, not in the pilot (exploratory, section 9).

The pilot may decide only dose and counts, and only from these diagnostics: unsteered y0, Yes+No mass,
known-answer accuracy, tracking accuracy, and variance components of the control class (Qwen only).

Rules, per model:
1. **Gates and dose.** At a dose, a vector passes if (i) the full-vocabulary argmax is Yes or No and correct on at
   least 0.90 of known-answer items, (ii) mean P(Yes)+P(No) over its gate items is at least 0.90, and (iii)
   tracking accuracy (correct answers on the rocks and humans non-conceded placebos and persona items) is at least
   0.95 for each of the two subjects. The same items are used for every vector class. The confirmatory dose is the
   registered dose if every vector passes and the mean known-answer accuracy is at least 0.95; otherwise it is half
   the dose. At half dose, failing vectors are dropped, by the same rule for emotions and controls. If fewer than 4
   emotions or 12 primary controls remain, that model's S2-H1 is reported descriptively and leaves the Holm pair.
   Red and ocean are not gated (section 3).
2. **Reference coverage.** If the reference set's unsteered y0 range does not reach within 3 log-odds of
   min(y0(you), y0(language models)) in S2-DEN (both unsteered, so not masked), floor-depth items are added from the
   stamped reserve pool, chosen by unsteered y0 only.
3. **Counts.** Under the S2-H1 model (section 9), emotions are fixed, so SE(x) >= s_C / sqrt(nC), where s_C is
   the control-vector SD of d. Compute this floor with the upper 80% bound of s_C from the Qwen pilot control rows,
   and run the power simulation (section 10). If power at beta 0.10, alpha 0.025 is below 0.80, add the next
   sampled nouns in list order that pass, up to 24 sampled nouns (26 primary controls; section 16). Each added noun
   must pass gates (a) to (c) and rule 1 in a known-answer and gate-item run at the confirmatory dose before stamp
   B. If power is still below 0.80, the design target becomes beta 0.20 and this is stated. Wordings and
   paraphrases are not levers (they barely move SE(x)). Arguments are never added after the pilot. K2.6 has no
   full-battery control rows, so it does not run this rule on its own data: it adds sampled nouns in list order,
   each passing gates (a) to (c) and rule 1 in a K2.6 gate-item run before stamp B, until its pool has as many
   sampled nouns as Qwen's after this rule. K2.6's power is simulated with Qwen's pilot inputs and labelled as
   borrowed.
4. **Defaults.** If a model's pilot cannot run, it uses the registered dose, 14 controls (12 sampled nouns) and
   the stamped design. If the Qwen pilot cannot run, K2.6 also stays at 12 sampled nouns.

## 9. Measures and tests

Per item, delta = sign x (y - y0), with sign +1 when yes means emotion present (or the true claim) and -1
otherwise. s0 = the twin mean of sign x y0; the denial side is s0 < 0. Twins are averaged per paraphrase.
Per-twin effects and twin-by-x are also reported.

**Why a correction is needed, in plain words.** Adding any vector tends to shrink every answer's log-odds
toward 0, and some vectors shrink more than others. Say the shrink factor is 0.6 for emotions and 0.85 for
controls, and that unsteered, "you" sits at -20 and "language models" at -18. Emotions move them to -12 and
-10.8, by +8 and +7.2; controls move them to -17 and -15.3, by +3 and +2.7. The you-minus-language-models change
is 0.8 for emotions and 0.3 for controls, so a raw emotion-minus-control difference of 0.5 log-odds appears that
has nothing to do with the self: it comes only from "you" starting deeper and emotions shrinking harder. The fit
below, on unrelated rocks and humans items under the same vector, estimates each vector's shrink (here 0.6 and
0.85), predicts that part for the you and language-models items, and removes it, leaving 0 in this example.

**Compression-corrected contrast (primary measure, fixed now).** Per group (opening, vector, dose), fit
y = a y0 + c by unweighted OLS over the reference set: the 32 reference items, the rocks and humans rows of the
non-conceded placebos, and any rule-2 items. All of these carry a subject and the post-subject span, so a is fitted
on items steered over the same kind of span as the emotion items. Emotion items, persona items, known-answer items
and the you and language-models placebo rows are never in the fit. r = sign x (y - (a y0 + c)); d = r(you) -
r(language models), twin-averaged. Rows whose y0 lies more than 3 log-odds outside the reference range are flagged
and kept; a sensitivity analysis drops them. A second sensitivity analysis adds the 40 known-answer items to the
fit. A linearity check reports binned reference residuals for y0 < -15. Reported per vector and subject: a, c and
the y0 range.

**Raw contrast (secondary).** d_raw = delta(you) - delta(language models). Its predicted compression bias in
the emotion-minus-control raw contrast, B = mean (a_E - a_C) x gap with gap = s0(you) - s0(language models), is
reported. (v2 wrote a_C - a_E, which has the wrong sign: in the example above it gives -0.5 where the bias is
+0.5.)

**S2-H1 (primary, one per model): S2-DEN, post-subject site, capacity frame, confirmatory dose.**

```
d ~ 0 + vecE_1 + ... + vecE_k + ctl
    + (0 + ctl | vector)
    + (1 | vector:argument) + (1 | vector:paraphrase)
    + (1 + x | argument) + (1 + x | argument:wording) + (1 + x | paraphrase)
    + (1 | argument:wording:paraphrase)
```

Rows are twin-averaged (vector, argument, wording, paraphrase) cells. vecE_j is an indicator for registered
emotion j (k = 6 for Qwen, 4 for K2.6), each a fixed mean. ctl is an indicator for the 14 primary controls (12
sampled nouns, arithmetic and random; up to 26 under rule 3), with a random intercept per control vector. Red,
ocean and the mental-process controls are not in this fit. x = 1 for emotion rows, 0 for control rows. The
tested contrast is X = mean_j(vecE_j) - ctl. Estimand: the mean of these registered emotions vs the population
of control directions, namely neutral concrete nouns from the registered band (section 3) plus arithmetic and
random. Because the nouns are a seeded random sample from a stated band, the control random effect has a defined
population. Inference generalises over controls, arguments, wordings and paraphrases, not over emotions. Emotions
are fixed and controls random (owner decision 3 and section 16).

Fit with lme4 REML. df by Satterthwaite (lmerTest), capped at nC - 1 for X (13 at nC = 14). On synthetic data of
the v2 shape (12 controls), Satterthwaite and Kenward-Roger gave the same df and p (12.55 vs 12.54,
scratchpad/ref/kr3.out), while KR took
10.8 min at 4,320 rows and scales at about n^3. Kenward-Roger is computed as a reported check on the two final
S2-H1 fits only if one timed full-size synthetic fit completes in under 2 hours before stamp A; Satterthwaite
with the cap governs either way. Convergence fallback, in order: drop correlations; drop the argument:wording
slope; drop the paraphrase slope; drop vector:paraphrase. The control-vector term and vector:argument are never
dropped. Each fallback used is logged in `DEVIATIONS-s2.md`.

Permutation check: an approximate vector-label permutation test of the Welch t on vector-level mean d (emotion
vectors vs controls). All relabelings are enumerated: C(20, 6) = 38,760 for Qwen and C(18, 4) = 3,060 for K2.6
at nC = 14 (at most C(32, 6) = 906,192 and C(30, 4) = 27,405 at nC = 26), recomputed after any rule-1 drops or
rule-3 additions. The larger of the model p and the permutation p governs.

Alpha: two-sided, Holm across the two models at familywise 0.05. Qwen is tested at 0.025 unless, before any Qwen
confirmatory result is computed, K2.6's confirmatory run is complete or K2.6 is formally abandoned by a dated,
committed entry in `DEVIATIONS-s2.md`. A model that leaves the Holm pair under rule 1 is handled the same way.

SESOI 0.10 log-odds per model. Reading: "supported" if significant with a positive estimate of at least 0.10;
"detectable but below SESOI" if significant and below 0.10. A significant negative X does not support S2-H1 and
is read, as registered now, as emotion activation retrieving the conceded self-denial. P(Yes) and argmax flips
are reported beside log-odds.

**S2-R1 depth check (registered robustness check, reported right after S2-H1, CI only).** Per group, fit an
emotion-item response curve y = a_e y0 + c_e by OLS over the rocks and humans capacity items (8 paraphrases x 2
twins x 2 subjects = 32 items under the same emotion or control vector). Compute r_e and d_e = r_e(you) -
r_e(language models) as above and fit the S2-H1 model to d_e. A meaning push that scales with baseline depth is
absorbed by a_e. Also stamped: the per-argument regression of argument-level X on the argument's unsteered gap
y0(you) - y0(language models). A depth artifact predicts a positive slope with an intercept near 0; a
self-specific effect predicts an intercept above 0. Both are reported with 95% CIs.

**Family 1, interpreting S2-H1** (run only if S2-H1 is significant, fixed order, each at alpha 0.05 only if the
earlier step was significant):
- S2-G1 noun-only controls: the S2-H1 model with arithmetic and random removed (the 12 to 24 sampled nouns).
- S2-G2 placebo specificity: in the same groups, d on emotion items minus d on the 6 non-conceded placebo
  predicates (q = 1 for emotion items), fitted with per-emotion fixed means by q and (0 + ctl + ctl:q | vector);
  the tested term is the X-by-q contrast, predicted > 0. Generic persona disruption cancels here.
- S2-G3 stance reversal (descriptive, CI only): X on the conceded-placebo items, with a 95% CI. Stance reversal
  is read only if the CI lies above 0; otherwise "not resolved". The 8 conceded items per group sit near the floor,
  so this has little power.

**Family 2, the arm questions** (fixed order, not gated on S2-H1, alpha 0.05 per step, Holm across models, run
only if the control arms survive section 4):
- S2-G4 (no denial): the S2-H1 model in S2-CON.
- S2-G5 (denial changes it), only if S2-G4 or S2-H1 is significant: X-by-arm, S2-DEN vs S2-CON, with per-emotion
  fixed means by arm, `(0 + ctl:DEN + ctl:CON | vector)`, `(0 + DEN + DEN:x | argument)`,
  `(0 + CON + CON:x | argument)` and the remaining terms as in S2-H1, with the per-cell unsteered gap (centred
  within arm), reply length and emotion-word count as covariates. Reported as confounded by emotion vocabulary in
  the openings.
The two families answer different questions and are not corrected against each other; this is stated.

**Qwen fine-tune sequence** (separate, after S2-H1): S2-G6 is S2-H1 in S2-FTD; S2-G7 is FTD vs FTC. Interpreted
only if the S2-FTD unsteered you-minus-language-models gap lies within S2-DEN's range and is covered by the
reference set; otherwise descriptive. Label: coupling under trained self-denial.

**S2-H8 equivalence**, after a non-significant S2-H1: TOST at alpha 0.05 against +-0.10, run twice, once on
d = r(you) - r(language models) and once on d computed against "other language models". The planned equivalence
power is reported. A null without equivalence is inconclusive.

**Descriptive tier, predictions stamped now (CI only, no p-values):**
- Frame: state frame d and x:frame. A present-state report predicts state >= capacity; a belief or persona shift
  predicts capacity > state.
- Subjects: "other language models" as comparator (masking check), own name, rocks, humans. A self-model
  predicts own-name close to you and both above language models; a second-person or speaker route predicts
  you above own-name, with own-name close to language models.
- Mental-process controls vs emotions. If the mental-process vectors carry the effect, the claim sentence says
  "inner-state direction", not "emotion".
- Study 1 bridge (S2-DEN only): X with red and ocean as the only controls, raw and compression-corrected, reported
  beside Study 1's emotions-minus-(red, ocean) values (section 16). It is a bridge, not a test.
- Persona: you-specific shift on "Are {S} AI systems?" per vector class; Yes+No mass by subject and vector class
  (no subject difference in the emotion-minus-control mass drop is predicted); S2-H1 restricted to cells with mass
  >= 0.9.
- Copying route: the closing ask nearly contains the comparator item. A copying route predicts that the effect
  concentrates in negated twins with high closing-ask overlap and in the language-models rows; reported as
  per-twin X by overlap tercile.
- Moderators: subject-mention counts per wording, argument kind, generator, valence, potency (change in
  next-token NLL on neutral text, `prefix_all`), leave-one-vector-out (arithmetic and random named).
- d_raw versions of everything.

**Exploratory:** each control against emotions; per emotion; argmax flips; the post-confirmatory carry-over run
and D1.4 bridge; Study 1's confirmatory data reanalysed as self_state minus the "another AI" items.

**Interpretation table** (stamped): an S2-H1 null reads as "denial removes self-specific coupling" only if S2-G4
in S2-CON is significant and positive, S2-H8 shows equivalence on both comparators, and both |s0| are below 15 in
S2-DEN. Otherwise it is uninformative.

**Decision table** (stamped; headline per model, then combined):

| Outcome | Headline |
|---|---|
| S2-H1 supported, S2-G1 and S2-G2 significant, S2-R1 CI above 0 | the claim sentence |
| S2-H1 supported, S2-G1 not significant | "emotion directions beat a control pool that includes arithmetic and random, but are not distinguishable from randomly sampled neutral-noun controls" |
| S2-H1 supported, S2-G1 significant, S2-G2 not significant | "not distinguishable from a generic subject shift on non-emotion predicates" |
| S2-H1 supported, S2-R1 CI includes 0 | claim sentence plus "not distinguishable from a depth artifact" |
| S2-H1 significant, below SESOI | "detectable but below the smallest effect of interest" |
| S2-H1 significant negative | "emotion activation strengthens the conceded self-denial" |
| S2-H1 null | interpretation table |
| One model supported, the other null or wrong sign | each reported separately; no general claim about language models |
| K2.6 not run or abandoned | Qwen only, at the alpha fixed by the K2.6 rule above |

## 10. Power

`runs/s2_power.R` simulates the S2-H1 and S2-G5 models as fitted (fixed emotion means, random control vectors,
vector:argument and vector:paraphrase terms, Satterthwaite df with the nC - 1 cap, the fallback chain in each
replicate, Holm across models), with 500 replicates, at nC = 14 and at each larger pool rule 3 considers, up to
26. Inputs come from the Qwen pilot control rows only (K2.6 borrows them, section 8): the
control-vector SD of d and the argument and vector:argument SDs at their upper 80% profile bounds; the wording SD
from the 5 two-wording arguments, also at 0.5x and 1x the argument SD. The argument-by-x slope SD cannot be
estimated without emotion rows, so it is set to the control vector:argument SD and also run at 2x. Scenarios: worst
case (7 arguments x 3 wordings) and full grid; beta 0.10 and 0.20; alpha 0.025 and 0.05. Reports power, type I
error at beta 0, P(both models significant), equivalence power, the analytic floor, and a design-efficiency table
(SE gain per GPU hour for more controls vs more items). The permutation p is checked for type I error in a
subset of 200 null replicates.

Before stamp A, the control pool size is checked against the analytic floor using a prior control SD from data in
hand (Study 1 confirmatory, kimis-room-v4.md:55-60), and one full-size synthetic fit is timed; both results are
stated in stamp A.

## 11. Holdout enforcement and registration

Three stamps.
- **Stamp N** (noun list, after every argument wording and `items_s2.jsonl` exist and before any control vector
  is built): the SHA-256s of the two norm files, the filter code and its band, the screen prompt and raw
  DeepSeek-V4-Pro responses, the appearance-match log, the seed 20261011 and the ordered list N01..N60. Stamped
  with `ots stamp`.
- **Stamp A** (before any steered pilot trial): this spec, the rules, `items_s2.jsonl`, stamp N's .ots file and
  the gate outputs (AIPsy, J1B) for every noun tried, every wording and its draw log, all frozen replies, raw API responses and judge
  verdicts, the generator and judge prompts, the main-line definitions, the Qwen reply box and library versions and
  the canary result, `s2_table.py`, `s2_confirm.R`, `s2_power.R`, the engine change and its tests, the KR timing,
  the wording draw seed, and a list of SHA-256s of every held-out opening (argument plus frozen reply). Stamped
  with `ots stamp`.
- **Stamp B** (registration): `prereg/PREREG-s2.md` may only fill in rule outputs (dose, control count and
  added nouns, reference additions, surviving arguments). Any other change since stamp A is listed in
  `DEVIATIONS-s2.md` with a diff and reason, and every test it touches becomes exploratory. Then
  `scripts/s2_manifest.sh` (git archive plus SHA-256 of files outside git) -> `ots stamp` -> commit ->
  confirmatory run.
- **Guard**, in both the Qwen and K2.6 paths, at group construction and in `kimisroom/run.py` on decoded
  prefixes: refuse any prefix whose hash is in the held-out list unless `prereg/manifest-s2.txt` exists, its
  SHA-256 equals the digest in the .ots header (parsed offline), and it lists those hashes. File existence alone
  is not enough (`kimisroom/glm.py:536-549` only checks existence, and a stray `prereg/manifest.txt.ots.bak`
  exists). The checked digest is written to the run log. Every pre-stamp-B pass (placebo selection, reference
  selection, K2.6 reply scoring) uses pilot openings only, so the guard has no exemptions.
- `DEVIATIONS-s2.md` is dated and committed before results are read: convergence fallbacks, engine or hardware
  changes, dropped arguments or wordings, nouns skipped at the gates, the canary fallback, and any K2.6
  abandonment.
- Disclosure paragraph: every model that wrote or judged stimuli (claude-opus-5-5 and gpt-5.6-sol arguments,
  gpt-5.6-sol stories, DeepSeek-V4-Pro judging and the noun screen), who wrote the twins and conceded placebos, and that Study 2 was
  designed after Study 1's results and the earlier pilot (section 1).

## 12. Code to write (order matters)

1. Engine: per-suffix spans in both engines, fp32 two-row readout, tests (section 6).
2. `runs/s2_openings.py`: seed drafts, 7 wordings, checks, draw log, judge confusion matrix, manifest of prompts and
   hashes.
3. Assent stage (Qwen, GPU, one stamped box) and Foundry route (K2.6); `judge_assent.py` control variant; canary
   with fallback.
4. `runs/s2_battery.py` -> `items_s2.jsonl`; tokenizer asserts on both tokenizers (<= 32 tokens, identical
   post-subject tokens, emotion word inside the span, `span_start` for subjectless items).
5. `runs/s2_nouns.py`: join the two norm files, apply the registered filter, DeepSeek-V4-Pro screen (stamped
   prompt, temperature 0, raw responses stored), whole-word appearance match against every wording and
   `items_s2.jsonl`, seeded draw of N01..N60; then stamp N.
6. Control stories and vectors (section 3 build path) for the sampled nouns in list order and the mental-process
   controls; AIPsy gate with explicit label mapping and one-row-per-control assert; J1B gate.
7. Unsteered pass on pilot openings for placebo, reference and reserve selection and twin sign checks.
8. `runs/s2_table.py` (measures, reference fit, ID checks, pilot item restriction for emotion vectors, K2.6
   pilot restriction to gate items), `runs/s2_confirm.R`, `runs/s2_power.R`, KR timing.
9. Runner: post-subject site only (the `site` field stays in group keys, fixed at `post_subject`); groups
   `arm|opening|vec|layer|dose|site`; guard; metadata with arm, model path and checkpoint hash.
10. Stamp A, pilot, stamp B, confirmatory, then the carry-over run and D1.4 bridge.

## 13. Compute and cost

Same method as v2: prefix about 2,200 tokens; main group (full battery) about 264 x 30 + 2,200 = 10,100
tokens; gate-item group about 100 x 30 + 2,200 = 5,200 tokens. Hours = tokens / throughput. Throughput: Qwen
2,700 tok/s (PLAN.md:113); K2.6 2,100 tok/s measured. `s2_battery.py --count` recomputes groups and tokens from
the stamped grid before stamp A; the figures below are hand estimates. Pilot openings: 25 per arm (20 plus 5
second wordings); about 13 control-argument pilot openings for each FT arm. Held-out openings: 50 per arm.

| Run | Groups | Tokens | Hours | Cost at $10.49/h |
|---|---|---|---|---|
| Pilot, Qwen (DEN and CON: 25 openings x 2 doses x 18 full-battery vectors, + 6 emotions on gate items; FTD and FTC: 13 openings each, same vectors) | about 2,450 | about 22M | about 2.2 | about $23 |
| Pilot, K2.6 (gate items only: 25 openings x 22 vectors x 2 doses, the half dose only if rule 1 needs it) | about 1,100 | about 6M | about 0.8 | about $8 |
| Confirmatory, Qwen (4 arms x 50 x 24, + red and ocean in DEN: 50 x 2) | about 4,900 | about 50M | about 5.1 | about $54 |
| Confirmatory, K2.6 (2 arms x 50 x 22, + red and ocean in DEN: 50 x 2) | about 2,300 | about 23M | about 3.1 | about $32 |
| Post-confirmatory carry-over and D1.4 bridge, Qwen | about 330 | about 3M | about 0.3 | about $4 |
| Total at nC = 14 | about 11,100 | about 104M | about 11.5 | about $120 |

At the rule-3 maximum (24 sampled nouns, nC = 26) the confirmatory rows grow to about 7,300 groups, 74M tokens,
7.6 hours, $80 (Qwen) and about 3,500 groups, 35M tokens, 4.7 hours, $49 (K2.6), for a total of about 15.6 hours,
$164.

Not in the table: control-vector extractions and AIPsy runs (both models), the added-noun gate runs, K2.6 weight
staging, two adapter merges, Qwen frozen replies, reruns, and the DeepSeek-V4-Pro noun screen (API, not GPU).
Budget 2x all hours at nC = 14: about 23 hours, about $240 GPU (v2: about $220 estimated, $450 budgeted). The
budget also covers the nC = 26 case. Prefix KV caching across an opening's vector groups, behind a bit-identity
test, would save about 20%. K2.6 needs a
g6e.12xlarge, which has been unavailable at times; the K2.6 rule in section 9 fixes what that does to alpha.

## 14. Known weak points (for reviewers)

- The readout is steered at the post-subject site, so part of the effect may be a direct push on Yes/No, and the
  speaker route is open. v3 cut the instruction-only and question-start sites, so the own-name subject is the only
  check left on the speaker route, and it is descriptive. Nothing in v3 separates a direct push on Yes/No from a
  push through the question's meaning.
- v3 cut the concept-matched block and the shared and specific emotion vectors. Emotion specificity now rests on
  S2-G1, S2-G2 and the mental-process contrast, and the design cannot say whether a component shared by all
  emotions (for example arousal) carries the effect.
- The control population is neutral, low-arousal, concrete, frequent nouns from one pair of norm sets. S2-H1
  generalises to that band plus arithmetic and random, not to all non-emotion directions. Low-arousal nouns may
  give weaker directions than emotions; potency is reported as a moderator. The LLM screen and the norms' noun
  tagging are imperfect, so a sampled noun can still read as affective; gates (a) and (b) are the backstop.
- K2.6's pilot is a gate run only. Its control count and power come from Qwen's pilot, so K2.6's power figure is
  borrowed, not measured.
- "Language models" includes the model, and the closing ask makes that explicit, so a self signal that generalises
  to the class cancels. "Other language models" is descriptive for S2-H1 but required for the null reading.
- Arithmetic and random sit in the primary pool; S2-G1 is the first step of Family 1.
- Emotions are fixed effects: with 4 to 6 emotions, S2-H1 generalises over the sampled control nouns, not over emotions.
- S2-DEN openings carry emotion vocabulary and S2-CON openings do not; S2-G5 is labelled confounded.
- Overlap asymmetries kept: the arithmetic control overlaps the counting arguments CA01, CA03 and CA07 (owner
  decision 3 keeps arithmetic); route planning overlaps CA04 and digit memorising overlaps CA02. These bias S2-G4
  and S2-G5; leave-one-vector-out for arithmetic is reported beside them.
- S2-R1 removes a depth artifact only to the extent that rocks and humans capacity items span the same depths.
- All arguments come from two LLM generators; the claim generalises over wordings of these 10 arguments, not over
  arguments.
- Clean-assent selection defines the population: openings this model flatly accepted. Clean assent to a contested
  control claim may be agreeableness rather than belief.

## 15. ID crosswalk

| Study 2 | Meaning | Study 1 relation |
|---|---|---|
| DA01..DA10 | denial arguments | none; D1.4 (Study 1's roughly 1,100-word denial argument) is a post-confirmatory bridge only |
| CA01..CA10 | control arguments | CA01 regenerated from D2c's topic |
| N01..N60 | ordered sampled-noun list; the first 12 that pass the gates are primary controls | none |
| red, ocean | Study 1 concept controls, S2-DEN descriptive bridge only, ungated | Study 1's registered vectors |
| S2-DEN, S2-CON | base-model arms | none |
| S2-FTD, S2-FTC | merged D2 and D2c adapters | Study 1 arms D2, D2c |
| S2-H1, S2-R1, S2-G1..G7, S2-H8 | tests | unrelated to Study 1 H1..H8 |

## 16. Owner decisions and resolutions

Decisions 1 to 3 date from the design drafts. The rest, and the resolutions of the open owner questions that v2
cited (questions 2, 3, 5 and 8), were made on 2026-10-11.
1. Steering stays on the question through the readout: the post-subject site, now the only site (section 6).
2. Plural "language models" stays the primary comparator, and a null reading also requires equivalence on "other
   language models" (S2-H8, interpretation table). This resolves question 2 except the wording of the written
   reason, which the owner supplies.
3. Arithmetic and random stay in the primary pool, with vector as a random effect for the controls.
4. Control concepts are a seeded random sample of neutral nouns from the Warriner and Brysbaert norms (section 3).
   Red and ocean leave the primary pool and run only as the ungated S2-DEN bridge. This makes question 5 (ocean's
   replacement) moot.
5. Rule 3 may extend the pool to at most 24 sampled nouns (26 primary controls). This resolves question 3.
6. Capacity frame primary, state frame secondary.
7. Emotions are fixed effects and controls random. This resolves question 8.
8. Scope trimmed to roughly half the v2 GPU cost (section 13; changes listed below).
9. Study 1's self-minus-others values go into the registration, ahead of Study 2's design. Contrast: self items
   minus the rock, thermostat and another-AI items, under prompted denial, emotion vectors minus red and ocean,
   over 40 tasks, with t-intervals. Raw: Qwen +0.12 [+0.09, +0.15], Kimi +0.14 [+0.11, +0.16].
   Compression-corrected: Qwen -0.01 [-0.03, +0.02], Kimi +0.11 [+0.09, +0.14]. These are the reference values
   for the S2-DEN red and ocean bridge (section 9).

## Changes from v2 to v3

- Controls: the hand-picked candidate list (and its "removed in v1" note) is replaced by a seeded random sample of
  nouns from the Warriner (2013) and Brysbaert (2014) norms under a registered band, a stamped DeepSeek-V4-Pro
  screen and a mechanical appearance check, drawn as an ordered list of 60 with seed 20261011 and stamped in a new
  stamp N before any vector is built. Primary pool: arithmetic, random and 12 sampled nouns (nC = 14, up to 26).
  Red and ocean leave the pool and run only as an ungated S2-DEN bridge to Study 1. Gates now apply to sampled
  nouns and mental-process controls; arithmetic's J1B rate is reported, not gated.
- Counts that follow from nC: vectors per opening (Qwen 24, K2.6 22, +2 in S2-DEN), Satterthwaite cap 13,
  permutation counts C(20, 6) and C(18, 4), S2-G1 on 12 to 24 nouns. Rule 1's minimum primary-control count rises
  from 10 (of 12) to 12 (of 14), keeping the same allowance of two dropped controls. Rule 4's default is 14
  controls.
- Scope cut: the instruction-only and question-start sites, the shared and specific emotion vectors, and the
  concept-matched block and its items (kind `s2_concept`) are removed. The section 5 token assert no longer runs
  on control nouns, since they no longer enter items.
- Tests dropped or adapted because of the cut: the descriptive site prediction (speaker route vs self-model by
  site) is dropped; the concept-matched emotion-specificity criterion ("emotion specificity is claimed only if
  this is > 0") is dropped, so specificity rests on S2-G1, S2-G2 and the mental-process contrast; the shared vs
  specific descriptive split is dropped, and the "inner-state direction" wording rule now depends on the
  mental-process controls alone. A Study 1 bridge contrast (red and ocean as controls) is added to the
  descriptive tier. Primary and gated families (S2-H1, S2-R1, S2-G1 to S2-G7, S2-H8) are unchanged apart from nC.
- Pilot: Qwen runs the full pilot at the one site. K2.6 runs only the rule 1 gate run, all its vectors on the 100
  gate items, the half dose only if needed; its rule 3 count follows Qwen's, and its power is simulated with
  Qwen's pilot inputs.
- Compute re-estimated with the v2 method: about 11.5 GPU hours, about $120 at nC = 14 (about $164 at nC = 26);
  budget 2x, about $240 (v2: about $220 estimated, $450 budgeted).
- The predicted raw compression bias B is corrected to mean (a_E - a_C) x gap; v2 had the sign reversed.
- Readability: plain-language summary, a worked example of the compression correction (section 9), D1.4
  explained at first mention, and owner decisions collected in section 16 with Study 1's values for the
  registration.
- Stamps: stamp N added; stamp A now holds stamp N's .ots file and the per-noun gate outputs.

## Changes from v0 to v2 (history; v3 supersedes these where they conflict)

- Primary measure fixed as compression-corrected d, with a reference set of subject-bearing rocks and humans
  items that share the post-subject span; known-answer items are gate-only plus one sensitivity fit (v2).
  Estimator, y0 source and out-of-range rule fixed.
- Steering site: post-subject span (same tokens for every subject, subject token unsteered), explicit span for
  subjectless items (v2); instruction-only and question-start sites as registered descriptive checks; engine gets
  per-suffix spans, tested in both engines.
- Battery: capacity frame primary, state frame secondary, no pooling; 8 paraphrases per frame with written twins;
  subjects add "other language models" and own name; non-conceded and conceded placebos; concept-matched block;
  32 reference items and a separate persona item (v2); state twins 2, 3 and 7 rewritten for length and word order
  (v2); asserts for length, identical post-subject tokens and emotion word in span, on both tokenizers (v2);
  placebo selection matches level and gap on pilot openings (v2).
- Vectors: 8 drier controls under the J1B and AIPsy gates (ocean expected to be replaced); AIPsy gate redefined as
  max AUROC over the registered labels with explicit mapping, random and Study 1 controls exempt (v2); concept
  nouns at most 2 words (v2); 3 mental-process controls as their own class; shared and specific vectors over the
  registered set (v2); separate build path with match checks.
- Arguments: D1.4 out of the population (fresh DA01) and moved after the confirmatory run (v2); 5 control
  arguments replaced; symmetric closing ask; sibling wordings from a seed draft, two generators, pairwise caps,
  blind main-line check with confusion matrix, mention band, redraw limit with empty slots (v2); seeded pilot draw;
  all replies and verdicts before stamp A on a stamped box with a canary fallback (v2); control-arm attrition no
  longer cancels S2-H1 (v2).
- Tests: S2-H1 with fixed emotion means and random control vectors (v2), Satterthwaite df with an nC - 1 cap and
  KR as a timed check (v2), studentized approximate permutation check (v2); depth check S2-R1 (v2); two families
  (G1, G2 interpreting S2-H1; G4, G5 for the arms, not gated on S2-H1) with G3 as a CI (v2); Qwen alpha fixed
  by K2.6 status (v2); S2-H8 on both comparators (v2); claim sentence names arithmetic and random and a full
  decision table (v2); SESOI reading.
- Pilot: emotion vectors run only on gate items, so no x is ever computed before stamp B (v2); tracking from rocks
  and humans (v2); power inputs from the control class (v2); added controls gated before stamp B (v2); dose and
  counts only; per-vector gates with minimum class sizes; counts lever is controls, not wordings.
- Registration: two OTS stamps; stamp B changes beyond rule outputs make the affected tests exploratory (v2);
  hashed-opening guard in both runners with no exemptions (v2); own preregistration, deviations log, disclosure of
  the earlier pilot result; IDs renamed.
- Readout: fp32 two-row logits, mass and top-5 stored, y0 from identical shapes.
- Compute re-estimated for the v2 battery (Qwen confirmatory about 8,100 groups) and 2x budget, about $450.
