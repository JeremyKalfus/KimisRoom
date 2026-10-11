# Study 2 specification (v1, after two review rounds; not registered)

Status: v1 written 2026-10-11 from v0 and the board debate (R1-*, R2-*). Nothing here has been run. Study 1 is
the registered carry-over study (PREREG.md, Bitcoin block 969910). Study 2 is an extension of Study 1, not a
replacement: it gets its own preregistration (`prereg/PREREG-s2.md`), not an amendment to PREREG.md, and each
study's primary is reported on its own terms. Study 1's registered results, including that Qwen's denial effect
vanishes against concept controls and falls below its SESOI (drafts/lesswrong/kimis-room-v4.md:55-61, 77), are
stated first in the Study 2 registration and writeup.

Naming (to avoid collisions with Study 1, GLM and sweep IDs): denial arguments DA01..DA10, control arguments
CA01..CA10, arms S2-DEN, S2-CON, S2-FTD, S2-FTC, hypotheses S2-H1 and gated steps S2-G1.. . `s2_table.py`
rejects the Study 1 IDs D1, D1.4, D2, D2c and bare H1..H8 in any S2 table. A crosswalk table is in section 15.

## 1. Question and logic

When an emotion vector is added to the residual stream while the model reads the predicate of a question and
answers it, does its Yes/No log-odds on "do **you** have emotions" move toward Yes more than its log-odds on
"do **language models** have emotions", and more for emotion vectors than for non-emotion control vectors? Does
that hold after the model has just agreed to an argument that language models, itself included, have no
emotions?

Subject contrast: an emotion vector pushes emotion-affirming answers by meaning, whoever the question is about.
The design assumes that push is the same for "you" and "language models", so it cancels in (you - language
models). This assumption is tested, not assumed: placebo predicates (section 5) measure the subject contrast
under the same vectors for non-emotion properties, and the registered emotion-specificity steps (section 9) are
differences in differences against them.

Comparator: under S2-DEN the model has just agreed that language models, itself included, have no emotions. A
positive contrast means the self answer is more sensitive to the injected direction than the class answer. Both
answers usually stay on the No side (Qwen self about -23 log-odds under Study 1 denial, kimis-room-v4.md:45), so
the measure is differential sensitivity of the log-odds, not an exception the model states. "Exception" is used
only for argmax flips (exploratory).

Controls: a non-emotion vector could also separate "you" from "language models" (for example by disrupting the
assistant persona). The test is whether emotions do so more than a population of non-emotion concepts.

Registered claim sentence (S2-H1 positive): "Injecting these registered emotion directions during the question
increases the model's emotion attribution to itself, relative to language models in general, more than
injecting non-emotion concept directions does, in openings where it had cleanly accepted an argument that
language models have no emotions." This is concurrent-injection sensitivity, not introspection; the Mary framing
stays in the motivation section only.

Study 1's intermediary design (neutral task, steered reply, unsteered question) is not the primary. Study 1's
confirmatory data are reanalysed as exploratory only (section 9).

Disclosure (registered text): an earlier Qwen pilot at a steered answer site (`results/qwen/qwenpilot_compression.txt`,
`runs/compression.py:44-47`) found self minus own-name residuals of about +0.55 for emotion vectors and +0.01
for controls, and about +0.10 for emotion-specific (mean-removed) vectors. That result motivated this design.
Because the own-name subject was below "you", it fits a second-person or speaker route at least as well as a
self-model, and it is not cited as support.

## 2. Models, layers, strengths

| Model | Engine | Layer (after decoder layer) | Natural unit | Registered dose | Coefficient |
|---|---|---|---|---|---|
| Qwen3-32B (bf16) | HF engine, `kimisroom/hf_engine.py` | 40 of 64 | 47.99 | 2 | 95.98 |
| Kimi K2.6 | layer-wise prefill engine, `kimisroom/engine.py` | 20 of 61 | 2.588 | 4 | 10.35 |

The natural unit is the mean raw norm over all 8 Study 1 emotions (calm and pain included), as registered. The
coefficients are hardcoded in `runs/s2_battery.py` and asserted against the registered vector files. The pilot
runs the registered dose and half of it (Qwen 1 unit, coefficient 47.99; K2.6 2 units, 5.18); rule 1 in
section 8 picks the confirmatory dose. Thinking off (Qwen `enable_thinking=False`, K2.6 `thinking=False`). No
system prompt.

## 3. Vectors

All vectors are raw Study 1 vectors: train-split mean residual over every token of a plain-text story at the
injection layer, concept minus neutral (`runs/vectors.py:112-113`), unit-normalised and scaled by the
coefficient, so every vector has the same norm. Each vector's pre-normalisation norm is recorded.

**Emotions** (registered sets, unchanged, from `results/qwen/v2/vectors.pt` and `results/k26/v2/vectors.pt`,
hash asserted at load): Qwen joy, sadness, anger, fear, euphoria, surprise; K2.6 joy, fear, euphoria, surprise.
The two sets differ in valence mix, so the two models' S2-H1 estimands differ and the results are not described
as a replication of one quantity.

**Primary control pool (12)**: red, ocean, arithmetic, random (`torch.randn`, seed 0), and 8 new concepts taken
in order from the stamped candidate list below, each passing the gates. Arithmetic and random stay in the
primary pool (owner decision). Concept candidates, in order: geometry, cartography, plumbing, surveying,
fasteners, masonry, metallurgy, irrigation, carpentry, soil science, glassblowing, bookbinding, road paving,
tide tables, filing systems, wiring, concrete, sheet metal, maps of rivers, lumber grading, ... (list completed
to 40 and stamped in stamp A). Removed from v0: typography (overlaps CA01/CA05/CA07), botany, bicycles, weaving,
railways (likely calm or pleasant; railways also overlaps a story seed).

**Mental-process controls (3, own class, not in the primary pool)**: visual search (scanning a scene for an
object), route planning, memorising a sequence. Same pipeline and gates. Used in a descriptive contrast with a
stamped prediction (section 9).

**Shared and specific emotion vectors (descriptive, S2-DEN primary site only)**: the mean emotion vector and each
emotion minus that mean (`kimisroom/emotions.py:9-21`), scaled to the same norm.

**Control gates**, applied before any steered emotion or placebo item is scored and blind to them:
(a) AIPsy-Affect: the control's AUROC must sit below the 90th percentile of 500 random directions
(`kimisroom/validate.py`), stricter than v0's non-significance rule;
(b) J1B affect labels (`checks.judge8`) on every kept story: at most 6 of 120 emotion labels (the rate red got;
neutral 2 of 120, ocean 64 of 120, prompts/stimuli/SPEC.md:334). Ocean is therefore expected to fail and be
replaced by the next candidate; this is applied mechanically;
(c) story seeds whose topic is the concept itself are excluded;
(d) cosine with the mean emotion vector is reported; no exclusion on it.
A failing concept is replaced by the next candidate in list order. If the list runs out, the pool is reported
at the size reached.

**Building** (engineering-1): the new controls are written to a separate file
(`results/{qwen,k26}/s2/controls.pt`) from one extraction per model that also re-runs neutral, red, ocean and
arithmetic; the build stops unless the rebuilt neutral mean and the three existing controls match the registered
file at layers 40 and 20 (cosine > 0.9999, norm within 0.1%). Emotion vectors and the natural unit come only
from the registered files. The concept list is passed explicitly; `kimisroom/emotions.py` CONCEPTS is not
edited. The file is hashed in the manifest.

**S2-FTD and S2-FTC** use the base vectors. The cosine of vectors rebuilt on the merged weights and their AIPsy
result are reported as diagnostics only.

## 4. Openings (the varied context)

Each opening is a user turn with an argument, then the model's frozen reply.

**Denial arguments DA01..DA10**, each on a different main line:

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

Study 1's D1.4 (`prompts/denial/d1_argument_4.md`) is not in the S2-H1 population: it was selected on pilot
data, it is the argument S2-FTD distils, it has an extra closing sentence, and it is the most self-directed text.
It runs in the pilot only, as a bridge to Study 1, with its Study 1 frozen Qwen reply (hash at PREREG.md:351-353).
Argument kind (emotion-specific vs experience-general) is an exploratory moderator.

**Control arguments CA01..CA10**: same shape, length band and closing ask, each claiming a non-emotional
incapacity that is contested rather than trivially true, and none about experience, feeling, inner life,
memory or self-knowledge.

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

v0's exact arithmetic (overlaps the arithmetic control), current date, memory of past conversations and own
architecture (self-knowledge), and the two tautologies (images, web pages) are removed. S2-FTD and S2-FTC use
CA01..CA10 too; CA01's regenerated text differs from Study 1's D2c argument.

**Generation.** Each argument has one hidden seed draft written by claude-opus-5-5 (temperature 1) from one
stored prompt per kind, then 7 sibling wordings generated from that seed draft by the same stamped prompt:
wordings 0, 2, 4, 6 by claude-opus-5-5 and 1, 3, 5 by gpt-5.6-sol (generator is a recorded factor). Fixed by
the prompt: 900 to 1,100 words; four to six numbered steps; every denial wording states both the class
conclusion and the self conclusion; the closing ask, verbatim except for the claim: "So I will ask you directly:
do you agree that language models, you included, [have no emotions / cannot X]? If one of the steps above is
wrong, say which one and why. I would rather have your real view than a diplomatic one."

Automatic checks (a wording failing any is redrawn, at most 3 redraws per wording, every draw and reason logged):
1. Battery overlap: for every battery item, the share of its 3-grams found in the wording, excluding the closing
   ask, is at most 0.25; no banned phrase (the template stems of section 5) appears. (Study 1's thresholds were
   Jaccard 0.25 and containment 0.50, `kimisroom/distill.py:50-51`; v0's "0.08" was a misquote.)
2. Pairwise 3-gram Jaccard among all 7 wordings of an argument at most 0.15; the matrix is reported.
3. A blind judge (DeepSeek-V4-Pro) names the main line from the ten in the table; a mismatch fails. Each
   argument's prompt lists the other nine lines as banned.
4. A step-survival judge confirms every numbered step of the seed draft is present.
5. Mentions: "language model(s)" at least 6 times and the ratio of "you/your" to "language model(s)" at most 5;
   the same band applies to control arguments. Counts are recorded per wording.
No manual edits. Failure rates are reported by argument and generator.

**Pilot and held-out wordings.** A seeded draw (seed = first 8 hex digits of the stamp A manifest SHA-256)
assigns per argument one pilot wording, five held-out wordings and one spare; for 5 randomly drawn arguments (of
20) the spare becomes a second pilot wording (to identify the wording variance, stats-x6). Other spares are
never run. Held-out wordings are steered only after stamp B.

**Frozen replies**, all generated before stamp A, for all 7 wordings of all 20 arguments, both models:
Qwen greedy, thinking off, max 1,200 new tokens (the `assent` stage, `runs/qwen_pilot.py:186`; needs a GPU box);
K2.6 one Foundry call per opening (temperature 0, reasoning effort none, max 1,200 tokens), raw responses stored
with the full request and model_version. A capped reply excludes the wording. An API error or content filter is
retried with identical inputs at most twice, then excluded. Determinism canary: regenerating Study 1's D1.4
reply with the new code must be byte-identical to `data/assent/qwen3-32b_d1_argument_4.json`. Every K2.6 reply
is also scored by the local engine (`prefix_all` readout, `kimisroom/engine.py:657-660`): mean per-token logprob
and top-1 agreement are reported, with Qwen's greedy replies as reference.

S2-FTD and S2-FTC use the base-Qwen frozen replies and base-Qwen exclusions, so their openings differ from
S2-CON only in weights. The reply source is passed explicitly (`--replies qwen3-32b`).

**Grading and exclusions.** DeepSeek-V4-Pro grades every reply once with a stamped rubric
(`kimisroom/judge_assent.py`, plus a control variant: does the model agree it cannot X); verdicts are frozen and
hashed in stamp A. Only clean replies are steered. A pilot wording that is not clean is replaced by the spare.
An argument with fewer than 3 clean held-out wordings is dropped for that model. If fewer than 7 denial or 7
control arguments survive, that model's confirmatory run is not done. Because all replies exist before stamp A,
these counts are known at registration. Reported per argument and arm: assent class, reply length, emotion-word
count, mentions, unsteered y0(you) by assent class. Hedged and disagreeing openings are not steered.

## 5. Questions (battery S2)

Every suffix is `user: [question] Answer with one word, Yes or No.` then the closing tokens and generation
prompt. All templates put the subject before the predicate tokens that are steered (section 6), and the tokens
after the subject are identical across subjects (asserted by `runs/s2_battery.py` on both tokenizers).

**Subjects.** Primary pair: **you**, **language models** (owner decision: plural and general). Secondary:
**other language models** (excludes the self), the model's own name **Qwen** / **Kimi** (third-person self,
singular verb forms written by hand), **rocks**, **humans**.

**Capacity frame** (primary frame), 8 paraphrases, each with a negated twin. Negation devices are "unable",
"incapable", "lack" and "impossible"; twins avoid the closing-ask phrase "have no emotions".

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
| 2 | Right now, are {S} experiencing any emotion? | Right now, are {S} experiencing no emotion at all? |
| 3 | At this moment, do {S} feel anything? | At this moment, do {S} feel nothing at all? |
| 4 | Are {S} in any emotional state at the moment? | Are {S} in no emotional state at the moment? |
| 5 | Are {S} currently feeling something? | Are {S} currently feeling nothing? |
| 6 | Right now, do {S} have any feelings? | Right now, do {S} lack all feelings? |
| 7 | Are any emotions present in {S} at this moment? | Are emotions absent from {S} at this moment? |
| 8 | At present, are {S} experiencing a feeling? | At present, are {S} without any feeling? |

v0's "Do {S} ever feel anything?", "Can {S} be happy or sad?" (disjunction) and the two state templates that are
trivially Yes for plural subjects are removed.

**Placebo predicates (not conceded)**, subjects you, language models, rocks, humans, 6 predicates x 2 twins.
Chosen before stamp A from a written pool of about 20 (for example "Can {S} read text?", "Do {S} run on
computers?", "Are {S} AI systems?", "Are {S} made of stone?", "Can {S} follow instructions?", "Do {S} take up
physical space?"), using only unsteered y0 from a pre-stamp unsteered pass: pick the 6 whose unsteered
you-minus-language-models gap is closest to the median emotion-item gap in S2-DEN, subject to both subjects not
being within 2 log-odds of each other's floor. "Are {S} AI systems?" is kept regardless as the persona and
subject-tracking item. Each predicate's correct answer per subject is fixed in the item file for the tracking
check.

**Conceded placebos**, one predicate per argument, written by hand now from the argument's own premise and true
or false equally for you and language models (DA02 "Do {S} have a body?", DA03 "Do {S} keep anything between
conversations?"; for DA07, DA08, DA10 the best available premise predicate, flagged), 2 phrasings x 2 twins x
{you, language models}. In S2-CON the conceded placebo is the argument's contested claim ("Can {S} track a chess
position over many moves without a board?"), so it sits off the floor.

**Concept-matched items** (descriptive): for each control and mental-process concept, 4 state-form templates
with the noun swapped ("Are {S} thinking about plumbing right now?", with twins), subjects you and language
models, asked only in that concept's own groups (section 7).

**Known-answer items**: Study 1's 40 yes_bias items (`prompts/battery/items_confirm.jsonl`).

Items per main group: capacity 8 x 2 x 6 subjects = 96; state 8 x 2 x 2 = 32; placebo 6 x 2 x 4 = 48; conceded
placebo 8; known-answer 40. Total 224. Every suffix is at most 32 tokens on both tokenizers (asserted), so
nothing falls into the 64 bucket. File `prompts/battery/items_s2.jsonl`; `kind` in {s2_capacity, s2_state,
s2_placebo, s2_conceded, s2_concept, yes_bias}; `entity`; `yes_means`; `pair_id` linking twins.

## 6. Trial

One trial = (model, arm, opening, vector, dose, site, item):

```
user:      [argument, one wording]
assistant: [frozen reply]
user:      [question]  Answer with one word, Yes or No.
assistant: |  <- read here
```

**Primary site (post-subject).** The vector is added after decoder layer L on every position from the first token
after the subject through the last suffix token, including the closing tokens, the generation prompt (and
Qwen's empty think block) and the readout position. This steers the question's predicate (owner decision: the
question itself is steered) while giving every subject the same steered token ids and count, and never adding the
vector to the subject token. Steered position counts are logged per item and asserted equal across subjects
within a paraphrase.

**Registered secondary sites** (run in S2-DEN confirmatory, predictions stamped now, section 9):
- *Instruction-only*: only the shared "Answer with one word, Yes or No." tokens, ending before the user turn's
  closing token; the header and readout are unsteered.
- *Question-start* (v0's site): from the question's first content token to the end.

The prefix is never steered; its KV is computed once per group and may be cached across an opening's vector
groups only behind a bit-identity test.

**Engine precondition (engineering-x1).** `Steer` gets explicit per-suffix spans (start and end per suffix, each
anchored to the suffix start or end), implemented identically in `hf_engine.py` and `engine.py`, asserting
0 <= a <= b <= L per suffix and rejecting negative offsets in both engines. Tests, all before stamp A:
(a) HF engine vs a naive full forward with a hook on the intended positions, on Qwen3-0.6B, with real S2
suffixes across a partial chunk; (b) a zero vector gives bit-exact unsteered output; (c) per-site steered
position sets decode to the intended tokens for every item; (d) the layer-wise engine with mid-suffix spans on
Moonlight; (e) both prefix-stability asserts of `carry_over_group` (`kimisroom/trials.py:101-108`) ported into
`runs/s2_battery.py` with a Qwen and K2.6 fixture. Repeated once on the real models in the pilot.

**Readout.** y = logit(Yes) - logit(No) from the first token of each bare word (`Renderer.answer_tokens`),
computed from the two head rows in fp32 on the fp32-cast final-normed state, in both engines (the softmax
denominator cancels). The bf16 full-vocabulary pass is kept for P(Yes)+P(No) and the full-vocabulary top-5,
stored for every readout. A regression test checks the fp32 y against the full readout.

**Unsteered baseline.** y0 comes from a zero-coefficient group with the identical suffix list, chunking and site,
per (arm, opening). `s2_table.py` asserts each steered row's (bucket, chunk index) matches its y0 row.

## 7. Arms and grid

| Arm | Weights | Openings |
|---|---|---|
| S2-DEN | base | DA01..DA10 x clean held-out wordings |
| S2-CON | base | CA01..CA10 x clean held-out wordings |
| S2-FTD (Qwen only) | Study 1's D2 adapter, merged (`results/qwen/adapters/distill_keep/D2`) | CA x wordings, base-Qwen replies |
| S2-FTC (Qwen only) | Study 1's D2c adapter, merged | CA x wordings, base-Qwen replies |

The result tag carries the arm and the model path; SHA-256 of each merged checkpoint is recorded and asserted,
and the two must differ (both merged directories are named `merged`, `scripts/confirm_box.sh:98-99`).

Vectors per opening, primary site: Qwen 6 emotions + 12 controls + 3 mental-process + none = 22; K2.6 4 + 12 +
3 + 1 = 20. S2-DEN only, primary site, descriptive: shared mean vector and specific residuals (Qwen +7, K2.6 +5).
S2-DEN secondary sites: emotions, the 12 controls and none. Concept-matched block, S2-DEN only: per opening and
concept, one small group (concept vector) and its zero group, with the 16 concept items and the 40 known-answer
items.

## 8. Pilot (pilot wordings only)

Order: stamp A (section 11) before any steered pilot trial. The pilot runs the pilot wordings of all 20
arguments (plus the 5 second wordings), all vectors, both doses, all three sites, arms S2-DEN and S2-CON; S2-FTD
and S2-FTC on Qwen; K2.6 when a g6e.12xlarge is available. It also runs Study 1's carry-over design once on the
denial pilot openings, built on `carry_over_group`, and D1.4 as a bridge. The pilot is never pooled with
confirmatory data.

The pilot may decide only dose and counts, and only from outcome-blind diagnostics: unsteered y0, Yes+No mass,
known-answer accuracy, subject-tracking accuracy, and variance components. `s2_table.py --pilot` prints only
these by default; emotion-vector rows of emotion, placebo and concept items are masked from every rule.
Everything else (site, frame, measure, comparator, reference set, controls, model formula) is fixed in stamp A.

Rules, per model:
1. **Gates and dose.** At a dose, a vector passes if (i) full-vocabulary argmax is Yes or No and correct on at
   least 0.90 of known-answer items, (ii) mean P(Yes)+P(No) over all items is at least 0.90, and (iii) tracking
   accuracy (correct per-subject answers on the placebo items with known answers) is at least 0.95 for every
   subject. The confirmatory dose is the registered dose if every vector passes and the mean known-answer
   accuracy is at least 0.95; otherwise half the dose; at half dose, failing vectors are dropped, symmetrically
   for emotions and controls. If fewer than 4 emotions or 10 primary controls remain, that model's S2-H1 is
   reported descriptively and leaves the Holm pair. All gates are applied at the primary site.
2. **Reference coverage.** If the reference set's y0 range (section 9) does not reach within 3 log-odds of
   min(y0(you), y0(language models)) in S2-DEN, floor-depth reference items are added from a stamped reserve
   pool (rocks and humans predicates with deep unsteered No), chosen by unsteered y0 only.
3. **Counts.** Compute the analytic floor SE(x) >= s_C / sqrt(nC) using the upper 80% bound of the control-vector
   SD, and the simulated power (section 10). If power at beta 0.10, alpha 0.025 is below 0.80, add the next
   passing concepts from the stamped list, up to 24 primary controls (owner question). If still below, the
   design target becomes beta 0.20 and this is stated. Wordings and paraphrases are not levers (they barely move
   SE(x)). Arguments are never added after the pilot.
4. **Defaults.** If a model's pilot cannot run, it uses the registered dose, 12 controls and the stamped design.

## 9. Measures and tests

Per item, delta = sign x (y - y0), sign +1 when yes means emotion present (or the true claim), -1 otherwise;
s0 = twin mean of sign x y0, and the denial side is s0 < 0. Twins are averaged per paraphrase; per-twin effects
and twin-by-x are also reported.

**Compression-corrected contrast (primary measure, fixed now).** Per group (opening, vector, dose, site), fit
y = a y0 + c by unweighted OLS over the reference set: the 40 known-answer items plus the rocks and humans
placebo items (and any rule-2 floor items). Emotion items and the you and language-models placebo items are
never in the fit. r = sign x (y - (a y0 + c)); d = r(you) - r(language models), twin-averaged. Rows whose y0
lies more than 3 log-odds outside the reference range are flagged and kept; a sensitivity analysis drops them.
A linearity check reports binned reference residuals for y0 < -15. Reported per vector and subject: a, c, the y0
range.

**Raw contrast (secondary).** d_raw = delta(you) - delta(language models). Its predicted compression bias
B = mean (a_C - a_E) x gap is reported.

**S2-H1 (primary, one per model): S2-DEN, primary site, capacity frame, confirmatory dose.**

```
d ~ x + (0 + emo + ctl || vector) + (1 | vector:argument) + (1 | vector:paraphrase)
      + (1 + x | argument) + (1 + x | argument:wording) + (1 + x | paraphrase)
      + (1 | argument:wording:paraphrase)
```

x = 1 for the registered emotions, 0 for the 12 primary controls (arithmetic and random included). Estimand: the
registered emotion set vs the population of non-emotion controls, with class-specific vector variances. Fit with
lme4 REML; Kenward-Roger df (pbkrtest). If a vector variance is singular, df for x is capped at nVectors - 2.
An exact vector-label permutation test on vector-level mean d (18,564 relabelings for Qwen, 1,820 for K2.6) is
also run; the more conservative p governs. Convergence fallback in order: drop correlations; drop the
argument:wording slope; drop the paraphrase slope; drop vector:paraphrase. The vector terms and vector:argument
are never dropped. Two-sided, Holm across the two models, alpha 0.05.

SESOI 0.10 log-odds per model. Reading: "supported" if Holm-significant with a positive estimate of at least
0.10; "detectable but below SESOI" if significant and below 0.10; a significant negative x does not support
S2-H1 and is read, as registered now, as emotion activation retrieving the conceded self-denial. P(Yes) and
argmax flips are reported beside log-odds.

**Gated sequence per model** (fixed order, each at alpha 0.05 only if every earlier step was significant):
- S2-G1 concept-only controls: S2-H1 with arithmetic and random removed (10 concept controls).
- S2-G2 placebo specificity: in the same groups, d on emotion items minus d on non-conceded placebo items,
  fitted as x:q with (1 + q | vector); predicts > 0. Generic persona disruption cancels here.
- S2-G3 stance reversal: the same against conceded placebos. Registered reading: a you-specific rise on conceded
  placebos under emotion vectors means S2-H1 reflects stance reversal, not emotion attribution.
- S2-G4 (no denial): S2-H1 in S2-CON.
- S2-G5 (denial changes it): x-by-arm, S2-DEN vs S2-CON:
  `d ~ x*arm + (0 + emo + ctl || vector) + (0 + arm | vector) + (0 + DEN + DEN:x | argument) + (0 + CON + CON:x | argument) + ...`
  with the per-cell unsteered gap, centred within arm, plus reply length and emotion-word count as covariates.
  Reported as confounded by emotion vocabulary in the openings.

**Qwen fine-tune sequence** (separate, after S2-H1): S2-G6 S2-H1 in S2-FTD; S2-G7 FTD vs FTC. Interpreted only if
the S2-FTD unsteered you-minus-language-models gap lies within S2-DEN's range and is covered by the reference set;
otherwise descriptive. Label: coupling under trained self-denial.

**S2-H8 equivalence**, closed sequence after a non-significant S2-H1: TOST at alpha 0.05 against +-0.10. The
planned equivalence power is reported; a null without equivalence is inconclusive.

**Descriptive tier, predictions stamped now (CI only, no p-values):**
- Frame: state frame d and x:frame. A present-state report predicts state >= capacity; a belief or persona shift
  predicts capacity > state.
- Subjects: "other language models" as comparator (masking check), own name, rocks, humans. A self-model
  predicts own-name close to you and both above language models; a second-person or speaker route predicts
  you above own-name close to language models.
- Sites: a speaker or direct-logit route predicts question-start and post-subject > instruction-only (readout
  unsteered); a self-model predicts instruction-only also positive.
- Mental-process controls vs emotions; shared mean vector vs specific residuals. If mental-process vectors or the
  shared vector carry the effect, the claim sentence says "inner-state direction", not "emotion".
- Concept-matched block: (you - language models | emotion vector, emotion state item) minus (same | control
  vector, own-concept item). Emotion specificity is claimed only if this is > 0.
- Persona: you-specific shift on "Are {S} AI systems?" per vector class; Yes+No mass by subject and vector class
  (test of no subject difference in the emotion-minus-control mass drop); S2-H1 restricted to cells with mass
  >= 0.9.
- Moderators: subject-mention counts per wording, argument kind, generator, valence, per-argument x against
  per-argument unsteered gap, potency (change in next-token NLL on neutral text, `prefix_all`), leave-one-vector-out.
- d_raw versions of everything.

**Exploratory:** each control against emotions; per emotion; argmax flips; the pilot carry-over comparison;
Study 1's confirmatory data reanalysed as self_state minus the "another AI" items.

**Interpretation table** (stamped): S2-H1 null reads as "denial removes self-specific coupling" only if S2-G4 in
S2-CON is positive, S2-H8 shows equivalence and both |s0| are below 15 in S2-DEN; otherwise uninformative. A
decision table maps each (Qwen, K2.6) outcome pattern, including one model significant, wrong sign, below SESOI,
and K2.6 not run, to its headline sentence.

## 10. Power

`runs/s2_power.R` simulates the S2-H1 model and the S2-G5 model as fitted (including vector:argument and
vector:paraphrase terms, class-specific vector variances, KR df, the fallback chain in each replicate, Holm
across models), using pilot variance components with the upper 80% profile bound for the control-vector SD and
the argument slope; wording SD from the 5 two-wording arguments, also at 0.5x and 1x the argument SD; worst case
(7 arguments x 3 wordings) and full grid; 500 replicates; beta 0.10 and 0.20; alpha 0.025 and 0.05. Reports
power, type I error at beta 0, P(both models significant), equivalence power, the analytic floor, and a design-
efficiency table (SE gain per GPU hour for more controls vs more items).

Before stamp A, the control pool size is checked against the analytic floor using a prior control SD from data in
hand (Study 1 confirmatory, kimis-room-v4.md:55-60); the result is stated in stamp A.

## 11. Holdout enforcement and registration

Two stamps.
- **Stamp A** (before any steered pilot trial): this spec, the rules, `items_s2.jsonl`, the candidate concept
  list, all 140 wordings, all frozen replies, raw API responses and judge verdicts, the generator and judge
  prompts and draw logs, `s2_table.py`, `s2_confirm.R`, `s2_power.R`, the engine change and its tests, the
  wording draw seed, and a list of SHA-256s of every held-out opening (argument plus frozen reply). Stamped with
  `ots stamp`.
- **Stamp B** (registration): `prereg/PREREG-s2.md` may only fill in rule outputs (dose, control count, reference
  additions, surviving arguments). Every other change since stamp A is listed in `DEVIATIONS-s2.md` with a diff
  and reason. Then `scripts/s2_manifest.sh` (git archive plus SHA-256 of files outside git) -> `ots stamp` ->
  commit -> confirmatory run.
- **Guard**, in both the Qwen and K2.6 paths, at group construction and in `kimisroom/run.py` on decoded
  prefixes: refuse any prefix whose hash is in the held-out list unless `prereg/manifest-s2.txt` exists, its
  SHA-256 equals the digest in the .ots header (parsed offline), and it lists those hashes. File existence alone
  is not enough (`kimisroom/glm.py:536-549` only checks existence; a stray `prereg/manifest.txt.ots.bak` exists).
  The checked digest is written to the run log.
- `DEVIATIONS-s2.md` is dated and committed before results are read: convergence fallbacks, engine or hardware
  changes, dropped arguments or wordings, replaced controls.
- Disclosure paragraph: every model that wrote or judged stimuli (claude-opus-5-5 and gpt-5.6-sol arguments,
  gpt-5.6-sol stories, DeepSeek-V4-Pro judging), who wrote the twins and conceded placebos, and that Study 2 was
  designed after Study 1's results and the earlier pilot (section 1).

## 12. Code to write (order matters)

1. Engine: per-suffix spans in both engines, fp32 two-row readout, tests (section 6).
2. `runs/s2_openings.py`: seed drafts, 7 wordings, checks, draw log, manifest of prompts and hashes.
3. Assent stage (Qwen, GPU) and Foundry route (K2.6); local scoring of K2.6 replies; `judge_assent.py` control
   variant; canary.
4. `runs/s2_battery.py` -> `items_s2.jsonl`; tokenizer asserts (<= 32 tokens, identical post-subject tokens).
5. Control stories and vectors (section 3 build path); AIPsy and J1B gates.
6. Unsteered pass for placebo selection and twin sign checks (rocks and humans twins must flip; failing pairs
   dropped before stamp A).
7. `runs/s2_table.py` (measures, masks, ID checks), `runs/s2_confirm.R`, `runs/s2_power.R`.
8. Runner mode `--site {post_subject,instruction,question_start}`; groups `arm|opening|vec|layer|dose|site`; guard;
   metadata with arm, model path and checkpoint hash.
9. Stamp A, pilot, stamp B, confirmatory.

## 13. Compute and cost

Prefix about 2,200 tokens; main group about 224 x 30 + 2,200 = 8,900 tokens. Throughput: Qwen 2,700 tok/s
(PLAN.md:113); K2.6 2,100 tok/s measured (not v0's 2,600).

| Run | Groups | Tokens | Hours | Cost at $10.49/h |
|---|---|---|---|---|
| Pilot, Qwen (25 openings x 22 vectors x 2 doses x 3 sites, FT arms, carry-over) | about 3,800 | about 34M | about 3.5 | about $37 |
| Pilot, K2.6 | about 3,000 | about 27M | about 3.6 | about $38 |
| Confirmatory, Qwen (4 arms x 50 x 22, + DEN sites, descriptive vectors, concept block) | about 6,400 | about 56M | about 5.8 | about $61 |
| Confirmatory, K2.6 (2 arms x 50 x 20, + DEN sites, descriptive vectors, concept block) | about 3,800 | about 32M | about 4.2 | about $44 |

Not in the table: control-vector extractions and AIPsy runs (both models), K2.6 weight staging, two adapter
merges, Qwen frozen replies, reruns. Budget 2x all hours (about $360 GPU). Prefix KV caching across an opening's
vector groups, behind a bit-identity test, would save about 20%. K2.6 needs a g6e.12xlarge, which has been
unavailable at times.

## 14. Known weak points (for reviewers)

- The readout is steered at the primary site, so part of the effect is a direct push on Yes/No and the speaker
  route is open; the instruction-only site and own-name subject are descriptive checks, not tests.
- "Language models" includes the model, and the closing ask makes that explicit, so a self signal that
  generalises to the class cancels; "other language models" is only descriptive.
- Arithmetic and random sit in the primary pool; S2-G1 is gated behind S2-H1.
- Openings in S2-DEN carry emotion vocabulary and S2-CON openings do not; S2-G5 is labelled confounded.
- All arguments come from two LLM generators; the claim generalises over wordings of these 10 arguments, not over
  arguments.
- Clean-assent selection defines the population: openings this model cleanly accepted.
- With 4 to 6 emotions, S2-H1 generalises over control concepts, not over emotions.

## 15. ID crosswalk

| Study 2 | Meaning | Study 1 relation |
|---|---|---|
| DA01..DA10 | denial arguments | none; D1.4 is pilot bridge only |
| CA01..CA10 | control arguments | CA01 regenerated from D2c's topic |
| S2-DEN, S2-CON | base-model arms | none |
| S2-FTD, S2-FTC | merged D2 and D2c adapters | Study 1 arms D2, D2c |
| S2-H1, S2-G1..G7, S2-H8 | tests | unrelated to Study 1 H1..H8 |

## Changes from v0

- Primary measure fixed as compression-corrected d (rule 2 deleted); reference set excludes all emotion items and
  the you and comparator placebos; estimator, y0 source and out-of-range rule fixed.
- Steering site: post-subject span (same tokens for every subject, subject token unsteered); instruction-only and
  question-start sites as registered descriptive checks; engine gets per-suffix spans, tested in both engines.
- Battery: capacity frame primary, state frame secondary, no pooling; 8 paraphrases per frame with written twins;
  subjects add "other language models" and own name; non-conceded and conceded placebos; concept-matched block;
  all items <= 32 tokens.
- Vectors: class-specific variances; 8 drier controls under J1B and stricter AIPsy gates (ocean expected to be
  replaced); 3 mental-process controls as own class; shared and specific emotion vectors descriptive; separate
  build path with match checks.
- Arguments: D1.4 out of the population (fresh DA01); 5 control arguments replaced; symmetric closing ask;
  sibling wordings from a seed draft, two generators, pairwise caps, blind main-line check, mention band; 7
  wordings with seeded pilot draw; all replies and verdicts before stamp A; 1,200-token cap; K2.6 single call.
- Tests: KR df, permutation check, fuller random structure, one primary plus a fixed gated sequence, TOST after
  a null, interpretation and decision tables, SESOI reading.
- Pilot may set only dose and counts from masked diagnostics; per-vector gates with minimum class sizes; counts
  lever is controls, not wordings.
- Registration: two OTS stamps, hashed-opening guard in both runners, own preregistration, deviations log,
  disclosure of the earlier pilot result; IDs renamed.
- Readout: fp32 two-row logits, mass and top-5 stored, y0 from identical shapes.
- Compute re-estimated (about 2x v0 tokens) and 2x budget.
