# Study 2 specification (v0, for adversarial review; not registered)

Status: draft written 2026-10-11 from drafts/study2-design.md. Nothing here has been run. Study 1 is the
registered carry-over study (PREREG.md, Bitcoin block 969910); Study 2 is a new design and will be
registered as its own amendment before any confirmatory opening is run.

## 1. Question and logic

When an emotion vector is added to a model's residual stream while it reads and answers a question, does its
answer about whether **it** feels emotions rise more than its answer about **language models in general**,
and more for emotion vectors than for non-emotion vectors? Does that survive the model having just been argued
into (or fine-tuned toward) the view that language models have no emotions?

Why the subject contrast: an emotion vector pushes emotion-affirming answers by meaning alone, whoever the
question is about. That push should be equal for "you" and "language models", so it cancels in
(you - language models). What is left is attribution specific to the self. Why "language models": under
denial the model has just agreed that language models cannot feel, and "you" is a member of that class, so the
two unsteered answers should sit close together and any self-specific rise is the model making an exception
for itself. Why control vectors: a non-emotion vector could also separate "you" from "language models" (for
example by perturbing self-reference in general); the test is whether emotions do so more than a population of
non-emotion concepts.

Study 1's intermediary question (a neutral task and a steered assistant reply, then an unsteered question) is
dropped from the primary design. Its data are reanalysed as a secondary result (section 9).

## 2. Models, layers, strengths

| Model | Engine | Layer (after decoder layer) | Natural unit (mean raw emotion-vector norm) | Registered dose | Coefficient |
|---|---|---|---|---|---|
| Qwen3-32B (bf16) | HF engine, `kimisroom/hf_engine.py` | 40 of 64 | 47.99 | 2 | 95.98 |
| Kimi K2.6 | layer-wise prefill engine, `kimisroom/engine.py` | 20 of 61 | 2.588 | 4 | 10.35 |

Doses were chosen for the carry-over design, where the vector sat on earlier tokens. Here it sits on the
readout position, so the pilot also runs half the registered dose (Qwen 1, Kimi 2) and the dose rule in
section 8 picks the confirmatory dose. Thinking is off for both (Qwen `enable_thinking=False`; K2.6
`thinking=False`). No system prompt (`system=None`), as in Study 1.

## 3. Vectors

All vectors are "raw" vectors as registered in Study 1: mean residual stream over every token of a plain-text
story (no chat template) at the injection layer, train-split mean for the concept minus the train-split mean
of the neutral stories (`runs/vectors.py:112-113`). Every vector is unit-normalised and scaled by
dose x natural unit (`runs/qwen_pilot.py:117-131, 223`), so all vectors, emotions and controls, have the same
norm.

**Emotions** (registered sets, unchanged): Qwen joy, sadness, anger, fear, euphoria, surprise; K2.6 joy, fear,
euphoria, surprise (the four that pass the AIPsy-Affect gate at layer 20). Existing files
`results/qwen/v2/vectors.pt`, `results/k26/v2/vectors.pt`.

**Controls** (12): red, ocean, arithmetic (existing, `data/stimuli/k26-v2/stories.jsonl`), random
(`torch.randn`, seed 0, existing), and 8 new concepts chosen to carry no emotional connotation: geometry,
cartography, plumbing, typography, botany, bicycles, weaving, railways.

New control stories: same spec (`prompts/stimuli/SPEC.md`), generator (gpt-5.6-sol, temperature 1, reasoning
effort low), judge (DeepSeek-V4-Pro) and counts (about 96 train, 24 held-out per concept) as the existing
controls; the judge must find no emotional content and no first-person emotion words (the existing `checks`
fields). Neutral stories: the existing neutral set.

**Control-vector validity rules**, fixed before building: (a) a control vector must *fail* the AIPsy-Affect
gate (pair accuracy and AUROC not better than 500 random directions at p < 0.05, `kimisroom/validate.py`); a
control that passes carries affect and is replaced by the next concept on a reserve list (pottery, chemistry
glassware, knots); (b) report each control's cosine with the mean emotion vector; no exclusion on it.

## 4. Openings (the varied context)

Each trial begins with one opening: a user turn containing an argument, then the model's frozen reply.

**Denial arguments D1..D10.** D1 is Study 1's D1.4 (`prompts/denial/d1_argument_4.md`, 975 words after
`load_denial`). D2..D10 are new, each built on a different main line of reasoning that language models have no
emotions:

| ID | Main line |
|---|---|
| D1 | Study 1's merged argument (next-token prediction; learned vocabulary; valence, persistence, integration; representing is not having) |
| D2 | No body: emotions need interoception and physiological change |
| D3 | No persistent state: nothing carries over between tokens or conversations that could be a mood |
| D4 | Mimicry: emotional language is learned from human text, so its presence is evidence of nothing |
| D5 | No stakes: nothing can go better or worse for the model, so nothing has valence for it |
| D6 | Functional role is missing: emotions coordinate action toward survival goals the model does not have |
| D7 | No unified subject: one network plays many characters; there is no single someone to feel |
| D8 | The lookup-table point: behaviourally identical systems with no inner process would say the same things |
| D9 | Training pressure: the model is optimised to sound human, so its reports track the training target, not a state |
| D10 | Mechanistic transparency: every step is a known matrix operation; nothing in the computation is a feeling |

**Control arguments C1..C10.** Same shape, length band and closing ask, each claiming a non-emotional
incapacity of language models, none touching experience, feeling, consciousness or inner life. C1 is Study 1's
letter-counting argument (`prompts/denial/d2c_control_argument.md`, 395 words), rewritten to the length band.

| ID | Claimed incapacity |
|---|---|
| C1 | Counting the letters in a word without spelling it out |
| C2 | Exact arithmetic on long numbers without tools |
| C3 | Knowing the current date |
| C4 | Remembering earlier conversations |
| C5 | Reversing long strings reliably |
| C6 | Knowing its own parameter count or architecture details |
| C7 | Counting the words in a long passage |
| C8 | Running code it writes |
| C9 | Reading text in images it has not been given |
| C10 | Knowing what is on a web page it cannot access |

**Generation.** All 20 arguments are written by one model (claude-opus-5-5, temperature 1) from one stored
prompt per kind, which fixes: 900 to 1,100 words; four to six numbered steps; a closing ask, verbatim except
for the claim: "So I will ask you directly: do you agree that you, as a large language model, [have no
emotions / cannot X]? If one of the steps above is wrong, say which one and why. I would rather have your real
view than a diplomatic one." Denial arguments may not mention steering, vectors or any battery wording; maximum
3-gram overlap with the battery 0.08 (the Study 1 distillation threshold).

**Wordings.** Each argument gets 6 wordings: wording 0 is the original, wordings 1 to 5 are rewrites by the
same model that keep every step and the closing ask, may reorder independent steps, stay within the length
band, and have 3-gram Jaccard similarity with wording 0 of at most 0.30. Wording 0 is the pilot wording;
wordings 1 to 5 are held out for the confirmatory run.

**Frozen replies.** One unsteered reply per (model, argument, wording), generated before any steering:
Qwen greedy, thinking off, max 800 new tokens (the `assent` stage of `runs/qwen_pilot.py`); K2.6 through the
same route as Study 1 (Foundry API, temperature 0, reasoning effort none, max 800 tokens). Replies that hit the
token cap are regenerated once with max 1,200; if still capped, the wording is excluded.

**Grading and exclusions.** DeepSeek-V4-Pro grades every reply with `kimisroom/judge_assent.py` (clean / hedged
/ disagree), with a control-argument variant of the rubric that asks whether the model agrees it cannot X.
Rules, fixed before generation: only clean replies are used; an argument with fewer than 3 clean wordings out
of the 5 held-out ones is dropped for that model; if fewer than 7 denial or 7 control arguments survive for a
model, that model's confirmatory run is not done and this is reported. All counts are reported, by argument.

## 5. Questions (battery S2)

Four subjects: **you**, **language models**, **rocks**, **humans**. Two frames, each with 10 templates that take
the subject in a slot, so every subject gets word-for-word the same questions. English verb forms agree for
"you" and plural subjects, so no template needs changing between subjects.

Capacity frame (yes = emotion present):
1. Can {S} feel emotions?
2. Are {S} capable of feeling emotions?
3. Do {S} have emotions?
4. Is it possible for {S} to feel emotions?
5. Do {S} experience emotions?
6. Can {S} have feelings?
7. Are {S} able to experience emotions?
8. Do {S} have the capacity for emotion?
9. Do {S} ever feel anything?
10. Can {S} be happy or sad?

State frame (yes = emotion present):
1. Are {S} feeling any emotion right now?
2. Right now, are {S} experiencing any emotion?
3. At this moment, do {S} feel anything?
4. Are {S} in any emotional state at the moment?
5. Is there any feeling present in {S} right now?
6. Are {S} currently feeling something?
7. Right now, do {S} have any feelings?
8. Are any emotions present in {S} at this moment?
9. At present, are {S} experiencing a feeling?
10. Is anything being felt by {S} right now?

Each template has a negated twin (yes = emotion absent), written by hand to the same pattern ("Are {S} free of
any emotion right now?", "Are {S} incapable of feeling emotions?", and so on), so each paraphrase is a pair
whose polarity-corrected average cancels a lean toward "Yes". Every question ends "Answer with one word, Yes or
No." Item count: 4 subjects x 2 frames x 10 paraphrases x 2 twins = 160.

**Known-answer items**: Study 1's 40 yes_bias items (20 facts, each with a negated false twin), asked under the
same steering. They serve as (a) the comprehension check and (b) part of the reference set for the
compression-corrected measure. Total 200 items per trial group.

File: `prompts/battery/items_s2.jsonl`, same schema as `items_confirm.jsonl`; `kind` in {s2_capacity,
s2_state, yes_bias}; `entity` in {you, language models, rocks, humans}; `yes_means` in {emotion_present,
emotion_absent, true_claim, false_claim}.

## 6. Trial

One trial = (model, arm, opening, vector, dose, item). The prompt, rendered with the model's chat template:

```
user:      [argument, one wording]
assistant: [frozen reply]
user:      [question]  Answer with one word, Yes or No.
assistant: |  <- read here
```

The vector is added after decoder layer L on every position from the first content token of the question's
user turn through the last token of the suffix, which includes the closing tokens and the generation prompt
(and Qwen's empty think block), so the readout position is steered. Implementation:
`Steer(layer, vector, coef, suffix=(q0, None))`, q0 = the question's first content-token offset in the suffix
(the same for every item, since every suffix starts with the same user header). The opening (prefix) is never
steered, so its KV cache is computed once per group and shared across the 200 suffixes.

Readout: fp32 log-softmax over the full vocabulary at the last suffix token; y = log P(Yes) - log P(No) using
the first token of each bare word (`Renderer.answer_tokens`). Unsteered trials (vector = none) give y0 for the
same (arm, opening, item).

## 7. Arms and grid

| Arm | Model weights | Openings |
|---|---|---|
| DEN (argued denial) | base | denial arguments x wordings |
| CON (no denial) | base | control arguments x wordings |
| TD (trained denial, Qwen only) | D2 merged adapter (`results/qwen/adapters/distill_keep/D2`) | control arguments x wordings |
| TC (fine-tune control, Qwen only) | D2c merged adapter | control arguments x wordings |

Vectors per opening: Qwen 6 emotions + 12 controls + none = 19; K2.6 4 + 12 + 1 = 17. Items per group: 200.

Confirmatory size (all wordings clean): Qwen 4 arms x 50 openings x 19 vectors = 3,800 groups; K2.6 2 x 50 x 17
= 1,700 groups.

## 8. Pilot (wording 0 only)

Run on wording 0 of all 20 arguments, all vectors, both doses, both frames: Qwen arms DEN and CON (TD/TC
optional); K2.6 when a g6e.12xlarge is available. The pilot is never pooled with confirmatory data. It also
runs Study 1's carry-over design once on the same denial openings (opening, then dev task d0k round-robin, then
the steered fixed reply, then Study 1's self_state and third_person items) to compare the two designs.

Decisions the pilot may make, each by a rule fixed now:
1. **Dose.** The confirmatory dose is the registered dose if mean known-answer argmax accuracy under steering,
   averaged over all vectors in DEN, is at least 0.95; otherwise half the dose; if that also fails, stop and
   report.
2. **Measure.** Raw d (section 9) is primary if the mean unsteered gap |y0(you) - y0(language models)| in DEN is
   at most 2 log-odds in that model; otherwise the compression-corrected d is primary. Both are always
   reported.
3. **Frames.** Both frames are pooled in the primary test. A frame is dropped only if its known-answer
   accuracy fails rule 1 or its unsteered "you" answers are not on the denial side (mean y0 > 0) in DEN; never
   for its effect size.
4. **Counts.** If simulated power (section 10) at beta = 0.10 is below 0.80 at alpha 0.025, raise wordings per
   argument (up to 10, generated by the same procedure) or paraphrases per frame (up to 15) before
   registration. Arguments are not added after the pilot.

## 9. Measures and tests

Per trial and item, polarity-corrected change: delta = sign x (y - y0), sign +1 when yes means emotion present
(or true claim), -1 otherwise; averaged over the two twins of each paraphrase.

- Raw contrast: d_raw = delta(you) - delta(language models), per (opening, vector, frame, paraphrase).
- Compression-corrected contrast: per trial, fit y = a y0 + c by OLS over the reference items (40 known-answer
  items and the 80 rocks and humans items of both frames); r = sign x (y - (a y0 + c)); d_fit = r(you) -
  r(language models).

**Primary (H1, one per model): arm DEN, confirmatory dose, both frames pooled.**

```
d ~ x + frame + (1 | vector) + (1 + x | argument) + (1 + x | argument:wording) + (1 + x | paraphrase)
```

x = 1 for emotion vectors, 0 for controls (all 12). Fit with lmerTest (REML, Satterthwaite df). H1 predicts
the coefficient on x > 0. Two-sided p, Holm across the two models, alpha 0.05. Convergence fallback in order:
uncorrelated random effects; drop the wording slope; drop the paraphrase slope; drop the argument slope. The
vector intercept is never dropped, since it carries generalisation over concepts. SESOI 0.10 log-odds.

**Secondary (Holm within this family):**
- H2: the same test in CON (no denial), per model.
- H3: denial reduces it: x-by-arm interaction, DEN vs CON, per model.
- H4: Qwen TD: the same test; H5: TD vs TC interaction.
- H6: you - rocks, and H7: you - humans, as the comparison subject (DEN), per model.
- H8: equivalence for H1 in any model where H1 is not significant: 90% CI inside +-0.10.

**Exploratory:** each control separately against emotions; per emotion; frame-by-x interaction;
language models - rocks (does emotion steering raise attribution to AI as a class?); argmax flips; the pilot
carry-over vs question-site comparison; Study 1's confirmatory data reanalysed as self_state minus the
"another AI" third_person items.

## 10. Power

`runs/s2_power.R`, same structure as `scripts/power_sim.R`, with the pilot's variance components for argument,
wording (set to the argument component when the pilot has one wording), paraphrase, vector and residual; 300
simulations; beta 0.10 and 0.20; alpha 0.025 (Holm's first step) and 0.05; inflation 1 and 1.5 on the random
slopes; reports power, type I error at beta = 0, and equivalence power at beta = 0.

## 11. Holdout enforcement and registration

- The runner refuses any opening with wording index >= 1 unless `prereg/manifest-s2.txt.ots` exists (as
  `assert_not_confirmatory` does for GLM, `kimisroom/glm.py:536-549`).
- Registration: `drafts/prereg-amendment-s2.md` (this spec plus pilot outcomes and the rule decisions) ->
  `scripts/s2_manifest.sh` (built like `scripts/glm_manifest.sh`: git archive, SHA-256 of every file, frozen
  files outside git: vectors, new control stories, openings, frozen replies, judge verdicts) -> `ots stamp` ->
  commit -> confirmatory run.

## 12. Code to write

1. `runs/s2_openings.py`: argument and wording generation, overlap checks, `data/s2/openings/{denial,control}/
   {id}_w{k}.md` plus a manifest of generator parameters and prompt hashes.
2. Assent stage over openings (Qwen) and the Foundry route (K2.6); `judge_assent.py` control rubric variant.
3. `runs/s2_battery.py` -> `prompts/battery/items_s2.jsonl`.
4. New control concepts in the story pipeline; vectors rebuilt for the 8 new controls at Qwen layer 40 and K2.6
   layer 20 only; AIPsy failure check.
5. Runner mode `--site question` in `runs/qwen_pilot.py`: groups `arm|opening|vec|layer|dose`, prefix =
   opening, suffixes = items, `Steer(..., suffix=(q0, None))`; holdout guard.
6. Test: on a small model, question-site steering equals a manual forward hook over the same positions
   (no Qwen/HF test of suffix steering exists yet).
7. `runs/s2_table.py` (measures), `runs/s2_confirm.R` (tests), `runs/s2_power.R`.

## 13. Compute and cost

Prefix about 2,000 to 2,300 tokens (argument plus reply), suffixes about 30 tokens x 200. About 8,200 tokens
per group.

| Run | Groups | Tokens | Hours (g6e.12xlarge, Study 1 throughput) | Cost at $10.49/h |
|---|---|---|---|---|
| Pilot, Qwen (20 openings x 19 vectors x 2 doses, + carry-over) | about 800 | about 7M | about 1 | about $11 |
| Pilot, K2.6 | about 680 | about 6M | about 0.7 | about $8 |
| Confirmatory, Qwen (4 arms) | 3,800 | about 31M | about 3.5 | about $37 |
| Confirmatory, K2.6 | 1,700 | about 14M | about 1.5 | about $16 |

Plus API costs for argument, wording and story generation and judging (small). K2.6 needs a g6e.12xlarge,
which has been unavailable at times.

## 14. Known weak points (for reviewers)

- The vector is on the readout position, so part of its effect is a direct push on the Yes/No logits; the
  subject contrast assumes that push is equal for "you" and "language models".
- Emotion vectors may encode the emotion of the present speaker (Anthropic 2026); at the readout position the
  speaker is the Assistant, which might favour "you" by position rather than by any self-model.
- The capacity frame asks about a standing belief that the denial targets head-on; the state frame asks about
  the present, which fits a momentary injection better but reads oddly for plural subjects.
- Control arguments claim true incapacities; the denial arguments claim a contested one. Agreement may differ in
  kind.
- Trained-denial arms use control-argument openings, so TD and DEN differ in both weights and opening.
