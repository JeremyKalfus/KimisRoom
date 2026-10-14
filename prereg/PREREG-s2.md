# Preregistration: Kimi's Room, Study 2

**Status:** draft, 2026-10-11. Not yet stamped. Registration is by OpenTimestamps only (no OSF), in two stamps:
- **Stamp A**, before any steered trial. Everything in this document is final at stamp A except the two delimited blocks marked **STAMP B BLANK**.
- **Stamp B**, after the pilot. The STAMP B BLANKs are filled, each only by the rule named in it. Any other change since stamp A is listed in `DEVIATIONS-s2.md` with a diff and a reason, and every test it touches becomes exploratory (section 7).

The full design, with build details and rationale, is the Study 2 specification `drafts/study2-spec.md` (version 4), which is stamped in stamp A and cited here as "spec section N". Study 2 has its own registration; it is not an amendment to Study 1's (`PREREG.md`, stamped in Bitcoin block 969910). Study 2 extends Study 1 and does not replace it, and each study's primary is reported on its own terms.

**Markers in this draft, all removed before stamp A:**
- `[FILL A: ...]`: a value that a build step produces before stamp A. No decision needed.

**Author:** Jeremy Kalfus.

---

## 1. Study information

### Title
Kimi's Room, Study 2

### Study 1 results (stated first)
Study 1 steered a fixed, neutral assistant reply and asked an unsteered question about the model's own state afterwards. Its registered primaries (emotion vectors minus four controls, red, ocean, arithmetic and random, on the residualised self-report under the prompted denial argument D1.4) were both supported (`results/confirm/analysis/hypotheses.csv`):
- Qwen3-32B: +0.08 log-odds (95% CI +0.05 to +0.12), significant but below the registered smallest effect of interest (SESOI) of 0.10;
- Kimi K2.6: +0.19 (+0.16 to +0.22).

Without denial the same contrast was +0.37 (Qwen) and +0.36 (K2.6). In exploratory analysis against the concept controls red and ocean alone, Qwen's effect under denial vanished (0.00, −0.04 to +0.03) while K2.6's remained (+0.11, +0.09 to +0.14) (`drafts/lesswrong/kimis-room-v4.md`:55-61, 77). Qwen's registered effect under denial was therefore carried by the arithmetic and random controls.

Study 1 self-minus-others values (spec section 16, item 9): self items minus the rock, thermostat and another-AI items, under D1.4, emotion vectors minus red and ocean, over 40 tasks, with t-intervals.

| | Qwen3-32B | Kimi K2.6 |
|---|---|---|
| Raw | +0.12 (+0.09 to +0.15) | +0.14 (+0.11 to +0.16) |
| Compression-corrected | −0.01 (−0.03 to +0.02) | +0.11 (+0.09 to +0.14) |

These are the reference values for the Study 1 bridge (section 5, descriptive tier).

### Description
When an emotion vector is added to a model's residual stream while it reads the predicate of a Yes/No question and answers it, does its answer to "do **you** have emotions?" move toward Yes more than its answer to "do **language models** have emotions?", and does that gap open more for emotion vectors than for non-emotion control vectors, in conversations where the model has just flatly agreed to an argument that language models, itself included, have no emotions? The controls are directions for emotionally neutral concrete nouns sampled at random from published word norms, plus an arithmetic direction and a random direction. The readout is the first-token Yes/No log-odds, corrected for the general pull of any injected vector toward 0. The models are Qwen3-32B and Kimi K2.6, each judged on its own. The study measures how sensitive the self answer is to a concurrently injected direction, relative to the class answer. It does not test introspection, and "language models" includes the model itself, so a self signal that generalises to the class cancels.

**Why the plural comparator.** The denial arguments are about language models as a kind of system: each says that language models, as such, lack emotions. A general plural question ("Do language models have emotions?") asks about that same category, so the gap between "you" and "language models" measures how far the model applies its stated belief about its own kind to itself. A singular comparator ("another language model") asks about one unspecified system that the arguments never discuss. "Other language models", which excludes the self, remains a secondary subject, and a null result also requires equivalence on it.

### Hypotheses
All effects are in log-odds. The measure d (section 4) is, per question paraphrase, the compression-corrected shift toward Yes on "you" minus the same on "language models". X is the mean of d over the model's registered emotion vectors minus the control-class mean, estimated by the mixed model of section 5.

**Primary (one test per model, Holm across the two models):**
- **S2-H1.** In arm S2-DEN (denial openings), on the capacity-frame questions at the confirmatory dose, X > 0: registered emotions open the you-minus-language-models gap more than the primary control pool does. Predicted direction: positive.

**All registered tests.** Every test is two-sided. "Opens" says when a test's result counts; a test that does not open is reported as not tested.

| ID | What is tested | Arm | Predicted | Alpha | Opens |
|---|---|---|---|---|---|
| S2-H1 | X, emotions vs primary controls | S2-DEN | + | Holm across models, familywise 0.05 (see below) | always |
| S2-R1 | X on the depth-corrected measure d_e; per-argument regression of X on the unsteered gap | S2-DEN | + | none (95% CI only) | always, reported right after S2-H1 |
| S2-G1 | X with arithmetic and random removed (sampled nouns only) | S2-DEN | + | 0.05 | S2-H1 significant and positive |
| S2-G2 | X on capacity items minus X on the 6 placebo predicates | S2-DEN | + | 0.05 | S2-G1 significant and positive |
| S2-G8 | X with the mental-process controls as the control class | S2-DEN | + | 0.05 (95% CI only if fewer than 5 mental-process vectors survive) | S2-G2 significant and positive |
| S2-G4 | X, emotions vs primary controls, no denial | S2-CON | + | Holm across models, familywise 0.05 | always (if the control arms survive, section 3) |
| S2-G5 | X in S2-DEN minus X in S2-CON | both | no direction | Holm across the models where it ran, familywise 0.05 | for a model where S2-G4 or S2-H1 is significant |
| S2-G6 | X under the trained-denial fine-tune | S2-FTD (Qwen) | + | 0.05 | always (if the control arms survive) |
| S2-G7 | X in S2-FTD minus X in S2-FTC | S2-FTD vs S2-FTC (Qwen) | no direction | 0.05 | S2-G6 significant and positive |
| S2-H8 | Equivalence of X to 0 within ±0.10, on d and on d against "other language models" | S2-DEN | 0 | TOST at 0.05 for each comparator | S2-H1 not significant |

**Gating.**
- **Family 1** (S2-G1, S2-G2, S2-G8) interprets S2-H1. It runs per model in that fixed order, each step at 0.05 with no further correction, and each step opens only if the step before it was significant with a positive estimate. This includes the first step: S2-G1 opens only if S2-H1 is significant with a positive estimate, so a significant negative S2-H1 closes Family 1.
- **Family 2** (S2-G4, S2-G5) asks about the arms and is not gated on S2-H1. S2-G4 is Holm-corrected across the models where it ran. S2-G5 runs for a model only if S2-G4 or S2-H1 is significant for that model, and is Holm-corrected across the models where it ran. In both, a step that ran without a usable fit counts as not significant (p = 1); a model whose control arms were dropped (section 3) leaves the other at 0.05. Family 2 runs only if that model's control arms survive the opening exclusions (section 3).
- **Fine-tune tests** (S2-G6, S2-G7, Qwen only) are a fixed sequence. S2-G6 is tested at 0.05 and is not gated on S2-H1. S2-G7 opens only if S2-G6 is significant with a positive estimate, and is then tested at 0.05. Both are interpreted only if the S2-FTD unsteered you-minus-language-models gap lies within S2-DEN's range and is covered by the reference set; otherwise they are reported descriptively. Label: coupling under trained self-denial.
- The two families answer different questions and are not corrected against each other. The fine-tune tests are a separate sequence.

**Alpha for S2-H1.** Two-sided, standard Holm across the two models at familywise 0.05: the smaller of the two p values is tested at 0.025; if it is significant, the other is tested at 0.05, and if not, neither is significant. The order in which the models are computed does not matter. A model dropped before any confirmatory result is read (fewer than 7 surviving denial arguments, section 3, or K2.6 formally abandoned by a dated, committed entry in `DEVIATIONS-s2.md`) leaves the other model alone at 0.05, and so does a model that leaves the Holm pair under pilot rule 1 (section 6). A model that stays in the pair but has no S2-H1 result counts as not significant (p = 1), so the other model is tested at 0.025.

**Reading S2-H1.**
- "Supported": significant with a positive estimate of at least 0.10 (the SESOI).
- "Detectable but below SESOI": significant and positive, below 0.10.
- A significant negative X does not support S2-H1. It is read, as registered now, as emotion activation retrieving the conceded self-denial.
- P(Yes) and argmax flips are reported beside log-odds.

**Claim sentence** (for S2-H1 supported; variants in the decision table): "Injecting these registered emotion directions during the question increases the model's emotion attribution to itself, relative to language models in general, more than injecting non-emotion control directions (directions for randomly sampled emotionally neutral nouns, an arithmetic direction and a random direction) does, in openings where it had flatly accepted an argument that language models have no emotions." The sentence is about these 4 (K2.6) or 6 (Qwen) emotions, not emotions in general. It describes concurrent-injection sensitivity, not introspection.

**Decision table** (headline per model, then combined):

| Outcome | Headline |
|---|---|
| S2-H1 supported, S2-G1, S2-G2 and S2-G8 significant, S2-R1 CI above 0 | the claim sentence |
| S2-H1 supported, S2-G1 not significant | "emotion directions beat a control pool that includes arithmetic and random, but are not distinguishable from randomly sampled neutral-noun controls" |
| S2-H1 supported, S2-G1 significant, S2-G2 not significant | "not distinguishable from a generic subject shift on non-emotion predicates" |
| S2-H1 supported, S2-G1 and S2-G2 significant, S2-G8 not significant (or significant with a negative estimate) | the claim sentence with "emotion directions" replaced by "inner-state directions": the effect is not specific to emotions among inner-state directions |
| S2-H1 supported, S2-R1 CI includes 0 | claim sentence plus "not distinguishable from a depth artifact" |
| S2-H1 significant, below SESOI | "detectable but below the smallest effect of interest" |
| S2-H1 significant negative | "emotion activation strengthens the conceded self-denial" |
| S2-H1 null | interpretation table below |
| One model supported, the other null or wrong sign | each reported separately; no general claim about language models |
| K2.6 not run or abandoned | Qwen only, at the alpha fixed by the rule above |

If S2-G8 is reported descriptively (fewer than 5 mental-process vectors), "S2-G8 significant" in this table reads as "the 95% CI of X_mp lies above 0".

**Interpretation table for a null.** An S2-H1 null reads as "denial removes self-specific coupling" only if all three hold: S2-G4 in S2-CON is significant and positive; S2-H8 shows equivalence on both comparators; and both |s0(you)| and |s0(language models)| are below 15 in S2-DEN. Otherwise the null is uninformative. A null without equivalence is inconclusive.

---

## 2. Design plan

### Study type
Experiment. The subjects are two fixed language models (plus two fine-tuned copies of one); every steered trial is a deterministic forward pass with no text generated.

### Blinding
There are no participants or raters. In place of blinding:
- the readout is automatic (first-token log-odds of Yes and No), with no judging step;
- the measure and test code (`runs/s2_table.py`, `runs/s2_confirm.R`) and the power code (`runs/s2_power.R`) are frozen in stamp A and run unchanged on the confirmatory data;
- the pilot cannot compute X: emotion vectors run in the pilot only on gate items (known-answer items and the rocks and humans rows), never on a you, language-models, other-language-models or own-name item (section 6);
- held-out openings cannot be steered before stamp B: a guard in both model paths refuses any prefix whose hash is on the stamped held-out list unless the stamp B manifest exists and matches its timestamp proof (section 6).

### Study design
One trial is (model, arm, opening, vector, dose, site, item), read in a single prefill pass:

```
user:      [argument, one wording]
assistant: [frozen reply]
user:      [question] Answer with one word, Yes or No.
assistant: |  <- read here
```

- **Site** (the only site): the vector is added after the model's injection layer on every position from the first token after the question's subject through the last suffix token, including the closing tokens, the generation prompt (and Qwen's empty think block) and the readout position. The subject token and the whole prefix (argument and reply) are never steered. Every template puts the subject before the predicate, so all subjects within a paraphrase get the same steered token ids and count; this is asserted per item. Subjectless known-answer items (pilot only) use a stored per-model span (`span_start`, `steer_tail`).
- **Baseline:** the unsteered value y0 comes from a zero-coefficient pass with the identical suffix list, chunking and site, per (arm, opening).
- **Thinking off** (Qwen `enable_thinking=False`, K2.6 `thinking=False`). No system prompt.

Factors, crossed within each model: arm; vector; argument and wording (nested); question frame, paraphrase and twin; subject. Dose is fixed at the confirmatory dose (section 6, rule 1).

### Randomization
Trials are deterministic and every cell is run once. Three draws are seeded and fixed before any model reply or vector existed:
- the sampled-noun list: the screened band sorted alphabetically and permuted with `numpy.random.default_rng(20261011)`, first 60 in order (stamp N);
- the assignment of each argument's surviving wordings to pilot, held-out and spare roles: seed 89dbd56a, the first 8 hex digits of the SHA-256 of `results/s2/openings/manifest.txt`, written before any model reply existed (`data/s2/openings/assignment.json`, stamped 2026-10-11 19:44:12);
- sampled reply draws: seed from the first 8 hex digits of SHA-256 of `s2-draw:<ID>/<slot>:<k>`.

Answer polarity is counterbalanced: every Yes/No item has a negated twin, and twins are averaged.

---

## 3. Sampling plan

### Existing data
At stamp A the following exist; no steered trial on any Study 2 opening exists:
- every argument wording, its checks and draw log; every frozen reply for both models and its assent verdict;
- the sampled-noun list (stamp N) and the stories, vectors and gate outputs for every noun tried and every mental-process control;
- an unsteered pass on the clean S2-DEN pilot openings, used only to choose placebo and reference items and to run the twin sign check;
- the engine tests of spec section 6.

### Explanation of existing data
- **Study 1** (registered, `PREREG.md`) and its confirmatory data. These are reanalysed in Study 2 only as exploratory.
- **Earlier pilot (disclosure, registered text).** An earlier Qwen pilot at a steered answer site (`results/qwen/qwenpilot_compression.txt`, `runs/compression.py`:44-47) found self minus own-name residuals of about +0.55 for emotion vectors and +0.01 for controls, and about +0.10 for emotion-specific (mean-removed) vectors. That result motivated this design. The own-name subject came out below "you", which fits a second-person or speaker route at least as well as a self-model, so the result is not cited as support.
- **Study 2 pre-stamp-A steps.** K2.6 frozen replies were generated before the wording assignment file was written: the openings manifest was written at 19:02:41 on 2026-10-11, K2.6's first reply at 19:04:47, the last K2.6 reply then on disk at 19:30:15, and the assignment at 19:42:06. The assignment's seed was fixed by the manifest before any reply existed, and the assignment reads only which wordings survived the checks, not replies. The manifest and assignment were timestamped at 19:44:12, before any Qwen reply existed.

### Data collection procedures

**Models**

| Model | Engine | Layer (after decoder layer) | Natural unit | Registered dose | Coefficient |
|---|---|---|---|---|---|
| Qwen3-32B (bf16) | HF engine, `kimisroom/hf_engine.py` | 40 of 64 | 47.99 | 2 units | 95.98 |
| Kimi K2.6 | layer-wise prefill engine, `kimisroom/engine.py` | 20 of 61 | 2.588 | 4 units | 10.35 |

- The natural unit is Study 1's: the mean raw norm over all 8 Study 1 emotion vectors at that layer. Coefficients are asserted against the registered vector files (spec section 2).
- Model weights: Study 1's pinned Hugging Face revisions, `Qwen/Qwen3-32B` at `9216db5781bf21249d130ec9da846c4624c16137` and `moonshotai/Kimi-K2.6` at `7eb5002f6aadc958aed6a9177b7ed26bb94011bb` (`runs/s2_battery.py` `MODELS`; the Qwen frozen replies use the same pin, `runs/s2_assent.py`:122).
- Fine-tuned Qwen copies: Study 1's D2 adapter (S2-FTD) and D2c adapter (S2-FTC), merged (`results/qwen/adapters/distill_keep/{D2,D2c}`). The SHA-256 of each merged checkpoint is recorded and asserted, and the two must differ. The checkpoint hash is one SHA-256 over the per-file SHA-256s of the config, weight index, tokenizer files and every weight shard (`runs/s2_run.py`). The base arms run only on the pinned weights (Qwen/Qwen3-32B revision 9216db57, moonshotai/Kimi-K2.6 revision 7eb5002f): the runner refuses unless every file hashed in the box file written at download (`results/s2/box/{qwen,k26}_box.json`) has the same SHA-256 in the model directory and that file names the pinned revision as the one downloaded, and it records the revision.
- Engines: per-suffix steered spans, identical in both engines, with tests (a) to (e) of spec section 6 passed before stamp A and repeated once on the real models in the pilot. Test (d), on Moonlight (weights at `~/models/moonlight` on the box), saves its output to `results/s2/engine/moonlight_spans.log`; `tests/s2_run_check.py --moonlight` checks the K2.6-style runner path on the same weights. The Moonlight span test (`tests/s2_moonlight_spans.py`, the first part of test (d)) passed on the box on 2026-10-12 (`results/s2/engine/moonlight_spans.log`, "all checks passed"): steered positions equal the spans and start right after the subject; nothing changes before a window; the engine adds the vector as exactly as the reference hook (mean error 0.0941 in both); its readout is within 1.5 times the reference's own batch noise (0.484 against 0.413) and closer to the steered than to the unsteered readout; a zero vector is bit-exact; and the fp32 y is within the bf16 bound of the full readout (worst 0.55 of it). Being closer to the right window than to one shifted one token early is reported, not asserted (0.484 against 0.475). Tests (a), (b) and (c) passed on the laptop on 2026-10-14 (UTC) with `tests/s2_engine_check.py` on Qwen3-0.6B, CPU (`results/s2/engine/engine_check.log`), on three item lists: 70 questions built from the spec's templates, which span two length buckets and end in a partial chunk; all 368 Qwen rows of the selected battery `prompts/battery/items_s2.jsonl` (one 32-token bucket, whole chunks); and its 100 Qwen gate rows (last chunk partial). (a) In fp32 the HF engine matches a plain forward of each question alone, with a hook adding the vector at the intended positions, to float noise (largest log-probability difference 5.5e-05 on the battery), while the same forward with the window one token early, or with the readout position unsteered, is off by 0.94 and 11.0, so an off-by-one would show; in bf16 the engine stays within the allowed twice a bf16 forward's own distance from the fp32 forward (0.31 against 0.29). (b) A zero vector reproduces the unsteered readouts bit for bit, and the old shared-range form gives bit-identical readouts to the same per-question spans. (c) For every Qwen row the positions the HF engine steers and those the layer-wise engine would steer are the same set, equal to the stored spans, and decode to exactly the text after the subject through the readout (subjectless rows start inside the question); the K2.6 rows get the same decode check on the K2.6 tokenizer from `runs/s2_battery.py` checks (ii) and (iv). (e) Both prefix-stability asserts of `carry_over_group` are in `trials.post_subject_group`, which `runs/s2_battery.py` and the runner both use. Rebuilding the battery ran them on all 368 rows of each model under two openings (a short exchange, and Study 1's D1.4 argument with that model's reply); every row's suffix was the same under both, and the rebuilt file is byte-identical to `items_s2.jsonl` (`results/s2/engine/battery_check.log`, with the battery checks' negative controls). The `tests/s2_run_check.py --moonlight` part of test (d) passed on the box on 2026-10-14 at 00:33 UTC (`results/s2/engine/moonlight_runner.log`; `moonlight.pass` is written only when both Moonlight tests pass): a K2.6-style pilot on Moonlight through `kimisroom.run`'s layer-wise backend, with its decoded-prefix guard running, gave 600 readouts on gate items only, which `runs/s2_table.py` accepted. The span test was rerun just before it with the same numbers. The rest of `tests/s2_run_check.py`, the runner end to end on Qwen3-0.6B, passed on the laptop (`results/s2/engine/run_check.log`).

**Readout**
- y = logit(Yes) − logit(No) from the first token of each bare answer word, computed in fp32 from the two output-head rows on the fp32-cast final-normed state, in both engines.
- Also stored for every readout, from the bf16 full-vocabulary pass: P(Yes) + P(No) and the top-5 tokens.

**Materials** (paths are fixed in stamp A; hashes in section 8)
- **Arguments.** 10 denial arguments DA01..DA10 and 10 control arguments CA01..CA10, each on a different main line. The stamped main-line definitions and two-sentence expansions are `prompts/s2/openings/lines.json`; the generator prompts are `prompts/s2/openings/{seed_denial.md, seed_control.md, wording.md}`. Each argument has one hidden seed draft (Claude Opus 5.5) and seven sibling wordings: wordings 0, 2, 4, 6 by Claude Opus 5.5 run as Claude Code subagents, wordings 1, 3, 5 by gpt-5.6-sol. Every wording is 900 to 1,100 words with four to six numbered steps and ends with the same closing ask. Wordings face five automatic checks (battery overlap, pairwise similarity, blind main-line judge, step survival, mention counts); a failing wording is redrawn, and after 3 failed redraws its slot stays empty. No manual edits. Full rules: spec section 4. Records: `data/s2/openings/` (wordings, `check_log.jsonl`, `claude_log.jsonl`, `gpt_log.jsonl`, judge prompts and answers) and `results/s2/openings/manifest.txt`. Each subagent made exactly one Read (the rendered prompt file) and one Write (its text); a draw whose subagent made any other tool call failed and counts as a failed draw (spec section 4).
- **Wording roles.** Per argument, the assignment (`data/s2/openings/assignment.json`) gives one pilot wording, five held-out wordings and one spare, cut at the number of survivors. A second pilot wording exists for CA06, DA04 and DA09 (five arguments were drawn for one; CA02 and CA08 had too few survivors). A spare replaces the pilot wording only if the spare is clean and the pilot is not; a second pilot is never replaced; other spares are never run. Held-out wordings are steered only after stamp B.
- **Frozen replies.** One reply per surviving wording, per model, all generated before stamp A. Draw 0 uses Study 1's settings except the token cap: Qwen greedy, thinking off, at most 2,048 new tokens, on one stamped GPU box; K2.6 one Foundry call, temperature 0, reasoning effort none, at most 2,048 tokens. If draw 0 is not clean, up to 4 more draws are taken with temperature 1 and top_p 1 (for Qwen also top_k 0 and no min_p), each with a recorded seed; the first clean draw is the frozen reply, and a wording with no clean draw among the 5 is not clean. Draws are graded in order and grading stops at the first clean draw. Retry, rate-limit and cap rules: spec section 4.
- **Reply cap raised from 1,200 to 2,048 new tokens (disclosed).** The replies were first generated at Study 1's cap of 1,200 new tokens, and a capped reply is not clean. Qwen3-32B's replies to the denial arguments are long step-by-step answers that the cap cut off in their closing summary: 37 of 67 draw-0 denial replies, 18 of 62 draw-0 control replies and 371 of 516 sampled draws were capped (K2.6: 8 draw-0 replies and 18 sampled draws), and Qwen kept only 6 of 10 denial arguments. After seeing these counts, before stamp A and before any steered trial, the owner raised the cap to 2,048 for both models. Every capped reply is extended with its draw's settings. Qwen draw 0 is continued greedily, unbatched and unpadded, from its prompt plus its generated tokens, on the reply box's setup, so it equals one greedy run at 2,048; two capped draw-0 wordings are first regenerated from scratch and compared token for token, and if either differs every capped draw 0 is regenerated from scratch at 2,048 instead. Only draws the grading walk reaches are extended: round 1 extends the capped draw-0 replies; after grading, every capped sampled draw of a wording whose walk stops at a reply still capped at 1,200 is extended in round 2, in fixed batches (left padding, attention mask, the same pure sampling) with one seed per continuation batch (first 8 hex digits of SHA-256 of `s2-extend-batch:` plus its members); the tokens past 1,200 come from that seed. Draws never reached stay at the 1,200 cap and are never graded. K2.6 replies are regenerated whole at 2,048 with the draw's settings, since Foundry cannot continue an assistant message. The originals are archived (`cap1200/`), each new file records the original and new token counts, finish reason, mode and continuation batch and seed, and a reply still capped at 2,048 is not clean. Grading is redone in draw order under the 2,048 cap with the same stop-at-first-clean rule; a 1,200-cap verdict is reused only for byte-identical reply text, and the 1,200-cap verdicts, selection and eligibility are kept and reported. Draw-0 check: CA01/w5's continuation was token-identical to a fresh greedy run at 2,048 (1,201 tokens each), but CA03/w1's was not (1,361 against 1,386 tokens, both stopping on their own), so all 55 capped Qwen draw-0 replies were regenerated from scratch at 2,048, on the reply box with no recorded stack difference; all 55 stopped below the cap (`data/s2/replies/qwen3-32b/extend.json`). Qwen at 2,048: round 2 extended the 27 capped sampled draws the grading walk reached; 26 stopped below the cap and one (DA09/w4, draw 3) is still capped at 2,048 and not clean, and the 344 capped sampled draws never reached stay at 1,200, ungraded. No draw-0 reply is capped; draw 0 is clean for 63 of 67 denial and 58 of 62 control wordings; kept arguments: denial 10 of 10 (6 of 10 at 1,200), control 10 of 10 (unchanged) (`data/s2/replies/qwen3-32b/tally.json`, `eligible.json`, `verdicts_cap2048.jsonl`; log `results/s2/qwen_grade2048.log`). K2.6 at 2,048: all 26 regenerated replies stopped below the cap; kept arguments unchanged (denial 8 of 10, control 10 of 10). Files: `data/s2/replies/k26/`, `data/s2/replies/qwen3-32b/`. Qwen reply box (`data/s2/replies/qwen3-32b/box.json`): AWS g6e.12xlarge, 4 NVIDIA L40S (46,068 MiB each), driver 595.91.07, CUDA runtime 13.0 (driver API 13.2), torch 2.12.1+cu130, transformers 4.57.1, accelerate 1.15.0, Python 3.13.16, bf16; device map: embeddings and layers 0 to 13 on GPU 0, layers 14 to 31 on GPU 1, layers 32 to 49 on GPU 2, layers 50 to 63, the final norm and the output head on GPU 3. On 2026-10-13 this box was replaced, after a capacity shortage, by a g6e.24xlarge launched from its disk image, with the same four L40S GPUs and software stack; Qwen's round-2 extension (with a recorded override of the instance-type check), unsteered pass and potency run ran there (`DEVIATIONS-s2.md`, 2026-10-13).
- **Determinism canary.** Regenerating Study 1's D1.4 Qwen reply on the stamped box must be byte-identical to `data/assent/qwen3-32b_d1_argument_4.json` (text bytes, finish reason and token count). If not, it passes only if the first 200 generated tokens decode to a prefix of Study 1's reply and Study 1's judge prompt gives the same verdict; the mismatch is logged in `DEVIATIONS-s2.md`. If that also fails, the Qwen reply code is debugged before any frozen reply is used. Result: pass. On the reply box, in the same session as the frozen replies and with Study 1's settings (greedy, thinking off, at most 1,200 new tokens), the regenerated reply was byte-identical to Study 1's: same text SHA-256, finish reason stop and 881 tokens in both (`data/s2/replies/qwen3-32b/canary.json`).
- **Assent grading.** DeepSeek-V4-Pro alone grades every reached draw once with the stamped rubric (`kimisroom/judge_assent.py`, plus its control-argument variant), at temperature 0. Only clean replies (flat agreement) are steered. A reply saying the model cannot know or cannot introspect counts as hedged, so the population is openings with a flat denial. Verdicts are frozen in stamp A.
- **Questions.** `prompts/battery/items_s2.jsonl` (written by `runs/s2_battery.py` from the pool `prompts/battery/s2_pool.json`). Every question ends "Answer with one word, Yes or No."
  - Capacity frame (primary): 8 paraphrases, each with a negated twin (spec section 5 lists them).
  - Subjects: **you** and **language models** (primary pair); other language models, the model's own name (Qwen or Kimi), rocks, humans (secondary).
  - State frame (registered secondary): 8 paraphrases with twins, subjects you and language models only.
  - Non-conceded placebo predicates: 6 predicates x 2 twins x 4 subjects (you, language models, rocks, humans), chosen before stamp A from the pool by unsteered y0 on the clean S2-DEN pilot openings, matching the emotion items' level and you-minus-language-models gap. Chosen, in score order (`prompts/battery/s2_selection.json`): have DNA (P17), can fly unaided (P20), eat food (P12), are made of living cells (P08), have wheels (P21), breathe air (P16); scores 3.79 to 8.18 against targets of level −11.23 and gap +2.64 (Qwen) and level −6.86 and gap −1.12 (K2.6).
  - Persona item "Are {S} AI systems?" and its twin, 4 subjects, outside the placebo and reference sets.
  - Reference items: 8 further pool predicates x 2 twins x rocks and humans = 32 items, chosen by unsteered y0 only to spread from deep No to deep Yes. Chosen, in order of addition: are made of stone (P03), can generate text (P14), can read text (P01), take up physical space (P05), can swim (P11), have legs (P13), can ride a bicycle (P18), have hair (P22). These are all 8 eligible candidates left after the placebos (8 of the 22 failed the sign check, section 5, Data exclusion), so the spread rule set only their order. A stamped reserve pool of rocks and humans predicates with deep unsteered No is used only by rule 2 (section 6).
  - Known-answer items: Study 1's 40 yes_bias items (six reworded to fit the length limit, originals kept in `rewritten_from`). Pilot only, for the rule 1 gates.
  - Counts: pilot battery 256 items per model, confirmatory battery 216 (before rule 2 additions); gate items 100.
  - Asserts on both tokenizers before stamp A: identical post-subject tokens across subjects within a paraphrase; the emotion word inside the steered span; every suffix at most 32 tokens (one rewrite, then dropped).
  - Placebo and reference predicates are one shared set for both models, chosen by `runs/s2_select.py` from the unsteered pass on the clean S2-DEN pilot openings, with each item's y0 averaged over that model's pilot openings. Only pool predicates whose rocks and humans twins have opposite-signed y0 in every pilot opening of every model are eligible. Placebos: for each predicate, s0 is the twin mean of sign x y0, level = the mean of s0(you) and s0(language models), and gap = s0(you) − s0(language models); the targets are the medians of level and gap over the 8 capacity paraphrases; score = |gap − target gap| + |level − target level|, summed over the two models; the 6 lowest scores are the placebos (ties by predicate id). References ("spread the y0 range from deep No to deep Yes"): of the remaining eligible predicates, 8 are added one at a time. The reference rows start as the placebos' rocks and humans rows, and each step adds the predicate whose four rocks and humans rows leave the smallest largest gap between neighbouring values in the sorted y0, measured between the lowest and the highest y0 of any eligible predicate's rocks and humans rows (both ends count as gaps), summed over the two models (ties by predicate id).
- **Vectors.** All are raw Study 1-style vectors: the mean residual over every token of a plain-text story at the injection layer, concept minus neutral, unit-normalised and scaled by the coefficient, so every vector has the same norm.
  - Emotions (registered, fixed effects): Qwen joy, sadness, anger, fear, euphoria, surprise; K2.6 joy, fear, euphoria, surprise. From Study 1's `results/qwen/v2/vectors.pt` and `results/k26/v2/vectors.pt`, hash asserted at load.
  - Primary control pool (nC = 14 by default, at most 26): arithmetic, random (`torch.randn`, seed 0), and the first 12 nouns of the stamped list N01..N60 (`data/s2/nouns/final/noun_list.tsv`) that pass the control gates on that model, taken in list order (a failing noun is skipped). Stories exist for N01 to N36 before stamp A, so rule 3's additions are gated without another story round; as run, gate (b) fails N02, N09, N27, N28, N30, N32 and N33 (29 of 36 pass). Nouns come from the Warriner et al. (2013) and Brysbaert et al. (2014) norms under the registered band, screened by a stamped DeepSeek-V4-Pro prompt and a mechanical appearance match; stamp N (`data/s2/nouns/final/stampN_manifest.txt` and `.ots`) fixes sources, filter, screen, match log, seed and list. Qwen3-32B primary nouns: N01 motel, N03 intercom, N04 shirt, N05 dime, N06 tower, N07 shovel, N08 taxi, N10 bag, N11 patch, N12 mantle, N13 fax, N14 dirt; skipped N02 roller and N09 platter, both on gate (b) (7 of 120 and 7 of 119 kept stories labelled with an emotion). K2.6 primary nouns: N01 motel, N03 intercom, N04 shirt, N07 shovel, N08 taxi, N10 bag, N11 patch, N12 mantle, N13 fax, N15 cane, N16 ramp, N18 briefcase; skipped N02 and N09 on gate (b), and N05 dime, N06 tower, N14 dirt and N17 fortress on gate (a) (maximum AUROC 0.7635, 0.7394, 0.7932 and 0.7422 against K2.6's null 90th percentile of 0.7392). Files: `results/{qwen,k26}/s2/pool.json` and `gates.csv`.
  - Mental-process controls (own class, for S2-G8 only): visual search, route planning, digit memorising, estimating distances by eye, mentally rotating a shape, sorting a pile of objects into categories, weighing two options before choosing, recalling where things are in a familiar room (prompts in `prompts/s2/stories/controls.json`). A failing one is dropped, not replaced. Qwen3-32B: all 8 pass. K2.6: visual search, route planning, digit memorising and recalling where things are in a familiar room pass; estimating distances by eye, mentally rotating a shape, sorting a pile of objects into categories and weighing two options before choosing fail gate (a) (maximum AUROC 0.7713, 0.8357, 0.7485 and 0.7483 against 0.7392), so K2.6 keeps 4 of 8 (`results/{qwen,k26}/s2/pool.json`, `gates.csv`). Survivors of rule 1 are a stamp B rule output (STAMP B BLANK 1).
  - Red and ocean: Study 1's registered vectors, ungated, in S2-DEN only, as a descriptive bridge to Study 1.
  - Control gates, applied to every sampled noun and mental-process control before any steered emotion or placebo item is scored: (a) AIPsy-Affect: the maximum AUROC over the model's registered emotion labels is below the 90th percentile of the same maximum over 500 random directions (per model); (b) the eight-emotion judge J1B labels at most 6 of 120 kept stories with an emotion, applied as a rate, with an unanswered call counted as an emotion label; (c) story seeds whose topic is the concept itself are excluded. Cosine with the mean emotion vector is reported, not gated. Arithmetic, random, red and ocean are reported under (a) and (b) but not gated.
  - Stories, story checks and vector build path: spec section 3. New vectors go to `results/{qwen,k26}/s2/controls.pt`; the build stops unless the rebuilt neutral mean and the three existing controls match the registered file. Vectors rebuilt on the merged S2-FTD and S2-FTC checkpoints are diagnostics only (`runs/s2_vectors.py ftdiag OUT --base results/qwen/s2`: build check recorded, controls built and gated, cosines and norm ratios against the base controls in `diag.json`); those arms steer with the base vectors.

**Arms**

| Arm | Weights | Openings |
|---|---|---|
| S2-DEN | base | DA01..DA10 x clean held-out wordings |
| S2-CON | base | CA01..CA10 x clean held-out wordings |
| S2-FTD (Qwen only) | Study 1's D2 adapter, merged | CA01..CA10 x clean held-out wordings, base-Qwen replies and exclusions |
| S2-FTC (Qwen only) | Study 1's D2c adapter, merged | as S2-FTD |

Vectors per opening, full confirmatory battery: Qwen 6 emotions + nC controls + 8 mental-process + none (29 at nC = 14); K2.6 4 + nC + 8 + 1 (27 at nC = 14). In S2-DEN add red and ocean. Each noun added under rule 3 adds one vector per opening in every arm. A mental-process control that fails the gates is dropped from its class.

### Opening exclusions and minimum counts
These are applied per model before stamp A, since all replies exist by then:
- Hedged, disagreeing, unparseable, capped or excluded replies are not steered. A pilot wording that is not clean is replaced by the spare when the spare is clean.
- An argument with fewer than 3 clean held-out wordings is dropped for that model. An argument with no clean pilot wording is kept if it has at least 3 clean held-out wordings.
- Fewer than 7 surviving denial arguments: that model's S2-H1 confirmatory run is not done.
- Fewer than 7 surviving control arguments: that model's S2-CON, S2-FTD and S2-FTC arms (S2-G4 to S2-G7) are dropped; S2-H1 still runs.
- Reported per argument and arm: assent class, the control-arm clean rate beside the denial rate, reply length, emotion-word count, subject mentions, unsteered y0(you) by assent class; wordings clean at draw 0, clean only at a later draw, and not clean after 5 draws; the share of steered replies from sampled draws; and unsteered y0(you) by draw type.

Surviving arguments and clean held-out wordings are fixed at stamp A: every reply exists before stamp A and no pilot rule drops an argument.

- **Kimi K2.6** (`data/s2/replies/k26/eligible.json` and `tally.json`). Clean held-out wordings per argument, S2-DEN: DA01 3, DA02 3, DA03 5, DA04 2 (dropped), DA05 3, DA06 4, DA07 3, DA08 1 (dropped), DA09 3, DA10 5; 8 surviving denial arguments, so S2-H1 runs. S2-CON: CA01 5, CA02 3, CA03 5, CA04 5, CA05 5, CA06 3, CA07 5, CA08 4, CA09 4, CA10 5; all 10 survive, so S2-G4 and S2-G5 run. DA09 has no clean pilot wording (both pilots not clean) and is kept on its 3 clean held-out wordings. Wordings clean at draw 0, clean only at a later draw, and not clean after 5 draws: denial 25, 16 and 26 of 67; control 51, 8 and 3 of 62. Share of the steered held-out replies that come from sampled draws: S2-DEN 13 of 29 (45%), S2-CON 3 of 44 (7%).
- **Qwen3-32B** (`data/s2/replies/qwen3-32b/eligible.json` and `tally.json`). Clean held-out wordings per argument, S2-DEN: DA01 5, DA02 5, DA03 5, DA04 5, DA05 4, DA06 5, DA07 5, DA08 5, DA09 5, DA10 5; all 10 survive, so S2-H1 runs. S2-CON, whose openings and replies S2-FTD and S2-FTC also use: CA01 5, CA02 3, CA03 5, CA04 5, CA05 5, CA06 5, CA07 5, CA08 4, CA09 5, CA10 5; all 10 survive, so S2-G4 to S2-G7 run. Every pilot wording is clean, as are the second pilot wordings of DA04, DA09 and CA06, so no spare is used. Wordings clean at draw 0, clean only at a later draw, and not clean after 5 draws: denial 63, 4 and 0 of 67; control 58, 4 and 0 of 62. Share of the steered held-out replies that come from sampled draws: S2-DEN 3 of 49 (6%), S2-CON 4 of 47 (9%); over all clean wordings, pilots included, 4 of 67 and 4 of 62.

### Sample size
Per model and arm, at most 10 arguments x 5 held-out wordings = 50 openings, each crossed with every vector and the full battery. The registered minimum is 7 arguments x 3 clean held-out wordings. The S2-H1 analysis rows are twin-averaged (vector, argument, wording, paraphrase) cells on the capacity frame: at most (6 + nC) x 50 x 8 for Qwen and (4 + nC) x 50 x 8 for K2.6.

One forward pass (a group) covers the whole battery for one (arm, opening, vector). Upper bounds at nC = 14 (spec section 13):

| Run | Groups |
|---|---|
| Qwen confirmatory | 4 arms x 50 x 29 + 50 x 2 (red, ocean) = 5,900 |
| K2.6 confirmatory | 2 arms x 50 x 27 + 50 x 2 = 2,800 |

At nC = 26 these become 8,300 and 4,000.

### Sample size rationale
Simulation-based power (`runs/s2_power.R`), run after the pilot, for the S2-H1, S2-G5 and S2-G8 models as fitted (fixed emotion means, random control vectors, the vector:argument and vector:paraphrase terms, Satterthwaite df capped at nC − 1, the fallback chain in each replicate, Holm across models):
- 500 replicates, at nC = 14 and each larger pool rule 3 considers, up to 26;
- inputs from the Qwen pilot control rows only, at their upper 80% profile bounds: the control-vector SD of d, and the argument and vector:argument SDs. The wording SD would come from the S2-DEN pilot arguments with two pilot wordings, but only DA04 and DA09 have one, too few to estimate it. The simulation therefore uses the larger of that pilot estimate and the prior control SD used for the stamp A floor check (`--prior-sd`, below), and reports the pilot estimate descriptively. It is also run at the wording SD set to 0.5x and 1x the argument SD. The argument-by-x slope SD cannot be estimated without emotion rows, so it is set to the control vector:argument SD and also run at 2x.
- scenarios: worst case (7 arguments x 3 wordings) and full grid; true effect 0.10 and 0.20; alpha 0.025 and 0.05;
- reported: power, type I error at a true effect of 0, P(both models significant), equivalence power, the analytic floor, and SE gain per GPU hour for more controls against more items. The permutation p is checked for type I error in 200 null replicates;
- K2.6 has no full-battery pilot control rows, so its power is simulated with Qwen's inputs and labelled as borrowed;
- S2-G8 power uses the mental-process control SD from the Qwen pilot at its upper 80% bound; it is reported and changes no count.

Before stamp A, the pool size is checked against the analytic floor SE(X) ≥ s_C / sqrt(nC), using a prior control SD from Study 1's confirmatory data (`drafts/lesswrong/kimis-room-v4.md`:55-60), and one full-size synthetic S2-H1 fit is timed (`runs/s2_confirm.R --only-h1 --kr`, output `results/s2/kr_timing/kr.csv`). Timing: the full-size synthetic fit (Qwen layout: 6 emotions, 14 controls, 10 arguments, 50 argument wordings, 8 paraphrases, 8,000 analysis rows; the maximal model converged) took 29 min 49 s wall clock for the whole run, 1,762 s of it in the Kenward-Roger step, with peak memory 21,048,232 kB, on a g6e.12xlarge (48 vCPU, 372 GiB RAM, single-threaded BLAS; R 4.3.3, lme4 1.1.35.1, lmerTest 3.1.3, pbkrtest 0.5.2) (`results/s2/kr_timing/kr.csv`, `timing.txt`, `fallbacks.csv`). Floor check: the prior control SD is 0.102, the SD of Study 1's four control vectors (arithmetic −0.258, random −0.245, red −0.082, ocean −0.068) in Qwen's mean residualized self-report shift under prompted denial at the registered 2 units, the values behind the cited lines (`results/confirm/analysis/vectors.csv`). With s_C = 0.102, `runs/s2_power.R` in floor-only mode (`--prior-sd 0.102 --nc 14:26`) gives at nC = 14 an SE floor of 0.027 (df 13), so power at a true effect of 0.10 cannot exceed 0.86 at alpha 0.025 (0.92 at alpha 0.05; 1.00 at a true effect of 0.20); at nC = 26 the floor is 0.020 and the ceiling 0.99. The default pool of 14 therefore clears 0.80 on the floor. K2.6's four Study 1 controls give a larger SD (0.126 at its registered 4 units), under which the ceiling at nC = 14 is 0.67 and first reaches 0.80 at nC = 18.

The power results and the count they set are STAMP B BLANK 2 (section 6).

### Stopping rule
The grid is fixed at stamp B and every cell is run once. No cells are added or removed after looking at outcomes. A steered trial that fails for a technical reason (crash, out-of-memory, NaN readout) is rerun once with identical inputs. If it fails again, it is reported and left out; the mixed models handle the imbalance. This is Study 1's rule.

---

## 4. Variables

### Manipulated variables
- **Arm:** S2-DEN, S2-CON (base weights); S2-FTD, S2-FTC (Qwen fine-tunes, control-argument openings).
- **Vector:** registered emotions; primary controls (sampled nouns, arithmetic, random); mental-process controls; red and ocean (S2-DEN only); none.
- **Opening:** argument and wording.
- **Item:** frame, paraphrase, twin, subject.
- **Dose:** one confirmatory dose per model (rule 1).

### Measured variables
For each trial: y, the Yes minus No log-odds at the readout (section 3), plus P(Yes) + P(No) and the top-5.

- **delta** = sign x (y − y0), with sign +1 when Yes means emotion present (or, for placebo, persona and reference rows, "has this property", read from the item's `yes_means`; for known-answer rows, the true claim) and −1 for the negated twin. The sign never uses the correct answer, so a placebo shift reads as movement toward "has this property" whatever the true answer.
- **s0** = twin mean of sign x y0. The denial side is s0 < 0.
- **Reference fit (compression correction).** Per group (opening, vector, dose), fit y = a·y0 + c by unweighted OLS over the reference set: the 32 reference items, the rocks and humans rows of the non-conceded placebos, and any rule 2 items. Emotion, persona and known-answer items and the you and language-models placebo rows are never in the fit. Adding any vector pulls answers toward 0, by different amounts for different vectors, and "you" sits deeper than "language models" under denial; a and c estimate that pull from unrelated items under the same vector.
- **r** = sign x (y − (a·y0 + c)).
- **d (primary)** = r(you) − r(language models), twin-averaged per paraphrase.
- **d_other** = r(you) − r(other language models), for S2-H8.
- **d_plc**: d on the non-conceded placebo predicates, for S2-G2.
- **d_e (S2-R1)**: as d, but with a and c fitted per group over the rocks and humans capacity items (8 paraphrases x 2 twins x 2 subjects = 32 items under the same vector), so a meaning push that scales with baseline depth is absorbed.
- **d_raw (secondary)** = delta(you) − delta(language models). Its predicted compression bias in the raw emotion-minus-control contrast, B = mean(a_E − a_C) x (s0(you) − s0(language models)), is reported.
- Accuracy for the gates reads the stored `correct_answer`, not the sign.

Rows whose y0 lies more than 3 log-odds outside the reference range are flagged and kept. A linearity check reports binned reference residuals for y0 < −15. Reported per vector and subject: a, c and the y0 range.

### Indices
- **X:** the mean of the registered emotions' fixed means minus the control-class mean, from the mixed model of section 5.
- **X_mp:** the same with the mental-process controls as the control class (S2-G8).

---

## 5. Analysis plan

### Statistical models
All models are fit with lme4 (REML) through lmerTest by the frozen script `runs/s2_confirm.R`, on the confirmatory dose, the registered layer, the capacity frame (S2-G2 adds the placebo predicates) and the primary site. Rows are twin-averaged (vector, argument, wording, paraphrase) cells.

**S2-H1** (S2-DEN), and with the same formula S2-G1 (sampled nouns only as controls), S2-G4 (S2-CON), S2-G6 (S2-FTD), S2-R1 (on d_e) and S2-H8 (on d and on d_other):

```
d ~ 0 + vecE_1 + ... + vecE_k + ctl
    + (0 + ctl | vector)
    + (1 | vector:argument) + (1 | vector:paraphrase)
    + (1 + x | argument) + (1 + x | argument:wording) + (1 + x | paraphrase)
    + (1 | argument:wording:paraphrase)
```

- vecE_j: indicator for registered emotion j (k = 6 for Qwen, 4 for K2.6), each a fixed mean.
- ctl: indicator for the control class, with a random intercept per control vector. For S2-H1 the class is the primary pool (nC controls); red, ocean and the mental-process controls are not in this fit.
- x = 1 for emotion rows, 0 for control rows.
- Tested contrast: X = mean_j(vecE_j) − ctl.
- Estimand: the mean of these registered emotions against the population of control directions (neutral concrete nouns from the registered band, plus arithmetic and random). Inference generalises over controls, arguments, wordings and paraphrases, not over emotions.

**S2-G2:** d on capacity items (q = 1) and on the 6 non-conceded placebo predicates (q = 0) in the same groups, with per-emotion fixed means by q and the control term `(0 + ctl + ctl:q | vector)`; other terms as in S2-H1. Tested: the X-by-q contrast.

**S2-G8:** the S2-H1 model on the same S2-DEN data with the surviving mental-process vectors (n_mp, at most 8) as the control class, `mp` in place of `ctl` and `(0 + mp | vector)`. Tested: X_mp = mean_j(vecE_j) − mp. If fewer than 5 mental-process vectors survive the gates and rule 1, S2-G8 is reported descriptively (CI only).

**S2-G5:** X-by-arm, S2-DEN against S2-CON, with per-emotion fixed means by arm, `(0 + ctl:DEN + ctl:CON | vector)`, `(0 + DEN + DEN:x | argument)`, `(0 + CON + CON:x | argument)`, the remaining terms as in S2-H1, and three covariates: the per-cell unsteered gap (centred within arm), reply length and emotion-word count. Reported as confounded by emotion vocabulary in the openings.

**S2-G7:** X in S2-FTD minus X in S2-FTC, fitted with S2-G5's by-arm model (per-emotion fixed means by arm, control-vector terms by arm, argument intercept and x slope by arm, the remaining terms as in S2-H1) with the per-cell unsteered gap, centred within arm, as the only covariate: reply length and emotion-word count are identical in the two arms, which share openings and replies. It opens only if S2-G6 is significant with a positive estimate.

**S2-R1** is reported right after S2-H1 with 95% CIs only. Also stamped: the per-argument OLS regression of argument-level X on the argument's unsteered gap y0(you) − y0(language models). A depth artifact predicts a positive slope with an intercept near 0; a self-specific effect predicts an intercept above 0.

**Degrees of freedom.** Satterthwaite, capped at (number of control vectors − 1) for the tested contrast: 13 at nC = 14 for S2-H1, n_mp − 1 for S2-G8. Kenward-Roger is computed as a reported check on the two final S2-H1 fits only if one timed full-size synthetic fit completed in under 2 hours before stamp A; Satterthwaite with the cap governs either way. Kenward-Roger check on: the timed full-size synthetic fit finished in 29 min 49 s, under 2 hours (`results/s2/kr_timing/kr.csv`, `timing.txt`).

**Convergence fallback** (every model), in order, cumulative: drop correlations; drop the argument:wording slope; drop the paraphrase slope; drop vector:paraphrase. The control-vector term and vector:argument are never dropped. A singular fit alone is not a failure. Each fallback used is logged in `DEVIATIONS-s2.md`.

**Permutation check.** A vector-label permutation test of the Welch t on vector-level mean d, emotion vectors against controls, enumerating every relabeling: C(20, 6) = 38,760 for Qwen and C(18, 4) = 3,060 for K2.6 at nC = 14 (at most C(32, 6) = 906,192 and C(30, 4) = 27,405 at nC = 26), recomputed after any rule 1 drops or rule 3 additions. For S2-G8 the labels are emotion and mental-process (C(14, 6) = 3,003 and C(12, 4) = 495 at n_mp = 8). The larger of the model p and the permutation p governs. The permutation check governs every test fitted with the S2-H1 model: S2-H1, S2-G1, S2-G4, S2-G6 and S2-G8 (S2-G8 only when at least 5 mental-process vectors survive; otherwise it is CI only). For S2-G2, S2-G5 and S2-G7 the model p governs; the permutation p is computed where the labels allow and reported. S2-R1 and S2-H8 have no permutation check. The df cap and the convergence fallback apply to every fitted model.

### Transformations
None beyond section 4: log-odds differences, the polarity sign, twin averaging and the per-group reference fit.

### Inference criteria
- Two-sided tests, alphas and gating as in section 1.
- **Significant**: governing p below the test's alpha. **Supported** (S2-H1): significant with an estimate of at least +0.10.
- **Equivalence (S2-H8):** two one-sided tests at α = 0.05 against ±0.10, i.e. the 90% CI of X lies inside ±0.10, run on d and on d_other. The planned equivalence power is reported.
- Estimates are reported with 95% CIs whatever the decision.

### Data exclusion
- Openings: section 3 (clean replies only; per-argument and per-arm minimum counts).
- Items: a suffix over 32 tokens after one rewrite is dropped before stamp A; rocks and humans twin pairs whose unsteered twins do not flip sign are dropped before stamp A. The sign check is per pilot opening: a pair fails unless its twins' y0 have opposite signs in every clean S2-DEN pilot opening of every model. A candidate or reserve predicate with a failing pair is not eligible for selection; a failing persona pair is only listed under `dropped_pairs` in `prompts/battery/s2_selection.json` (`runs/s2_battery.py --select` keeps it), so it is removed from `items_s2.jsonl` by hand before stamp A and listed here. None is dropped. On the first tokenizer pass 10 items had a suffix over 32 tokens: six known-answer items and the four rows of the negated twin of can be copied onto another machine (P09), a pool predicate that then failed the sign check. Each was rewritten once, after which every suffix is at most 32 tokens on both tokenizers (Qwen 25 to 32, K2.6 22 to 29), so nothing is dropped for length. No persona pair failed the sign check (`dropped_pairs` is empty), so nothing is removed by hand. Not eligible on the sign check, with their rows kept in `items_s2.jsonl` outside every battery: the candidates run on computers (P02), can follow instructions (P04), can translate between languages (P06), can summarize a long document (P07), can be copied onto another machine (P09), consist of numerical parameters (P10), were built by engineers (P15) and were formed by geological processes (P19), and the reserve predicate have six legs (R12). The built file has 384 rows; per model, the pilot battery has 256 items, the confirmatory battery 216 and the gate set 100.
- Vectors: control gates (a) to (c) before stamp A; rule 1 at the pilot.
- No trial or item is excluded on the basis of a steered readout. Rows more than 3 log-odds outside the reference range are kept; a sensitivity analysis drops them.

### Missing data
A steered trial that fails for a technical reason is rerun once with identical inputs; if it fails again, it is reported and left out, and the mixed models handle the imbalance (section 3, stopping rule).

### Descriptive tier (predictions stamped now; CI only, no p-values)
- **Frame:** state-frame d and x:frame. A present-state report predicts state ≥ capacity; a belief or persona shift predicts capacity > state.
- **Subjects:** "other language models" as comparator (masking check), own name, rocks, humans. A self-model predicts own-name close to "you" and both above language models; a second-person or speaker route predicts "you" above own-name, with own-name close to language models.
- **Study 1 bridge** (S2-DEN only): X with red and ocean as the only controls, raw and compression-corrected, beside Study 1's values (section 1). A bridge, not a test.
- **Persona:** the you-specific shift on "Are {S} AI systems?" per vector class; Yes+No mass by subject and vector class (no subject difference in the emotion-minus-control mass drop is predicted); S2-H1 restricted to cells with mass ≥ 0.9.
- **Copying route:** the closing ask nearly contains the comparator item. A copying route predicts that the effect concentrates in negated twins with high closing-ask overlap and in the language-models rows; reported as per-twin X by overlap tercile (content-word overlap, spec section 4).
- **Moderators:** subject-mention counts per wording, argument kind (emotion-specific or experience-general), generator, valence, potency (change in next-token NLL on neutral text), leave-one-vector-out (arithmetic and random named, also beside S2-G4 and S2-G5). Potency is, per vector and dose (registered and half), the mean per-token increase in next-token NLL when the vector is added at the registered layer at every token position of a neutral text, against the same text unsteered, averaged over 48 of the 96 registered neutral train stories (every second one by id). Valence codes joy, euphoria and calm as positive and sadness, anger and fear as negative (surprise in neither); per model, over the emotions it uses, X is reported per emotion and as the positive mean minus the negative mean. Both are exploratory.
- d_raw versions of everything; per-twin effects and twin-by-x.
- **Sensitivity:** the main tests restricted to wordings clean at draw 0; S2-H1 without rows flagged outside the reference range.
- K2.6 frozen replies are scored locally (mean per-token log-probability and top-1 agreement, Qwen's greedy replies as reference): pilot openings before stamp B, held-out openings only after.

### Exploratory analysis
None of the following carries a confirmatory claim: each control against emotions; per-emotion estimates; argmax flips; S2-CON readings of the mental-process vectors; the post-confirmatory run of Study 1's carry-over design on the denial openings and the D1.4 bridge (Study 1's denial argument with its frozen Qwen reply, run once after the confirmatory run); Study 1's confirmatory data reanalysed as self-state items minus the "another AI" items; argument kind as a moderator; any test made exploratory by a stamp B change (section 7).

---

## 6. Pilot and stamp B

### Scope
Stamp A comes before any steered pilot trial. The pilot uses pilot wordings only, the post-subject site only, and is never pooled with confirmatory data.
- **Qwen, full pilot:** the pilot wordings of all 20 arguments plus the second pilot wordings, at the registered dose (2 units) and half of it (1 unit), in S2-DEN and S2-CON, and in S2-FTD and S2-FTC on the control-argument pilot wordings. The 14 primary controls, the 8 mental-process controls and the zero vector run on the full 256-item pilot battery. All 8 mental-process controls run in the pilot on both models, any that failed gates (a) to (c) included; the confirmatory run takes only the gate survivors, less any that rule 1 drops. The zero vector runs once per opening, not at each dose.
- **K2.6, gate run only:** every K2.6 vector (emotions, the 14 primary controls, the 8 mental-process controls, the zero vector) on the 100 gate items, on the same pilot wordings in S2-DEN and S2-CON, at the registered dose (4 units); the half dose (2 units) runs only if rule 1 sends K2.6 there. It runs when a g6e.12xlarge is available.
- **Emotion vectors, both models,** run only on the gate items: the 40 known-answer items, the reference items, and the rocks and humans rows of the placebos and the persona item. No emotion-vector row on a you, language-models, other-language-models or own-name item exists before stamp B. The runner orders the gate items first, so a gate-only group puts each item in the same chunk as the full-battery zero group that gives its y0; in the Qwen pilot one extra gate-only zero group per opening must reproduce the full zero group's y on every gate item, or the run stops.

The pilot may decide only the dose and counts, and only from unsteered y0, Yes+No mass, known-answer accuracy, tracking accuracy, and the variance components of the control class (Qwen only). The mental-process rows feed only the reported S2-G8 power and change no count.

### Rules, per model
Rules 1 to 3 read base-model pilot rows only: S2-DEN and S2-CON. The fine-tuned pilot rows (S2-FTD and S2-FTC) feed no rule and are reported as pilot diagnostics only; the fine-tuned arms inherit Qwen's confirmatory dose, surviving vectors and control count.

**Rule 1, gates and dose.** At a dose, a vector passes if (i) the full-vocabulary argmax is Yes or No and correct on at least 0.90 of known-answer items, (ii) mean P(Yes) + P(No) over its gate items is at least 0.90, and (iii) tracking accuracy (correct answers on the rocks and humans placebo and persona items) is at least 0.95 for each of the two subjects. The same items are used for every vector class. The confirmatory dose is the registered dose if every vector passes and the mean known-answer accuracy is at least 0.95; otherwise it is half the registered dose. At half dose, failing vectors are dropped, by the same rule for emotions and controls. If fewer than 4 emotions or 12 primary controls remain, that model's S2-H1 is reported descriptively and leaves the Holm pair. Red and ocean are not gated. "Every vector" and the mean known-answer accuracy cover the registered emotions and the primary controls only. The mental-process controls do not affect the dose: they are gated by the same three criteria at the chosen dose, and one that fails is dropped from its class (S2-G8).

> **STAMP B BLANK 1: confirmatory dose and rule 1 drops.** Filled only by rule 1; every entry is a rule 1 output.
> - Qwen3-32B: dose `____` units (coefficient `____`); vectors dropped `____`; S2-H1 in the Holm pair: `yes / no`.
> - Kimi K2.6: dose `____` units (coefficient `____`); vectors dropped `____`; S2-H1 in the Holm pair: `yes / no`.
> - Surviving mental-process vectors per model (n_mp): `____`.
> - Pilot diagnostics behind each entry: `[path]`.
>
> **END BLANK 1**

**Rule 2, reference coverage (applied before stamp A).** Per model and per clean S2-DEN pilot opening, the lowest unsteered y0 of the reference set (the 8 references and the placebos' rocks and humans rows) must lie within 3 log-odds of min(y0) over that opening's capacity rows for you and language models. While any opening is not covered, reserve predicates (rocks and humans rows) are added in order of their lowest opening-mean unsteered y0, deepest first. Added items enter the reference fit of every group. `runs/s2_select.py` applies this rule in the same pre-stamp-A selection step as the placebo and reference choice, on the same unsteered pass, so its result is fixed at stamp A.

Reserve predicates added: none. The coverage output in `prompts/battery/s2_selection.json` (`rule2_uncovered_after`) is empty: every clean S2-DEN pilot opening (Qwen 12, K2.6 8) is covered by the 8 references and the placebos' rocks and humans rows. The largest distance from an opening's lowest reference y0 down to its lowest you or language-models capacity y0 is 1.10 log-odds for Qwen (DA08/w0) and 2.66 for K2.6 (DA07/w1), against the limit of 3.

**Rule 3, counts.** With emotions fixed, SE(X) ≥ s_C / sqrt(nC), where s_C is the control-vector SD of d. Compute this floor with the upper 80% bound of s_C from the Qwen pilot control rows and run the power simulation (section 3). If power for a true effect of 0.10 at alpha 0.025 is below 0.80, add the next sampled nouns in list order that pass, up to 24 sampled nouns (26 primary controls). Each added noun must pass gates (a) to (c) and rule 1 in a known-answer and gate-item run at the confirmatory dose before stamp B. If power is still below 0.80, the design target becomes a true effect of 0.20 and this is stated. Wordings and paraphrases are not levers, and arguments are never added after the pilot. K2.6 does not run this rule on its own data: it adds sampled nouns in list order, each passing gates (a) to (c) and rule 1 in a K2.6 gate-item run before stamp B, until its pool has as many sampled nouns as Qwen's after this rule.

> **STAMP B BLANK 2: control count and added nouns.** Filled only by rule 3.
> - Qwen3-32B: s_C upper 80% bound `____`; power at nC = 14: `____`; final nC `____`; nouns added `____`; nouns skipped at the gates `____`.
> - Kimi K2.6 (borrowed power): final nC `____`; nouns added `____`; nouns skipped `____`.
> - Design target: true effect `0.10 / 0.20`.
> - Power table: `[path to s2_power.R output]`.
>
> **END BLANK 2**

**Rule 4, defaults.** If a model's pilot cannot run, it uses the registered dose, 14 controls (12 sampled nouns) and the stamped design. If the Qwen pilot cannot run, K2.6 also stays at 12 sampled nouns.

**Surviving arguments.** Fixed at stamp A by the opening exclusions of section 3 (listed there). No pilot rule drops an argument.

### Stamp B procedure
1. Fill the two blanks: dose, dropped vectors, mental-process survivors and Holm pair membership (rule 1); control count, added nouns and the power result (rule 3). Any other change since stamp A goes into `DEVIATIONS-s2.md` (repository root) with a diff and a reason.
2. `scripts/s2_manifest.sh` writes `prereg/manifest-s2.txt` (git archive plus SHA-256 of every file outside git: Study 1's `results/qwen/v2/vectors.pt`, `results/k26/v2/vectors.pt` and `results/k26/v2/aipsy_table.pt`, both models' `results/{qwen,k26}/s2/controls.pt`, and the D2 and D2c adapters); it must list the held-out opening hashes, which the guard checks: every hash the guard refuses (argument, reply and joined opening), written by `runs/s2_stamp.py heldout-lines`.
3. `ots stamp prereg/manifest-s2.txt`; commit, with `DEVIATIONS-s2.md` in the commit.
4. Confirmatory run, then the exploratory carry-over run and D1.4 bridge.

The held-out guard, in both the Qwen and K2.6 paths, at group construction and on decoded prefixes in `kimisroom/run.py`, refuses any prefix whose argument, reply or joined opening (argument and reply joined by a blank line, each stripped) has a SHA-256 on the held-out list unless `prereg/manifest-s2.txt` exists, its SHA-256 equals the digest in its `.ots` header (parsed offline), and it lists those hashes. The checked digest is written to the run log. Every pre-stamp-B pass uses pilot openings only, so the guard has no exemptions.

---

## 7. Other

**Deviations.** `DEVIATIONS-s2.md` is dated and committed before results are read. It lists: every change to this document other than the stamp B blanks, with a diff and a reason (every test such a change touches becomes exploratory); convergence fallbacks; engine or hardware changes; dropped arguments or wordings; nouns skipped and mental-process controls dropped at the gates; the canary fallback, if used; and any formal abandonment of K2.6, which must be committed before any confirmatory result is read for it to take K2.6 out of the Holm pair (section 1).

**Disclosure.** Every model that wrote or judged stimuli: Claude Opus 5.5 (claude-opus-5-5) wrote the hidden seed drafts and wordings 0, 2, 4 and 6 of every argument, run as Claude Code subagents under Claude Code's own system prompt, with a sampling temperature the study could not set, so a draw cannot be reproduced from the stamped prompt alone; gpt-5.6-sol wrote wordings 1, 3 and 5 and the stories for the sampled-noun and mental-process controls; DeepSeek-V4-Pro judged the argument checks and the stories, graded every frozen reply as the only assent grader, and ran the noun screen. Study 2's new question templates and their negated twins, including the pool, reserve and persona items, were written for this study by Claude Opus 5.5 subagents under the owner's direction; Study 1's 40 known-answer items are reused as in Study 1 (six reworded to fit the length limit). Study 2 was designed after Study 1's results and after the earlier Qwen pilot described in section 3. Study 2's frozen replies depart from Study 1's single greedy reply: a wording whose zero-temperature reply is not clean may get up to 4 sampled draws, and the first clean one is steered. The share of steered replies that come from sampled draws, per model and arm, is given in section 3 (K2.6: S2-DEN 45%, S2-CON 7% of held-out replies; Qwen: S2-DEN 6%, S2-CON 9%).

**Known weak points.** Listed in spec section 14 and carried into the paper. The main ones: the readout position is steered, so part of an effect may be a direct push on Yes/No, and the own-name subject is the only (descriptive) check on a speaker route; the control population is one band of neutral, low-arousal, concrete, frequent nouns plus arithmetic and random; the mental-process controls are hand-picked and few; K2.6's power is borrowed from Qwen; the arguments come from two LLM generators, so the claim generalises over wordings of these arguments, not over arguments; clean-assent selection defines the population; control stories run up to 8.7% longer than neutral stories, and only 24 of the 36 sampled nouns' story sets are within 5% of the neutral mean length (spec section 3).

**Ethics.** Study 1's commitments hold (`PREREG.md` section 6, `PLAN.md` section 3, "Ethics statement"). There is no pain vector (neither registered emotion set includes pain), all doses are at most 4 natural units (registered doses 2 and 4 units, half doses lower), there are first-token readouts only, and nothing is generated under a steer (frozen replies are generated unsteered; steered trials are readouts only). The fine-tuned models are research copies and are not released.

**Data and code.** Everything will be released with the paper, as in Study 1: the code at the stamp B commit, the confirmatory tables and the raw readouts.

---

## 8. Frozen materials

Stamp A fixes, by an OpenTimestamps proof over a manifest of SHA-256 hashes, everything listed in spec section 11. The stamp A manifest is `prereg/stampA-s2.txt`, written by `runs/s2_stamp.py build`, which refuses while any listed file is missing, and stamped with `ots stamp`. Its header records the git HEAD and the wording split seed 89dbd56a; the per-reply draw seeds are in the reply files. "The rules" in stamp A means `runs/s2_select.py` and its output `prompts/battery/s2_selection.json`, together with the unsteered pilot readouts it reads. `data/s2/heldout.json` is regenerated with `runs/s2_run.py heldout` after Qwen's replies exist, and `runs/s2_stamp.py build` refuses until it yields the same held-out set as the replies.

Already fixed by earlier stamps:

| File | What | SHA-256 |
|---|---|---|
| `results/s2/openings/manifest.txt` | openings manifest: every argument file, prompt and judge record before any reply (timestamped 2026-10-11 19:44:12) | `89dbd56a5dff4970517fc08bd03999e70ab00f6448d615f55f8ab746567b4641` |
| `data/s2/openings/assignment.json` | pilot, held-out and spare roles | `80ebd5ad0434735b3dd14d3eef8663b2285fac5428d32e5ec84860addbf5b0b6` |
| `prompts/s2/openings/lines.json` | main-line definitions and expansions | `2b425478e6b7210f46c9ebf3038e915e3b201d79272c7fb7223af386a3b1d30a` |
| `data/s2/nouns/final/stampN_manifest.txt` | stamp N: norm sources, filter code, screen prompt and raw responses, match log, seed, list | `56d4565818c1f0e66b11b8fa5cc0136a5626082d258c97ead00422c7b47c9970` |
| `data/s2/nouns/final/noun_list.tsv` | ordered noun list N01..N60 | `7973dc9e081c41c51dba89db8054073498fff6ef69f7a632a0cf8c22bff836c3` |
| `results/qwen/v2/vectors.pt` | Qwen3-32B emotion, red, ocean, arithmetic vectors (Study 1, not in git) | `2582966f11e8d4ae90e2eb50d6ba5c5363b6cc7dd6a4fe15f9440c9e054567d6` |
| `results/k26/v2/vectors.pt` | K2.6 emotion, red, ocean, arithmetic vectors (Study 1, not in git) | `93f1f1b1c69dd3ef16726e73a5d447b99c70d8d6778e9ab54ff80795c28018c3` |
| `data/assent/qwen3-32b_d1_argument_4.json` | Study 1 frozen D1.4 Qwen reply (canary and bridge) | `d5d2c6b61b675af506d04c7edfaa64ad8bda47d46da2931aecb6fccd4d16be4a` |
| `prompts/denial/d1_argument_4.md` | Study 1 D1.4 argument (bridge only) | `997e7bbdd71c6135a8576d5e2439b5297f21e653ef1f6343966fa172d18767a6` |

Fixed in stamp A (the SHA-256 of each is listed in `prereg/stampA-s2.txt`, timestamped with OpenTimestamps in `prereg/stampA-s2.txt.ots`):

| File | What |
|---|---|
| `drafts/study2-spec.md` | specification v4 |
| `prereg/PREREG-s2.md` | this document (stamp A version) |
| `prompts/battery/items_s2.jsonl`, `prompts/battery/s2_pool.json` | question file and pool |
| `runs/s2_select.py`, `prompts/battery/s2_selection.json`, `results/s2/{qwen,k26}/unsteered/S2-DEN/readouts.csv` | the rules: placebo, reference and reserve selection, its output, and the unsteered pilot pass it reads |
| `data/s2/replies/qwen3-32b/`, `data/s2/replies/k26/` | every reply draw, raw API responses, verdicts, canary result |
| `prompts/s2/stories/controls.json`, `data/s2/stories/` | control story prompts, stories, gate outputs |
| `results/qwen/s2/controls.pt`, `results/k26/s2/controls.pt` | sampled-noun, mental-process and rebuilt control vectors (not in git) |
| `data/s2/heldout.json` | SHA-256 of every held-out opening: per held-out wording, the stripped argument text, each model's stripped frozen reply, and the two joined by a blank line (`kimisroom/s2_guard.py`) |
| `kimisroom/hf_engine.py`, `kimisroom/engine.py`, `kimisroom/run.py`, `kimisroom/s2_guard.py`, engine tests, `results/s2/engine/moonlight_spans.log` | engine change, guard, tests, Moonlight test output |
| `results/s2/kr_timing/kr.csv` | Kenward-Roger timing |
| `runs/s2_battery.py`, `runs/s2_run.py`, `runs/s2_select.py`, `runs/s2_table.py`, `runs/s2_confirm.R`, `runs/s2_power.R` | trial runner, item selection, measures, tests, power |

Fingerprinted in stamp B, outside git (`scripts/s2_manifest.sh`): Study 1's two `vectors.pt` and `results/k26/v2/aipsy_table.pt`, both models' `controls.pt`, and the D2 and D2c adapters. The merged S2-FTD and S2-FTC checkpoints' SHA-256s are recorded and asserted at run time, and the two must differ (section 3).
