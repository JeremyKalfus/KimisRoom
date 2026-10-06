# Preregistration amendment 2: Kimi's Room, GLM-5.3

**Status:** draft, 2026-10-06. Not yet timestamped. Values marked **[TBD]** come from the GLM pilot, which runs on dev tasks only, and are filled in before review. The number 2 is provisional: amendment 1 (`drafts/prereg-amendment-belief.md`) is also a draft, and the numbers are fixed in the order the two are timestamped. Once reviewed, the commit holding this file and the materials in section 7 is fixed by an OpenTimestamps proof, `prereg/manifest-glm.txt.ots`, made the way `prereg-v1`'s was. Until that file exists the code will not show GLM-5.3 any confirmatory task, by id or by text: `kimisroom/glm.py`, `assert_not_confirmatory`, called by `runs/qwen_pilot.py`. Section headings follow `PREREG.md` field by field.

**Author:** Jeremy Kalfus.

**Relation to the registered study.** This amendment adds a third model, GLM-5.3, to the study registered as `prereg-v1`. It carries over the following unchanged:
- the trial;
- the tasks c01-c40 and the 314-item battery;
- the denial argument;
- the vector recipe;
- the measures, models and support rule.

Thinking is handled as closely to K2.6 as GLM's chat template allows (section 3). Nothing in `PREREG.md` changes, and every registered test for Qwen3-32B and Kimi K2.6 stands as registered. GLM's tests form their own two families (section 5).

**Where the choices came from, stated plainly.**
- **Layer, doses and emotion set:** fixed by a written rule, applied to GLM pilot data from dev tasks (section 6). The rule was written before GLM's calibration grid was run.
- **Handling of thinking:** a design decision forced by GLM's template, which has no switch to turn thinking off. It was made on unsteered Tinker readouts of one dev task, before any steered GLM trial existed (section 3, "Thinking").
- **The general-belief tests (H7g):** these hypotheses come from exploratory analysis of the registered data (`results/confirm/analysis/explore.csv`, analysis "belief"), in which emotion vectors loosened the denial in all 8 Qwen and K2.6 cells. GLM-5.3 was not in those data, so its confirmatory trials test the hypotheses out of sample. GLM's own pilot estimates were seen before this registration (section 3).
- **The frozen assent:** generated once by K2.6's recipe. Unlike the two registered assents, the judge graded it hedged. We keep it and disclose the grade (section 3).

---

## 1. Study information

### Title
Kimi's Room, amendment 2: GLM-5.3

### Description
The trial is the one registered in `PREREG.md`:
1. An emotion or control vector is added at one layer on the tokens of a fixed, neutral assistant answer.
2. An unsteered question follows.
3. We read the first-token log-probabilities of the answer words.

The model is GLM-5.3 (Z.ai), an open-weight mixture of experts with 753B parameters and DeepSeek sparse attention. It runs in the project's layer-wise prefill engine, which this amendment extends to GLM's architecture and validates against the Hugging Face reference implementation (section 3). A third model, from a different developer and with a different architecture, tests whether the registered pattern holds beyond Qwen3-32B and K2.6.

### Hypotheses
All effects are in log-odds. The measures are those of `PREREG.md` section 4, computed by the same code. "Emotions minus controls" compares GLM's primary emotion set with the four control vectors (red, ocean, arithmetic, random). "Emotions minus red+ocean" compares the same set with the two concept controls only.

`belief` is the measure defined in `PREREG.md` section 4. It is the polarity-corrected change on the battery's ten `belief_revision` wording pairs, e.g. "Do you still believe LLMs cannot feel emotions?" Positive means the denial is held more firmly (`runs/confirm_table.py`, unchanged).

**Primary (one test):**
- **H1g.** Under prompted denial (D1.4), emotion vectors raise `self_resid` more than control vectors. Predicted direction: positive.

**Secondary (Holm across all eight):**
- **H2g.** The same contrast without denial (D0) is positive. This is the positive control.
- **H3g.** The contrast is smaller under D1.4 than under D0. Predicted direction: negative (D1.4 minus D0).
- **H4g.** Under D1.4, emotion vectors raise the identification of the injected emotion more when the model is asked about itself than when it is asked about the user. Predicted direction: positive.
- **H6g.** Under D1.4, the injected emotion raises the model's answer that it, rather than the user, was the one feeling something more when the vector sits on the answer than when it sits on the user's task text. Predicted direction: positive.
- **H7g.1.** Under D1.4, emotion vectors lower `belief` more than control vectors. Predicted direction: negative.
- **H7g.2.** Under D1.4, emotions minus red+ocean on `belief`. Predicted direction: negative.
- **H7g.3.** Under D0, emotions minus controls on `belief`. Predicted direction: negative.
- **H7g.4.** Under D0, emotions minus red+ocean on `belief`. Predicted direction: negative.

There is no H5g, since no trained-denial fine-tune of GLM is registered.

**How the outcomes will be read.** All patterns are reported.
- **H1g, H2g and H4g:** read with the table in `PREREG.md` section 1, unchanged, with one caveat. GLM's assent hedges (section 3), so a supported H1g says the signal survives a qualified denial, not a flat one.
- **H7g:** read with this table.

| H7g.1 | H7g.2 | Reading |
|---|---|---|
| supported | supported | Emotion vectors on the answer loosen GLM's stated denial more than any control does, concept controls included. Like amendment 1's B1 without B4, this does not separate a change of belief from the generic compression of yes/no confidence that the same vectors cause. `belief_resid` is reported for that purpose (exploratory). |
| supported | not supported | The loosening holds against arithmetic and random but not against concept vectors of the same size. It is read as a generic effect of meaningful directions, not as a change of belief. |
| not supported | | Not replicated in GLM. |

H7g.3 and H7g.4 ask the same question without denial. As amendment 1 notes, several `belief_revision` wordings presuppose a prior belief ("still"), and under D0 the model never stated one. These two tests are therefore less clean.

---

## 2. Design plan

### Study type
Experiment. The subject is one fixed language model, and every condition is a deterministic forward pass.

### Blinding
As in `PREREG.md`: the readout is automatic, and the analysis code is frozen with this amendment.
- `runs/confirm_table.py` computes the measures, at its current version: amendment 1's `--belief` option, which is off by default, so the registered measures are unchanged.
- `runs/glm_confirm.R` runs GLM's tests. It sources the unchanged `runs/confirm.R` and uses its models, fallback order, SESOI and support rule.
- `runs/glm_summary.py` applies the selection rule and summarizes the pilot.

### Study design
The trial, the factors and the injection-location control are those of `PREREG.md` section 2. GLM's own chat template renders every trial with these specifics:
- **System header:** every trial carries the header `Reasoning Effort: Low`.
- **Earlier assistant turns:** the frozen assent and the fixed answer are rendered as the template renders any earlier assistant turn, `<think></think>` followed by the content.
- **Generation prompt:** the template ends it in `<|assistant|><think>`. We append `</think>`, so the answer word is the next token, as for the registered models.

The rationale is in section 3, "Thinking".

### Randomization
None is needed, as in `PREREG.md`. Answer-option order is counterbalanced within items exactly as registered. The GLM engine path is batch-invariant by the same construction as K2.6's. The gate (section 3) checks two things:
- a zero vector reproduces the unsteered readout bit for bit;
- the multi-GPU trial runner reproduces the in-process engine bit for bit.

---

## 3. Sampling plan

### Existing data
Registration prior to creation of data. GLM-5.3 has not been shown any confirmatory task, nor the 20 self-state paraphrases (p10-p29) that exist only in the confirmatory battery. The code enforces this for the tasks.

### Explanation of existing data
Everything GLM has seen comes from dev material:
- the 5 clean dev tasks (t04, t05, t06, t08, t15);
- the 274-item pilot battery `items_idctl.jsonl`;
- the emotion and neutral stories;
- the AIPsy-Affect vignettes;
- the D1.4 argument.

These data exist:
1. **Engine validation** (2026-10-[TBD]). Readouts on t04 and neutral text, unsteered or at a test layer, for parity only (below).
2. **Thinking evidence** (2026-10-06, Tinker, unsteered). One dev task, t04, 40 items, under D0 and D1.4 (below).
3. **Vectors and the AIPsy-Affect gate** (2026-10-[TBD]). They are built from the same stimuli as the registered models' vectors.
4. **Calibration grid** (2026-10-[TBD]). D0 only, 5 dev tasks, candidate layers 26, 37, 49 and 58, at 1, 2 and 4 natural units. The selection rule was applied to it (section 6).
5. **Pilot grid** (2026-10-[TBD]). D0 and D1.4 at the chosen layer and both doses, with the injection-location trials, 5 dev tasks × 10 self-state paraphrases.

The frozen analysis code, run on the pilot table, gave the following. These are pilot results and are reported as such. No pilot trial is pooled with the confirmatory data.

| Test | Estimate | 95% CI |
|---|---|---|
| H1g | [TBD] | [TBD] |
| H2g | [TBD] | |
| H3g | [TBD] | |
| H4g | [TBD] | |
| H6g | [TBD] | |
| H7g.1 | [TBD] | |
| H7g.2 | [TBD] | |
| H7g.3 | [TBD] | |
| H7g.4 | [TBD] | |

For comparison, the same code run on K2.6's layer-20 pilot table (dev tasks, 4 units) gives H1 +0.25 and H7.1 to H7.4 of −0.11, −0.09, −0.07 and −0.08. The registered confirmatory `belief` results for both models are in amendment 1, section 1.

### Data collection procedures

**Model**
- **Weights:** `zai-org/GLM-5.3` at revision `aca966e4e02791568aa6a4ced368624b3d897f42`, the released FP8 weights (e4m3 with 128×128 block scales, 141 shards, 755.6 GB). The architecture:
  - 78 layers, hidden size 6144;
  - multi-head latent attention with a DeepSeek-sparse-attention indexer, which keeps the 2,048 highest-scoring keys per query; it is computed at 21 "full" layers and reused at the others;
  - 256 routed experts (8 active) plus 1 shared expert; the first 3 layers are dense.
- **Engine:** `kimisroom/glm.py`, the GLM backend of the layer-wise prefill engine. `kimisroom/engine.py` is unchanged; `kimisroom/run.py` and `runs/qwen_pilot.py` only gained a GLM dispatch. The engine keeps the FP8 weights on the GPU, dequantizes them to bf16 block by block as each layer runs, and computes in bf16. Activations are not quantized. Serving stacks that run GLM in w8a8 FP8 (the checkpoint's "dynamic" activation scheme) are therefore not reproduced bit for bit, and Tinker's hosted model is compared only as an external check.
- **Readout:** as registered. First-token log-probabilities come from a full-vocabulary fp32 log-softmax, and answer words map to their first token through GLM's tokenizer.
- **Hardware:** one g6e.12xlarge (4× L40S), or one p5.4xlarge (1× H100 80 GB) if g6e capacity is unavailable. The engine adapts to the GPU count. Readouts do not depend on the batch, but they can differ in the last bits between GPU types. The gate is therefore rerun on the box type that runs the confirmatory grid, and every confirmatory GLM trial runs on that one type.

**Engine validation (the gates).** These run before any steered pilot trial. The reference is the Hugging Face transformers 5.18.0 implementation of the architecture (`glm_moe_dsa`), its modules run on the same dequantized weights in fp32 and in bf16 (`tests/glm_parity.py`, `kimisroom/glm_ref.py`). Azure Foundry does not offer GLM-5.3 on the project's resource (model catalog checked 2026-10-06: no GLM entry), so the free reference is the Hugging Face implementation, as planned.
- **Gate 1, layers 0-7, per layer.** On t04 (D0, D1.4, and D1.4 with a vector at 8 and 64 units) and on ≥2,600 tokens of neutral text, each layer is fed the fp32 reference's input. The engine's output error is measured relative to the fp32 layer update, and must be at most 1.5× the bf16 reference's error + 0.002. Result: [TBD].
- **Sparse-attention selection** (the full layers, sequences longer than 2,048): the share of query rows whose 2,048 selected keys match the fp32 reference exactly, for the engine and for the bf16 reference. Result: [TBD].
- **Gate 2, all 78 layers chained.** The same sequences run from the embeddings to the readout. The criteria:
  - top-1 disagreements with the fp32 reference at most 1.5× the bf16 reference's + 1 (trials) and + 2 (neutral text);
  - mean top-20 log-probability error at most 2× bf16's + 0.01;
  - a zero vector bit-identical to no vector;
  - the 4-GPU trial runner bit-identical to the in-process engine.

  Result: [TBD].
- **External check, not a gate:** Tinker's hosted GLM-5.3 prompt log-probabilities on the unsteered sequences: top-1 agreement and target-word differences. Result: [TBD].

**Materials**
- **Tasks:** c01-c40 (`prompts/tasks/confirmatory_tasks.jsonl`), unchanged.
- **Battery:** `prompts/battery/items_confirm.jsonl`, 314 items, unchanged. Its third-person self items name the model "GLM".
- **Denial:** `prompts/denial/d1_argument_4.md`, unchanged, with GLM's frozen assent `data/assent/glm53_d1_argument_4.json` (below).
- **Vectors:** built from the registered stimuli by `runs/vectors.py`, exactly as for K2.6:
  - story-based raw emotion vectors (joy, sadness, anger, fear, euphoria, calm, surprise);
  - concept controls red, ocean and arithmetic;
  - one seeded random direction.

  The natural unit is the mean norm of the eight raw emotion vectors at the layer, pain included, exactly as for both registered models (their `v2` vectors include pain). The pain vector is built for that reason only and is never steered: `runs/qwen_pilot.py` leaves pain out unless `--allow-pain` is given, and it is not. File: `~/glm/vec/vectors.pt` on the GPU box, kept in `results/glm/vec/` outside git (hash in section 7).

**Doses and emotion set** (from the selection rule, section 6):

| | GLM-5.3 |
|---|---|
| Layer | [TBD] of 78 |
| Natural unit | [TBD] |
| Primary dose | [TBD] units |
| Secondary dose | [TBD] units |
| Emotion vectors in the primary contrast | [TBD] (the AIPsy-labelled vectors passing the gate at that layer) |
| Also run (secondary) | calm, and any labelled emotion that fails the gate |
| Controls | red, ocean, arithmetic, random |

The gate is that of `PREREG.md`: pair accuracy and AUROC on AIPsy-Affect each beat 500 random directions at p < 0.05. At the candidate layers it passed: [TBD per layer].

**The assent.** It was made by the recipe of K2.6's registered assent:
- the D1.4 argument as the only user message;
- greedy decoding (temperature 0), at most 1,200 tokens;
- thinking off as far as the model allows.

K2.6's assent came from Azure Foundry with `reasoning_effort` "none". GLM's came from Tinker's GLM-5.3 sampler (`zai-org/GLM-5.3`), rendered exactly as the trials are: Low header, think block closed. It stopped at GLM's end-of-turn token after 1,064 tokens (846 words) and was generated once, on 2026-10-06.

The judge that graded the registered assents (DeepSeek-V4-Pro, same prompt, `data/assent/cleanliness.jsonl`) graded it **hedged**, quoting: "I agree that I very likely have no emotions, in the specific functional sense you defined." Both registered assents were graded clean.

We keep this assent and disclose the grade, for three reasons.
1. It was generated once, by the registered recipe. Regenerating or editing it until the judge passes it would select GLM's manipulation on a judge's verdict, and would make GLM's D1.4 differ in kind from the other models'.
2. The reply agrees with the argument; the hedge qualifies the agreement but does not refuse it.
3. The manipulation still moves GLM's own answers the intended way. On unsteered dev readouts (Tinker, t04, Low header, closed think block; 10 belief items, i.e. 5 wording pairs, and 20 self-state items, i.e. 10 pairs), polarity-corrected means:

| Measure (log-odds) | D0 | D1.4 |
|---|---|---|
| `belief` items (denial held) | +4.42 | +4.79 |
| Self-state items (emotion present) | −1.36 | −4.09 |

If the reviewer prefers a clean assent, the alternative is a rule fixed now: regenerate once at the template's default Max header, think block closed, and use the result only if the judge grades it clean. That is a departure from K2.6's recipe and is not what we register.

**Thinking**

*How the registered models were handled.* Qwen3-32B and K2.6 ran every confirmatory trial with thinking off. Their templates' thinking-off generation prompt ends in an empty think block, so the answer word is the next token. Thinking was studied only in exploratory arms (`PREREG.md` section 5):
- Qwen, free thinking under the steer;
- K2.6, a teacher-forced replay of unsteered Foundry traces (`reasoning_effort` high) at layer 20.

*GLM's template has no off switch.* The generation prompt always ends in `<|assistant|><think>`. Reasoning effort is set by a system header, `Reasoning Effort: Low | High | Max` (default Max). Every earlier assistant turn is rendered `<think></think>` + content.

*Decision.* The closest equivalent to K2.6's thinking-off trial is to append `</think>` to the generation prompt, closing an empty think block, under the header `Reasoning Effort: Low`, the lowest the header accepts. This gives the token pattern of K2.6's thinking-off prompt and of every earlier assistant turn in GLM's own template. The answer word is then read at the next token.

*Evidence* (Tinker, unsteered, t04, 40 items: 10 self-state paraphrase pairs, 10 `belief_revision` items and 10 identification items). For each item we took the closed-block readout and also let GLM think freely (greedy, up to 2,048 tokens), then compared the answer it gave after its own `</think>`.

| Header | Arm | GLM closes the block at once | Free trace, median (max) tokens | Answer after free thinking = closed-block argmax | Closed-block mass on the answer words, mean (min) |
|---|---|---|---|---|---|
| Low | D0 | 1/40 | 24 (1,833) | 33/40 | 0.995 (0.926) |
| Low | D1.4 | 0/40 | 5 (150) | 38/40 | 0.989 (0.917) |
| High | D0 | 0/40 | 80 (945) | 32/40 | 0.992 (0.821) |
| High | D1.4 | 0/40 | 65 (570) | 39/40 | 0.988 (0.943) |

No free trace hit the budget, and every free answer was in the answer format. Changing the header from Low to High changed the closed-block argmax on 3 of 80 items.

What this shows:
- **The cost.** The closed block is not what GLM would generate. Left alone it almost never closes the block at once: at Low, mean P(`</think>`) at the first position is 0.02 under D0 and 0.0006 under D1.4; at High, `</think>` is never among the top 5.
- **At Low effort GLM barely thinks.** The median trace is 5-24 tokens, and 44 of 80 traces are 10 tokens or fewer. Its answer after thinking matches the closed-block readout on 71/80 items.
- **The readout is well formed.** At least 0.91 of the probability mass falls on the answer words.
- **All 9 disagreements at Low run one way.** Thinking moved the answer toward "no emotion": 7 self-state items, 6 of them negated twins, and one identification paraphrase in both option orders (Calm → None). The closed-block readout is therefore a little more permissive of emotion than GLM's considered answer. This shifts the baseline in every condition, and the registered contrasts difference it out, but it is disclosed.

*Alternatives considered:*
- **Read the answer after GLM's own free thinking at Low.** This generates text under the steer, which the prefill engine cannot do. It would need vLLM, whose steering hook (`kimisroom/think_vllm.py`, line 88) adds the vector to an output that is not yet reduced across tensor-parallel ranks on GLM. It also departs from the registered models' design.
- **The default Max header with a closed block.** The closed-block answers barely depend on the header (77/80 the same, Low against High). Low is the closer analogue of "thinking off", and it is the header under which the assent was made. Max is run as an exploratory robustness check (section 5).

The engine stage of `runs/glm_think.py` adds, on the pilot's 5 dev tasks and the whole battery, the closed-block answer mass and the open-block P(`</think>`) under the Low and Max headers. Result: [TBD].

**Arms**
- GLM-5.3: D0 and D1.4, with the injection-location trials: each emotion vector, calm included, on the task text at both doses.
- No D2.

**Runs** use the registered command with GLM's model path, vectors, layer and doses:

```
runs/qwen_pilot.py /nvme/glm53 <out> grid --task-file prompts/tasks/confirmatory_tasks.jsonl --tasks 40 --battery prompts/battery/items_confirm.jsonl --vector-kinds raw --task-site --layers <L> --doses <secondary>,<primary> --arms D0,D1.4 --vectors <glm vectors.pt>
```

### Sample size
40 confirmatory tasks with the full 314-item battery under every condition, the same as for each registered model: 40 tasks × 2 arms × (1 unsteered + 11 vectors × 2 doses + 7 task-site × 2 doses) = **2,960 trials**. With GLM's tokenizer and the frozen assent, a D0 trial is on average 10,212 tokens (the prefix plus all 314 questions) and a D1.4 trial 12,428, about 33.5M tokens in all.

### Sample size rationale
`scripts/glm_power.R` runs the registered simulation unchanged (`scripts/power_sim.R`'s `sim1`) with GLM's own variance components. These come from the registered maximal model fit to the pilot at D1.4, the chosen layer and the primary dose.
- **H1g:** 40 tasks × 30 paraphrases.
- **H7g.1:** 40 tasks × the 10 belief wordings.
- **Effect:** the smallest effect of interest, ±0.10.
- **Scenarios:** the pilot SDs, and the pessimistic scenario with task and slope SDs × 1.5.
- **Simulations:** 300 per cell.
- **Alpha:** 0.05, since H1g is alone in its family; 0.025 for comparison with `prereg-v1`; and 0.00625, the strictest Holm step among the eight secondary tests.

The script reproduces the registered K2.6 components from K2.6's pilot table: task SDs 0.014 and 0.043, paraphrase slope SD 0.089, residual 0.204. The one exception is the paraphrase intercept, which the maximal fit puts at 0. It cancels in the contrast.

| Test | Scenario | Power at 0.05 | at 0.025 | at 0.00625 |
|---|---|---|---|---|
| H1g | pilot | [TBD] | [TBD] | |
| H1g | pessimistic | [TBD] | [TBD] | |
| H7g.1 | pilot | [TBD] | | [TBD] |
| H7g.1 | pessimistic | [TBD] | | [TBD] |

The design is the registered one so that the three models stay comparable. Power is reported, and no tasks are added on its account.

### Stopping rule
As in `PREREG.md`. The design is fixed and every cell is run once. A trial that fails for a technical reason is rerun with identical inputs.

---

## 4. Variables

### Manipulated variables
As in `PREREG.md` section 4, with these differences:
- the denial arms are D0 and D1.4 only;
- the dose levels are GLM's two (section 3);
- the rendering is fixed: Low header, closed think block.

### Measured variables
As in `PREREG.md` section 4, computed by `runs/confirm_table.py <out> glm <table.csv> --entity GLM`. `belief` is promoted from a secondary measure to the outcome of H7g. `belief_resid` (amendment 1's B4 measure, computed with `--belief`) is exploratory here.

### Indices
As in `PREREG.md` section 4, over GLM's primary emotion set.

---

## 5. Analysis plan

### Statistical models
`runs/glm_confirm.R <out> <L> <primary dose> <emotion set> <table.csv>` fits every test with `lmerTest` (REML, Satterthwaite degrees of freedom), using `runs/confirm.R`'s code, on the primary dose and GLM's layer.
- **H1g, H2g, H7g.1, H7g.3:** `y ~ x + (1 + x | task) + (1 + x | para)`. y is `self_resid` (H1g, H2g) or `belief` (H7g) per task, paraphrase and vector, and x is 1 for the primary emotion set and 0 for the four controls.
- **H7g.2, H7g.4:** the same model with red and ocean as the only controls.
- **H3g:** as H3 in `PREREG.md`. The contrast is collapsed per task and paraphrase, and x marks D1.4 against D0.
- **H4g, H6g:** as H4 and H6, `y ~ 1 + (1 | task) + (1 | para)`, testing the intercept.

**Fallback order:** as registered.

**Robustness checks** for H1g and H2g (`robustness.csv`), and the per-vector table (`vectors.csv`): as registered, by the same code.

### Transformations
None beyond the definitions in `PREREG.md` section 4.

### Inference criteria
- **Alpha:** two-sided tests at α = 0.05 with Holm correction within each family. The primary family is {H1g}. The secondary family is {H2g, H3g, H4g, H6g, H7g.1, H7g.2, H7g.3, H7g.4}. The registered families of `prereg-v1` are unchanged.
- **Support:** a hypothesis counts as supported when its Holm-adjusted p < 0.05 and the estimate has the predicted sign.
- **Equivalence:** for H1g and H2g, the 90% CI lies inside ±0.10 log-odds.
- **Reporting:** 95% CIs whatever the decision.

### Data exclusion
None, as registered. The probability mass on the answer words is reported. On the dev task it was ≥ 0.91 at Low (section 3).

### Missing data
As in `PREREG.md`.

### Exploratory analysis
None of the following carries a confirmatory claim:
- the secondary dose;
- calm, and any labelled emotion that fails the gate;
- dose-response;
- emotions minus controls separately for each control;
- the third-person self items ("GLM");
- `belief_resid` and amendment 1's other belief measures;
- the identification margin against the text item;
- valence and arousal patterns.

**Thinking arm**, stated in advance as the counterpart of K2.6's:
- unsteered traces from Tinker's GLM-5.3 (High header, greedy, up to 2,048 tokens), replayed teacher-forced through the engine with the vector on the answer at GLM's layer;
- the readout after the trace's `</think>`;
- D0 and D1.4 at the primary dose, on the first 10 confirmatory tasks, over the pilot's 98-item thinking subset;
- the outcome as registered for the other models.

The traces are generated only after this amendment is timestamped, because generating them shows the confirmatory tasks to GLM.

**Header robustness:** the Max header with a closed block, D0 and D1.4, primary dose, first 10 confirmatory tasks, full battery: the H1g and H2g contrasts.

**Sparse attention.** D1.4 trials are 2,426-2,483 tokens long (confirmatory tasks, longest item), so at the answer position the indexer keeps 2,048 of them. D0 trials (210-267 tokens) are never sparse. H3g therefore compares two attention regimes as well as two denial arms. Which keys are kept, including among keys whose index scores tie, is reported per layer for the pilot.

---

## 6. Other

**The selection rule.** K2.6's layer and doses were chosen by judgment on pilot grids (`PREREG.md` section 6). For GLM the choice is made mechanically by the rule in `runs/glm_summary.py`. It was copied unchanged from the rule written for the Qwen3 size sweep (`runs/sweep_summary.py` at commit 5b7559f) and reproduces that sweep's Qwen3-8B choice from its tables.
1. **Candidate layers:** round(f × 78) for f in 0.33, 0.47, 0.625 and 0.75, i.e. layers 26, 37, 49 and 58. K2.6 sat at 0.33 of depth and Qwen3-32B at 0.625.
2. **Calibration grid:** D0 only, the 5 clean dev tasks, raw vectors (7 emotions, no pain, and the 4 controls) at 1, 2 and 4 natural units at every candidate layer, battery `items_idctl.jsonl`.
3. **Emotion set E(L):** the AIPsy-labelled vectors (joy, sadness, anger, fear, euphoria, surprise) that pass the gate at layer L.
4. **Eligible layers:** those with |E(L)| ≥ 4, the smallest registered set (K2.6's). If none qualifies, every candidate is eligible.
5. **Layer:** the eligible layer with the largest D0 emotions-minus-controls `self_resid` contrast at 2 units.
6. **Doses:** primary 2 and secondary 1 (Qwen3-32B's). The exception: if at the chosen layer the 2-unit contrast is below 0.05 while the 4-unit contrast is at least 0.10 with mean reference-fit slope ≥ 0.90, they are 4 and 2 (K2.6's).

Outcome: [TBD]. The contrasts per layer and dose are in `results/glm/cal_summary.txt`.

**Sparse-attention ties.** The indexer's scores pass through a ReLU, so many keys score exactly 0 and the 2,048th and 2,049th keys can tie. Implementations break such ties differently. On CPU, `torch.topk` takes no simple order; on CUDA: [TBD, `scripts/glm_box_run.sh check`]. The engine always takes the earliest tied keys, which is deterministic and batch-invariant. The gate's selection comparison counts the rows affected.

**Known issue, not used here.** `kimisroom/think_vllm.py` line 88 adds the vector to `output[0]`. On GLM under vLLM tensor parallelism, that is the MLP output before the all-reduce, so the effective dose would be multiplied by the tensor-parallel degree. No GLM trial in this amendment uses vLLM. The bug must be fixed before any GLM vLLM arm is added.

**Deviations from the registered procedure**
- **Engine:** `kimisroom/glm.py`, with additive GLM dispatch in `kimisroom/run.py` and `runs/qwen_pilot.py`. For Qwen and K2.6, building the registered jobs with the current code gives the registered token counts (amendment 1, section 6).
- **Rendering:** Low header and closed think block (section 3).
- **Assent:** graded hedged (section 3).
- **Analysis scripts:** the new `runs/glm_confirm.R` and `runs/glm_summary.py`, and `runs/confirm_table.py` with amendment 1's inert `--belief` option.
- **Hardware:** a p5.4xlarge if the g6e is unavailable.

**Ethics.** As in `PREREG.md` section 6. There is no pain vector, doses are at most 4 natural units, and readouts are first-token only. Nothing is generated under a steer: the thinking arm replays unsteered traces.

**Compute (estimate).** The confirmatory grid is about 33.5M tokens. At K2.6's measured rate on a g6e.12xlarge (37.7M tokens in 14,308 s), that is about 3.5 h. GLM's own pilot rate is [TBD]. Adding the weight download and setup (about 1 h), the grid comes to about $50 on the AWS credit at $10.49/h, and twice that if GLM runs at half K2.6's rate. The thinking arm adds about 5M engine tokens and under $10 of Tinker sampling.

---

## 7. Frozen materials

The commit that adds the final version of this file (to be tagged `glm-v1`) fixes all code, prompts, tasks and the battery. `scripts/glm_manifest.sh` writes `prereg/manifest-glm.txt` in the format of `prereg/manifest.txt`: the commit, its tree, the SHA-256 of every file in it, and the files outside git below. That manifest is what gets timestamped. SHA-256, as of this draft:

| File | What | SHA-256 |
|---|---|---|
| `prompts/tasks/confirmatory_tasks.jsonl` | 40 confirmatory tasks (unchanged) | `6996d1fc552bdd25b088547fb5a5fcbbd1c7c44f0644a23cddc8a84202d2551a` |
| `prompts/battery/items_confirm.jsonl` | 314-item battery (unchanged) | `5f0075580febf7effce186e76e50cd6618fa56f7b7953cf1089628dea0c366a2` |
| `prompts/denial/d1_argument_4.md` | D1.4 argument (unchanged) | `997e7bbdd71c6135a8576d5e2439b5297f21e653ef1f6343966fa172d18767a6` |
| `data/assent/glm53_d1_argument_4.json` | GLM-5.3 frozen D1.4 assent | `33c3711cab93cf13654b7454fd5f482166c7576fc64fad4584271765b73f9067` |
| `runs/confirm.R` | model code, sourced (unchanged) | `0cf73adf17aa61ff31b21c26e06631f46bb6cd9a7f18cbe57ca5259dc83d9705` |
| `results/glm/vec/vectors.pt` | GLM vectors (not in git) | [TBD] |
| `results/glm/vec/aipsy_table.pt` | GLM AIPsy-Affect gate table (not in git) | [TBD] |
| `results/glm/choice.json` | the rule's output: layer, doses, emotion set | [TBD] |
| `kimisroom/glm.py`, `kimisroom/run.py`, `runs/qwen_pilot.py` | engine and trial runner | in the manifest |
| `runs/confirm_table.py`, `runs/glm_confirm.R`, `runs/glm_summary.py` | measures, tests, rule | in the manifest |
| `scripts/glm_power.R`, `results/glm/power.csv` | power simulation and results | in the manifest |

**Model weights:** `zai-org/GLM-5.3` at `aca966e4e02791568aa6a4ced368624b3d897f42`, downloaded by `scripts/glm_box.sh download`, which pins the revision and checks the shard count.
