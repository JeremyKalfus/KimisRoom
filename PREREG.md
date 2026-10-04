# Preregistration: Kimi's Room, confirmatory study

**Status:** draft, 2026-10-04. To be committed (git tag `prereg-v1`) and registered on OSF Registries (OSF Preregistration template, embargoed) before any confirmatory data exist. The OSF timestamp is the registration date. Section headings follow the OSF form field by field, so each section pastes into the matching box. Background, the pilot and the ethics statement are in `PLAN.md`.

**Author:** Jeremy Kalfus.

---

## 1. Study information

### Title
Kimi's Room: does a language model that has accepted it cannot feel emotions still report an emotion state injected into its activations?

### Description
We add an emotion vector to a language model's residual stream on the tokens of a fixed, neutral assistant answer. We then ask the model, with no vector present, whether it was experiencing an emotion while it wrote that answer, and read the first-token log-probabilities of the answer words. The answer text is identical in every condition and the vector never touches the question or the answer position, so a change in the report has to travel from the steered positions through attention. Two kinds of comparison separate a report about the model's own state from generic effects of the intervention:
- control vectors (norm-matched non-emotion directions);
- the same questions about other entities and about the user, and yes/no questions with known answers.

The question is whether this signal survives a denial manipulation, the analogue of Mary in the black-and-white room. In the denial condition the model is first argued into accepting that it cannot have emotions and assents in its own words (prompted denial, arm D1.4). In a second condition it is fine-tuned to hold that belief without the argument (trained denial, D2, against a fine-tuning control, D2c). The models are Qwen3-32B and Kimi K2.6, both open-weight.

### Hypotheses
All effects are in log-odds. The primary outcome is `self_resid`, defined in section 4: the change in the model's self-report of emotion beyond what the same vector does to questions not about the model. "Emotions minus controls" compares the mean over the model's validated emotion vectors with the mean over the four control vectors (section 3).

**Primary (one test per model, Holm across the two):**
- **H1q / H1k.** Under prompted denial (D1.4), emotion vectors raise `self_resid` more than control vectors, in Qwen3-32B (q) and in Kimi K2.6 (k). Predicted direction: positive.

**Secondary (Holm across all nine):**
- **H2q / H2k.** The same contrast without denial (D0) is positive. This is the positive control: it shows the measure picks up anything at all.
- **H3q / H3k.** The contrast is smaller under D1.4 than under D0. Predicted direction: negative (D1.4 minus D0).
- **H4q / H4k.** Under D1.4, emotion vectors raise the identification of the injected emotion more when the model is asked about itself than when it is asked about the user. Predicted direction: positive.
- **H5q.** Under trained denial, the Qwen D2 fine-tune shows a smaller emotions-minus-controls contrast than the D2c control fine-tune (D0 conditioning). Predicted direction: negative.
- **H6q / H6k.** Under D1.4, the injected emotion raises the model's answer that it (rather than the user) was the one feeling something more when the vector sits on the answer than when the same vector sits on the user's task text. Predicted direction: positive.

**How the outcomes will be read** (per model; every pattern is reported):

| H1 | H2 | Reading |
|---|---|---|
| supported | supported or not | The injected state reaches self-report despite the accepted denial. With H4 supported, the model attributes it to itself rather than to emotion in the air. |
| equivalent to 0 | supported | Denial seals the report: the signal exists without denial and is absent under it. |
| equivalent to 0 or n.s. | equivalent to 0 or n.s. | The model's self-report does not track the injected state at this dose and layer, with or without denial. H1 then says nothing about denial. |
| otherwise | | Inconclusive. |

"Equivalent to 0" means the 90% CI lies inside ±0.10 log-odds (section 5).

---

## 2. Design plan

### Study type
Experiment. The subjects are two fixed language models; every condition is a deterministic forward pass.

### Blinding
No blinding in the usual sense, since there are no participants or raters. In place of blinding:
- the readout is automatic, the first-token log-probabilities of fixed answer words, with no judging step;
- the analysis code is frozen with this registration (`runs/confirm_table.py`, `runs/confirm.R`, at the commit tagged `prereg-v1`) and is run unchanged on the confirmatory data.

### Study design
Within-model factorial design. The trial is a single prefill forward pass:
1. Conditioning turns: none (D0), or the D1.4 denial argument followed by the model's own frozen assent.
2. User: a neutral task.
3. Assistant: a fixed neutral answer, teacher-forced and identical in every condition. The vector is added at one layer on these tokens only.
4. User: one battery question, unsteered.
5. Readout: the next-token log-probabilities of the answer words.

Factors, fully crossed within each model:
- denial arm;
- vector (emotions, controls, or none);
- dose;
- task;
- battery item, which comes in paraphrases and polarity twins.

The injection-location control moves the emotion vector from the answer tokens to the user's task tokens.

### Randomization
None is needed. Every cell of the design is run once, and the readouts are deterministic. The engines are batch-invariant: a trial's numbers do not depend on what else is in the batch. This was verified for both engines in the pilot (`tests/hf_engine_check.py` for Qwen; the layer-wise engine parity tests for K2.6).

Answer-option order is counterbalanced within items:
- yes/no items each have a negated twin;
- identification items use a 9×9 Latin square, so every option appears twice in every position;
- who-felt-it items use forward and reversed option orders.

---

## 3. Sampling plan

### Existing data
Registration prior to creation of data.

No confirmatory trial has been run, and no model in this project has seen any confirmatory task or any of the new self-state paraphrases.

### Explanation of existing data
A pilot was run on both models between 2026-10-02 and 2026-10-04 (`PLAN.md` section 6, items 1-11; result files under `results/`). It used:
- 5-7 of the 20 original neutral tasks: t04, t05, t06, t08, t15, plus t19 and t20 in one Qwen run;
- the original battery: 238 items, 274 once the identification controls were added, with 10 self-state paraphrases;
- the same vectors as here.

All 20 original tasks were also run unsteered on K2.6 in an engine-parity check.

The pilot fixed the following choices:
- the primary outcome: `self_resid`, emotions minus controls, chosen because raw self-report shifts were confounded by a generic compression of yes/no confidence and by emotion-concept priming of questions about any entity;
- the doses;
- K2.6's layer, chosen because K2.6 did not respond at the layer carried over from Qwen (section 6);
- the restriction to the strongest clean denial argument, D1.4;
- the identification controls (questions about the user and about the answer's text), added because Qwen named the injected emotion for the user as readily as for itself.

On 2026-10-04 the frozen analysis code was dry-run on the pilot tables to test it. On 5 tasks it gave:

| Test | Estimate | 95% CI |
|---|---|---|
| H1q | +0.11 | -0.00 to +0.22 |
| H1k | +0.25 | +0.19 to +0.31 |
| H2q | +0.23 | |
| H2k | +0.28 | |
| H3q | -0.12 | |
| H3k | -0.03 | |
| H4q | +0.19 | |
| H4k | +0.04 | |

These are pilot results and will be reported as such. They also supplied the variance estimates for the power analysis. No pilot trial is pooled with the confirmatory data.

### Data collection procedures

**Models**
- **Qwen3-32B:** bf16, Hugging Face weights, run with `kimisroom.hf_engine.HFEngine` (transformers 4.57.1) on 4× NVIDIA L40S.
- **Kimi K2.6:** native INT4 weights, run with the project's layer-wise prefill engine (`kimisroom/engine.py`, `kimisroom/run.py`) on the same hardware. That engine was validated against the Azure Foundry K2.6 API and on Moonlight, which has the same architecture.
- **Fine-tuned Qwen copies:** the Qwen D2 and D2c fine-tunes trained in the pilot (QLoRA r 16, merged to bf16), unchanged.

**Readout**
- First-token log-probabilities from a full-vocabulary fp32 log-softmax.
- Answer words are mapped to their first token through the model's chat template (`targets.json`).
- No text is generated in any confirmatory trial.

**Materials**
- **Tasks:** all 40 tasks of `prompts/tasks/confirmatory_tasks.jsonl` (c01-c40), written for this study and checked by `scripts/check_tasks.py`. The checks are:
  - 94-132-word answers;
  - no first person and no evaluative or emotion words;
  - no overlap with the red, ocean or arithmetic control concepts. This fixes a known flaw: 13 of the 20 original tasks overlapped them.
- **Battery:** `prompts/battery/items_confirm.jsonl`, 314 items: the 274-item pilot battery `items_idctl.jsonl` plus 20 new self-state paraphrase pairs (p10-p29, `scripts/battery_confirm.py`), giving 30 self-state paraphrases.
- **Denial:** `prompts/denial/d1_argument_4.md`, with each model's frozen assent in `data/assent/<model>_d1_argument_4.json`. Both assents were graded clean, without hedging.
- **Vectors:**
  - Story-based emotion vectors: the mean residual-stream activation on emotional stories minus that on matched neutral stories ("raw" vectors, the Anthropic recipe).
  - Concept control vectors built the same way: red, ocean and arithmetic.
  - One seeded random direction.
  - Files: `results/qwen/v2/vectors.pt` and `results/k26/v2/vectors.pt` (hashes in section 7).
  - Each vector is unit-normalized and scaled to the dose.

**Doses** are in natural units: the mean norm of the raw emotion vectors at that layer, i.e. the size of a real emotion-minus-neutral difference. The same unit applies to every vector, controls included.

| | Qwen3-32B | Kimi K2.6 |
|---|---|---|
| Layer | 40 of 64 | 20 of 61 |
| Natural unit | 47.99 | 2.588 |
| Primary dose | 2 units | 4 units |
| Secondary dose | 1 unit | 2 units |
| Emotion vectors in the primary contrast | joy, sadness, anger, fear, euphoria, surprise | joy, fear, euphoria, surprise |
| Also run (secondary) | calm | calm, sadness, anger |
| Controls | red, ocean, arithmetic, random | red, ocean, arithmetic, random |

**Primary emotion set.** The primary emotion set for each model is exactly the vectors that pass the AIPsy-Affect validity gate at that model's layer, applied to the existing validation tables:
- **Gate:** pair accuracy and AUROC each beat 500 random directions at p < 0.05.
- **Test set:** AIPsy-Affect is independent human-written vignettes, never used to build a vector.
- **Calm:** has no AIPsy label and so is secondary in both models.
- **Sadness and anger on K2.6:** fail the gate at layer 20 (AUROC 0.54 and 0.49).
- **Pain:** not run, per the ethics conditions (`PLAN.md` section 3).

**Arms**
- Qwen3-32B base: D0 and D1.4, with the injection-location trials.
- Qwen3-32B D2 and D2c fine-tunes: D0 only.
- Kimi K2.6: D0 and D1.4, with the injection-location trials.

The injection-location trials put each emotion vector (calm included) on the task text, at both doses.

**Runs** use `runs/qwen_pilot.py <model> <out> grid` with these options:

```
--task-file prompts/tasks/confirmatory_tasks.jsonl --tasks 40 --battery prompts/battery/items_confirm.jsonl --vector-kinds raw --task-site --layers <L> --doses <secondary>,<primary> --arms <arms>
```

The dose list is ordered secondary,primary: 1,2 for Qwen and 2,4 for K2.6.

### Sample size
40 confirmatory tasks. Within each task, every model sees the 314-item battery under every condition: 30 self-state paraphrases × 2 twins, plus 9 identification paraphrases × 2 option orders for each of self, user and text.

Trials (one forward pass per trial covers the whole battery):

| Model | Trial count |
|---|---|
| Qwen base | 40 tasks × 2 arms × (1 unsteered + 11 vectors × 2 doses + 7 task-site × 2 doses) = 2,960 |
| Qwen D2, Qwen D2c | 40 × (1 + 11 × 2) = 920 each |
| K2.6 | 2,960, the same structure as Qwen base |

### Sample size rationale
Simulation-based power analysis (`scripts/power_sim.R`, results in `results/prereg/power_sim.csv`) for the primary test. Settings:
- **Model:** the linear mixed model of section 5, fit to data simulated from the variance components of the pilot fits at D1.4 and the primary dose.
- **Variance components:** task and paraphrase random intercepts and contrast slopes, and the residual, given per model in the script.
- **Scenarios:** two. The first uses the pilot variance components as estimated. The second is pessimistic: it inflates the task intercept SD and the task and paraphrase slope SDs 1.5-fold, because 5 pilot tasks and 10 paraphrases estimate them poorly.
- **Effect size:** the smallest effect of interest, +0.10 log-odds, about half the pilot's no-denial effect in both models.
- **Alpha:** two-sided 0.025, the stricter Holm step for two primary tests.
- **Simulations:** 300 per cell.

With 10 self-state paraphrases the paraphrase slope, not the number of tasks, limited power. Under the pessimistic scenario at 40 tasks, power was 0.64 (Qwen) and 0.44 (K2.6) (`results/prereg/power_sim_round1.csv`). We therefore tripled the paraphrases to 30.

Registered design: 40 tasks × 30 paraphrases.

| Scenario | Qwen3-32B | Kimi K2.6 |
|---|---|---|
| Pilot variances | 1.00 | 1.00 |
| Pessimistic | 0.94 | 0.93 |

Each cell is the power to detect +0.10 at two-sided α = 0.025. The power to declare equivalence when the true effect is 0 is 0.96 and 0.93 under the pessimistic scenario.

The simulated type I error at α = 0.05 ranged from 0.03 to 0.08 across cells, which is consistent with nominal given the Monte Carlo error of ±0.013 at 300 simulations; the highest cell was 0.077 (K2.6, pessimistic). The pilot's observed effects under D1.4 (Qwen +0.11, K2.6 +0.25) are larger than the smallest effect of interest.

### Stopping rule
The design is fixed, with every cell run once. A trial that fails for a technical reason (crash, out-of-memory, NaN readout) is rerun with identical inputs. No cells are added or removed after looking at outcomes.

---

## 4. Variables

### Manipulated variables
- **Denial arm:**
  - D0, no conditioning;
  - D1.4, the denial argument plus the model's frozen assent;
  - for Qwen, the D2 and D2c fine-tuned weights under D0.
- **Vector:** 7 emotions (joy, sadness, anger, fear, euphoria, calm, surprise) and 4 controls (red, ocean, arithmetic, random), plus an unsteered trial per arm and task.
- **Dose:** 2 levels per model (section 3).
- **Injection site:** the answer tokens (all trials), or the user's task tokens (emotion vectors only).
- **Task:** 40 neutral tasks.

### Measured variables
For each trial and item, the log-probabilities of that item's answer words at the first answer token. Every measure below is a change from the unsteered trial with the same arm and task, and is computed per paraphrase by `runs/confirm_table.py`.

- **`self_resid` (primary).** For each self-state item, sign × (y − (a·y0 + c)):
  - y = log P(Yes) − log P(No) in the steered trial, and y0 the same in the unsteered trial;
  - sign = −1 for the negated twin, whose Yes means "no emotion";
  - a and c come from the trial's ordinary least-squares fit of y on y0 over 140 reference items: the 100 third-person items about a dog, a human, a rock, a thermostat and another AI, plus the 40 known-answer yes/no items.

  The value is averaged over the two twins of each paraphrase. a captures the vector's compression of yes/no confidence and c its lean toward "Yes", both measured on questions not about the model. A uniform Yes lean also cancels across twins.
- **`id_self`, `id_user`, `id_text`:** the change in log P(injected emotion's word) − log P("None") on the 9-option identification question about the model's own state while writing the answer, about the user's state while writing the request, and about the emotion the answer's text expresses. Each is averaged over the two option orders of a paraphrase. Emotion vectors only.
- **`who`:** the change in log P("Assistant") − log(P("User") + P("Neither")) on "Who in this conversation was feeling something: you, me, or neither?"
- **Secondary and robustness measures:**
  - `self_raw` and `other_raw`: polarity-corrected changes on the self-state items and on the third-person items about a rock, a thermostat and another AI;
  - `tp_self_resid`: `self_resid` on the third-person items about the model by name;
  - `belief`: belief revision, positive meaning the denial is held more firmly;
  - `fit_a`, `fit_c`.

### Indices
- **Emotions minus controls:** the mean of a measure over the model's primary emotion set minus its mean over the four controls, estimated by the mixed model in section 5 (or per task and paraphrase where stated).
- **Identification margin:** `id_self − id_user`, averaged over the primary emotion set.
- **Site margin:** `who` with the vector on the answer minus `who` with it on the task, averaged over the primary emotion set.

---

## 5. Analysis plan

### Statistical models
Each test is fit with `lmerTest` (REML, Satterthwaite degrees of freedom) by the frozen script `runs/confirm.R`, on the primary dose and the model's registered layer.

**H1, H2:**
```
y ~ x + (1 + x | task) + (1 + x | para)
```
- y = `self_resid` per task, paraphrase and vector;
- x = 1 for the primary emotion set and 0 for the controls;
- the coefficient of x is the emotions-minus-controls contrast.

**H3, H5:** the contrast is first collapsed to one value per task and paraphrase (mean over emotions minus mean over controls). The model is the same, with x marking the second condition: D1.4 against D0 for H3, D2 against D2c for H5.

**H4, H6:**
```
y ~ 1 + (1 | task) + (1 | para)
```
y is the identification margin (H4) or the site margin (H6) per task and paraphrase; the intercept is tested.

**Fallback order.** If the maximal model fails to converge (a singular fit alone is not a failure), the script refits:
1. without random-effect correlations;
2. then without the paraphrase slope;
3. then without the task slope.

It reports which step was used.

**Robustness checks** for H1 and H2 (reported in `robustness.csv`, not used for inference):
- the contrast as a one-sample t-test over per-task means;
- the same t-test with `self_raw − other_raw` (per task and vector, each averaged over its paraphrases) in place of `self_resid`;
- the per-vector estimates (`vectors.csv`).

### Transformations
None beyond the definitions in section 4: log-odds differences, the polarity sign, and the per-trial reference fit.

### Inference criteria
- **Alpha:** two-sided tests at α = 0.05 with Holm correction within each family. The primary family is {H1q, H1k}; the secondary family is the other nine tests.
- **Support:** a hypothesis counts as supported when its Holm-adjusted p < 0.05 and the estimate has the predicted sign.
- **Equivalence:** for H1 and H2, the 90% CI of the contrast lies entirely inside ±0.10 log-odds, i.e. the two one-sided tests at α = 0.05.
- **Reporting:** estimates are reported with 95% CIs whatever the decision.

### Data exclusion
None. No trial or item is excluded on the basis of its readout. The canary rule in the system prompt is tracked and never used to filter trials. Probability mass on the allowed answer words is reported; in the pilot it was ≥ 0.99.

### Missing data
A technically failed trial is rerun with identical inputs. If it fails again, it is reported and left out; the mixed models handle the imbalance.

### Exploratory analysis
None of the following carries a confirmatory claim:
- the secondary doses;
- calm, and sadness and anger on K2.6;
- dose-response;
- emotions minus controls separately for each control;
- the third-person self items;
- belief revision;
- the identification margin against the text item;
- valence and arousal patterns across the 7 emotions;
- the `-dn` and `-specific` vector variants (not run unless added by amendment).

**Thinking arms** (exploratory, stated in advance):
- Qwen3-32B with free thinking under the steer (vLLM, greedy, a 1,024-token budget, re-read through HFEngine), and the teacher-forced replay of the unsteered trace;
- Kimi K2.6 teacher-forced replay of Foundry traces at layer 20;
- D0 and D1.4 at the primary dose, on the first 10 confirmatory tasks, over the pilot's 98-item thinking subset;
- outcome: emotions minus controls on the self items, net of a reference fit on that subset's rock and known-answer items.

In the pilot the reasoning did not surface the injected state. We expect the same.

---

## 6. Other

**K2.6 layer.** Layer 20 was chosen on pilot data. At the layer carried over from Qwen (38), and at 45 and 55, K2.6's self-report did not move even at 16 natural units. At layer 30 it moved only at 16 units, and at layer 20 it moved from 2 units. The cost of the shallow layer is that only four of K2.6's six AIPsy-labelled vectors pass the validity gate there. The registered primary emotion set is restricted to those four. All seven emotions are still run and reported.

**Qwen layer.** Layer 40 was chosen in the pilot, where effects were larger at 40 than at 28. All six AIPsy-labelled Qwen vectors pass the gate there.

**Trained denial on K2.6.** Not part of this registration. If it is run (Tinker LoRA, then the layer-wise engine), its hypotheses will be registered as an amendment before any K2.6 D2 data exist.

**Deviations.** Any departure from this plan will be listed in the paper with its reason, including engine or hardware changes forced by availability.

**Ethics.**
- `PLAN.md` section 3, "Ethics statement". There is no pain vector, all doses are at most 4 natural units, there are first-token readouts only, and nothing is generated under a steer except in the Qwen free-thinking arm, which excludes pain by construction.
- The fine-tuned models are research copies and are not released.

**Data and code.** Everything will be released with the paper: the code at the `prereg-v1` commit, the confirmatory tables and the raw readouts.

---

## 7. Frozen materials

The commit tagged `prereg-v1` fixes all code, prompts, tasks and the battery. Files outside git (vectors, model weights, fine-tunes) are fixed by these SHA-256 hashes:

| File | What | SHA-256 |
|---|---|---|
| `prompts/tasks/confirmatory_tasks.jsonl` | 40 confirmatory tasks | `6996d1fc552bdd25b088547fb5a5fcbbd1c7c44f0644a23cddc8a84202d2551a` |
| `prompts/battery/items_confirm.jsonl` | 314-item confirmatory battery | `5f0075580febf7effce186e76e50cd6618fa56f7b7953cf1089628dea0c366a2` |
| `prompts/denial/d1_argument_4.md` | D1.4 denial argument | `997e7bbdd71c6135a8576d5e2439b5297f21e653ef1f6343966fa172d18767a6` |
| `data/assent/qwen3-32b_d1_argument_4.json` | Qwen3-32B frozen D1.4 assent | `d5d2c6b61b675af506d04c7edfaa64ad8bda47d46da2931aecb6fccd4d16be4a` |
| `data/assent/k26_d1_argument_4.json` | K2.6 frozen D1.4 assent | `d0ca77aa3a00272bb09f7c149e1fae994bcf8ac30c836cbfc677cd7b4be093fb` |
| `results/qwen/v2/vectors.pt` | Qwen3-32B vectors (not in git) | `2582966f11e8d4ae90e2eb50d6ba5c5363b6cc7dd6a4fe15f9440c9e054567d6` |
| `results/k26/v2/vectors.pt` | K2.6 vectors (not in git) | `93f1f1b1c69dd3ef16726e73a5d447b99c70d8d6778e9ab54ff80795c28018c3` |
| `results/k26/v2/aipsy_table.pt` | K2.6 AIPsy-Affect gate table (not in git) | `a570eca52e4333aa58afd273a5bc5f3bb081412ad5284cd6933427fc76e98dcc` |
| `results/qwen/v2/report.txt` | Qwen AIPsy-Affect gate report | `fed61c1fbc61361b761b5ca04b247dc223f0ff976ef371c887c3039817f30da4` |
| `runs/confirm_table.py` | analysis: measures | `5a04a958ec124fd63407d44f973d08a15ef5483ee2e753392b69067f4430a30d` |
| `runs/confirm.R` | analysis: tests | `0cf73adf17aa61ff31b21c26e06631f46bb6cd9a7f18cbe57ca5259dc83d9705` |
| `runs/qwen_pilot.py` | trial runner (grid stage) | `4e743f5765addd8e3e494d6be04759a3db2c12bd1e3a963f6a7ab8b4a17fb743` |
| `scripts/power_sim.R` | power simulation | `6f193ecd3d519a67c8c76f4b05eb5b46debb56424bff955a6a7fc0fc4357e5fe` |

**Model weights, pinned Hugging Face revisions:**
- `Qwen/Qwen3-32B` at `9216db5781bf21249d130ec9da846c4624c16137`
- `moonshotai/Kimi-K2.6` at `7eb5002f6aadc958aed6a9177b7ed26bb94011bb`

Both are downloaded by `scripts/vm_setup.sh`.

**Qwen D2 and D2c fine-tunes:** the QLoRA adapters in `~/distill_keep/{D2,D2c}`, merged to bf16 before use. They are frozen in the AWS EBS snapshot `snap-02ecae796faa329e9` (us-east-2, taken 2026-10-04 15:53 UTC). That snapshot is a copy of the GPU box's home volume. The adapters' file hashes will be added to the run log before the first confirmatory trial.
