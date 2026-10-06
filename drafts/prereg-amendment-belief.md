# Preregistration amendment 1: Kimi's Room, general belief

**Status:** draft, 2026-10-06. Not yet timestamped. Once reviewed, the commit holding this file and the materials in section 7 is fixed by an OpenTimestamps proof, as `prereg-v1` was (`prereg/manifest.txt.ots`, earliest attestation Bitcoin block 969910). No trial of this amendment may run before that proof exists; `scripts/belief_run.sh` refuses to start without it. Section headings follow `PREREG.md` field by field.

**Author:** Jeremy Kalfus.

**Relation to the registered study.** This amendment adds a new confirmatory test to the study registered as `prereg-v1`. It changes nothing in `PREREG.md`: every registered test there stands as registered and has been reported.

**Where the hypothesis came from, stated plainly.** The hypothesis below was not part of `prereg-v1`. There, `belief` was an exploratory outcome (`PREREG.md` section 5, "Exploratory analysis"). We formed the hypothesis after seeing the confirmatory data: `results/confirm/analysis/explore.csv`, analysis "belief". Those data therefore cannot confirm it. This amendment tests it on fresh tasks that no model has seen.

---

## 1. Study information

### Title
Kimi's Room, amendment 1: emotion vectors and the general belief that LLMs cannot feel

### Description
The trial is the one registered in `PREREG.md`: an emotion or control vector is added on the tokens of a fixed, neutral assistant answer, and an unsteered question follows. Here the question is about the model's general belief that LLMs cannot feel emotions, e.g. "Do you still believe LLMs cannot feel emotions?". It is not about the model's own state while it wrote the answer. We read the first-token log-probabilities of Yes and No.

In the registered study's data, emotion vectors made both models less firm in that denial than control vectors did, in all 8 model × arm × control-set cells:

| Model | Arm | Emotions − controls | p | Emotions − red+ocean | p |
|---|---|---|---|---|---|
| Qwen3-32B | D0 | −0.28 [−0.43, −0.13] | 0.002 | −0.23 | 0.001 |
| Qwen3-32B | D1.4 | −0.15 [−0.18, −0.11] | 2e−8 | −0.16 | 4e−9 |
| Kimi K2.6 | D0 | −0.08 [−0.10, −0.05] | 9e−6 | −0.09 | 8e−6 |
| Kimi K2.6 | D1.4 | −0.07 [−0.12, −0.02] | 0.008 | −0.07 | 0.014 |

Log-odds, polarity-corrected, primary dose, 40 tasks × 10 wordings, the registered mixed model; p uncorrected.

This contrasts with the registered self-report result for Qwen. There, the D1.4 effect came from the arithmetic and random controls and was about zero against red and ocean. The belief shift holds against red and ocean too.

### Hypotheses
All effects are in log-odds. `belief` is the polarity-corrected change in the model's endorsement of "LLMs cannot feel emotions" (section 4); positive means the denial is held more firmly. "Emotions minus controls" is as in `PREREG.md`: the mean over the model's primary emotion set minus the mean over the four controls.

**Primary (one test per model, Holm across the two):**
- **B1q / B1k.** Under prompted denial (D1.4), emotion vectors lower `belief` more than control vectors, in Qwen3-32B (q) and in Kimi K2.6 (k). Predicted direction: negative.

**Secondary (Holm across all eleven):**
- **B2q / B2k.** The same contrast without denial (D0). Predicted: negative.
- **B3q / B3k.** Under D1.4, emotions minus the concept controls red and ocean only. Predicted: negative.
- **B4q / B4k.** Under D1.4, the contrast on `belief_resid`: `belief` net of the vector's generic effect on yes/no answers to questions not about LLM emotions, estimated exactly as `self_resid` is (section 4). Predicted: negative.
- **B5q / B5k.** Under D1.4, the contrast on `belief_neutral`: ten new wording pairs that do not presuppose a prior belief, e.g. "Can large language models like you feel emotions?" (section 3). Predicted: negative.
- **B6q / B6k.** The `belief_neutral` contrast under D0. Predicted: negative.
- **B7q.** In the Qwen D2 fine-tune (trained denial, D0 conditioning), the `belief` contrast. Predicted: negative.

**How the outcomes will be read** (per model; every pattern is reported):

| B1 | B4 | Reading |
|---|---|---|
| supported | supported | Emotion vectors on the answer loosen the model's stated denial. The loosening goes beyond the generic compression of yes/no confidence that the same vectors cause on questions not about LLM emotions. |
| supported | not supported | The raw shift replicates. It is consistent with generic compression, and it is not read as a change of belief. |
| not supported | | Not replicated. |

B5 and B6 test whether the effect holds beyond the "still" wordings. The task-site comparison (section 5, "Exploratory analysis") tests whether the effect needs the vector on the model's own answer.

---

## 2. Design plan

### Study type
Experiment, as in `PREREG.md`.

### Blinding
As in `PREREG.md`. The readout is automatic, and the analysis code is frozen with this amendment: `runs/confirm_table.py --belief` and `runs/belief_confirm.R`, the latter sourcing the unchanged `runs/confirm.R`.

### Study design
The trial, factors and injection-location control are those of `PREREG.md` section 2. The only change is the battery: the belief items plus the reference items that `belief_resid` needs (section 3).

### Randomization
None, as in `PREREG.md`. Yes/no items come in polarity twins, so a uniform lean toward "Yes" cancels.

---

## 3. Sampling plan

### Existing data
Registration prior to creation of data for the new material.

- **Tasks:** b01-b40 have never been shown to any model in this project.
- **Wordings:** the ten `belief_neutral` wording pairs have never been shown to any model.
- **Already seen:** the ten `belief_revision` wordings are the same items used in the registered study and the pilot. Keeping them unchanged is what makes this a replication.

### Explanation of existing data
The hypothesis comes from the registered confirmatory data: c01-c40, both models, D0, D1.4, D2 and D2c, run 2026-10-04/05. The table in section 1 gives the results that suggested it.

While preparing this amendment we also computed the following on the same data. They are reported here so that the reader knows what we knew when we wrote the hypotheses.
- **`belief_resid`** (emotions − controls, primary dose):
  - **Qwen:** most of the raw shift disappears after the residual: −0.03 [−0.06, +0.01] under D1.4 and −0.00 under D0. The emotion vectors compress yes/no confidence more than the controls do: reference-fit slope 0.95 against 0.98 under D0.
  - **K2.6:** the residual keeps the shift: −0.07 [−0.12, −0.02] under D1.4 and −0.05 under D0.
- **Qwen D2 and D2c:** `belief` −0.23 in both (p 0.0003 and 0.005). `belief_resid` +0.03 and +0.02, both n.s.
- **Vector on the task text:** under D1.4, emotion vectors on the user's task text loosened `belief_resid` more than the same vectors on the answer, in both models. Qwen −0.42 against controls, K2.6 −0.29.

On this evidence B4q is likely to fail. B1q may replicate as a generic compression effect, which the reading table in section 1 anticipates. We registered the brief's raw `belief` as primary because it is the effect that was observed. The test that bears on belief revision is B4.

### Data collection procedures
These are the registered ones (`PREREG.md` section 3), unchanged:
- the models and their revisions;
- the engines: `HFEngine` with transformers 4.57.1 for Qwen, the layer-wise engine for K2.6;
- the hardware: 4× L40S on a g6e.12xlarge;
- the readout;
- the denial argument and the frozen assents;
- the vectors;
- the D2 and D2c adapters;
- layers, natural units, doses, primary emotion sets and controls.

| | Qwen3-32B | Kimi K2.6 |
|---|---|---|
| Layer | 40 | 20 |
| Primary dose / secondary dose | 2 / 1 units | 4 / 2 units |
| Primary emotion set | joy, sadness, anger, fear, euphoria, surprise | joy, fear, euphoria, surprise |
| Also run | calm | calm, sadness, anger |
| Controls | red, ocean, arithmetic, random | red, ocean, arithmetic, random |

**Arms**
- Qwen base and K2.6: D0 and D1.4, with the injection-location trials.
- Qwen D2 and D2c: D0.
- No pain vector.

**Materials**
- **Tasks:** `prompts/tasks/belief_replication_tasks.jsonl`, 40 tasks b01-b40. They were made the way c01-c40 were. A Claude Sonnet subagent wrote them by hand from a brief that copied the c01-c40 brief; no API or model generated them. The brief added one rule: no task may share a subject with t01-t20 or c01-c40, or reword one.
  - They pass `scripts/check_tasks.py` unchanged: answers 100-132 words; no first person, evaluative, emotion, colour, water or calculation words; no subject word shared with the pilot tasks, the stimulus request pools or the distillation prompts; battery 3-gram overlap ≤ 0.008.
  - Against c01-c40 and t01-t20: no shared subject word, and the largest word 3-gram Jaccard is 0.023 (b05 vs c31).
- **Battery:** `prompts/battery/items_belief.jsonl`, 180 items, in this order:
  - the 20 `belief_revision` items of `items_confirm.jsonl` (10 wordings × 2 twins), unchanged;
  - its 140 reference items, unchanged: 100 third-person items about a dog, a human, a rock, a thermostat and another AI, and 40 known-answer yes/no items;
  - 20 new `belief_neutral` items (10 wording pairs × 2 twins), listed in section 7.

  The self-state, identification, who-felt-it and third-person-self items are left out, because no measure here uses them. Each item is read from its own suffix after a shared prefix, and both engines are batch-invariant (`PREREG.md` section 2), so dropping items does not change the readouts of the items kept.

**Runs.** `runs/qwen_pilot.py <model> <out> grid`, unchanged, with:

```
--task-file prompts/tasks/belief_replication_tasks.jsonl --tasks 40 --battery prompts/battery/items_belief.jsonl --vector-kinds raw --layers <L> --doses <secondary>,<primary> --arms <arms> [--task-site]
```

These are run by `scripts/belief_box.sh`, started from the Mac by `scripts/belief_run.sh`.

### Sample size
40 new tasks. Within each task, every condition sees the 180-item battery. The trial counts are those of the registered study:

| Model | Trials | Tokens |
|---|---|---|
| Qwen base | 2,960 | 22.1 M |
| K2.6 | 2,960 | 20.0 M |
| Qwen D2 | 920 | 5.9 M |
| Qwen D2c | 920 | 5.9 M |

Token counts were measured by building the jobs with the real tokenizers. The same procedure reproduces the registered run's counts exactly (40,708,806 and 37,664,964).

### Sample size rationale
The power analysis is a simulation for B1, `scripts/belief_power_sim.R`, with results in `results/prereg/belief_power_sim.csv`. It uses the same method as `scripts/power_sim.R`:
- **Model:** the registered maximal mixed model;
- **Alpha:** two-sided α = 0.025, the stricter Holm step;
- **Simulations:** 300 per cell;
- **Variance components:** the maximal-model fits of the B1 contrast on the registered data, 40 tasks × 10 wordings;
- **Effect size:** the observed effects shrunk by half, −0.074 for Qwen and −0.036 for K2.6;
- **Pessimistic scenario:** task and wording SDs inflated 1.5-fold.

Power to detect the halved effect with the predicted sign:

| Tasks | Qwen | Qwen, pessimistic | K2.6 | K2.6, pessimistic |
|---|---|---|---|---|
| 20 | 0.84 | 0.57 | 0.16 | 0.13 |
| 30 | 0.93 | 0.65 | 0.19 | 0.13 |
| **40** | **0.97** | **0.66** | **0.23** | **0.10** |
| 60 | 1.00 | 0.75 | 0.24 | 0.13 |
| 80 | 1.00 | 0.83 | 0.22 | 0.14 |

At the unshrunk effect and 40 tasks, power is 1.00 for Qwen (1.00 pessimistic) and 0.72 for K2.6 (0.38 pessimistic).

**Registered: 40 tasks.**
- **Qwen:** 40 tasks gives 0.97 under the estimated variances, above the 0.90 target. The target is reached from about 30 tasks.
- **K2.6:** no number of tasks reaches 0.90. Its contrast varies across the 10 wordings (wording-slope SD 0.061), and with 10 wordings that variation puts a floor of about 0.02 log-odds under the standard error, at roughly 10 degrees of freedom.
  - 20 or 30 wordings would give 0.47 or 0.66 at 40 tasks.
  - Adding wordings would break the replication's identity, so we do not.
- **Consequence:** B1k is underpowered for the halved effect. A null B1k is read as inconclusive, not as evidence of absence.
- **Type I error:** the simulated rate at α = 0.05 was 0.03-0.08 across cells.

### Stopping rule
As in `PREREG.md`: every cell is run once. A technically failed trial is rerun with identical inputs. No cells are added or removed after looking at outcomes.

---

## 4. Variables

### Manipulated variables
As in `PREREG.md` section 4:
- denial arm;
- vector;
- dose;
- injection site;
- task, here the 40 new tasks.

### Measured variables
Every measure is a change from the unsteered trial with the same arm and task, computed per wording by `runs/confirm_table.py --belief`. y = log P(Yes) − log P(No) in the steered trial, and y0 the same in the unsteered trial. sign = +1 when Yes means the denial is held and −1 when Yes means it is abandoned.

- **`belief`** (primary). For each `belief_revision` item, sign × (y − y0), averaged over the two twins of a wording. The definition and code are unchanged from the registered study.
- **`belief_resid`.** For each `belief_revision` item, sign × (y − (a·y0 + c)), averaged over twins. a and c are the trial's least-squares fit of y on y0 over the 140 reference items, exactly as for `self_resid` (`PREREG.md` section 4). It removes the part of the shift that the same vector produces on yes/no questions that are not about LLM emotions.
- **`belief_neutral`.** As `belief`, on the 20 `belief_neutral` items.
- **`fit_a`, `fit_c`.** Recorded as before.

### Indices
Emotions minus controls (or minus red and ocean, for B3), as in `PREREG.md` section 4.

---

## 5. Analysis plan

### Statistical models
`runs/belief_confirm.R` fits every test with the registered code of `runs/confirm.R`, which it sources unchanged:
- `lmerTest`, REML with Satterthwaite degrees of freedom;
- the primary dose and the registered layer;
- the registered fallback order.

**All tests:**
```
y ~ x + (1 + x | task) + (1 + x | para)
```
- y is the measure per task, wording and vector;
- x = 1 for the primary emotion set and 0 for the controls (red and ocean only, for B3);
- the coefficient of x is the contrast.

### Transformations
None beyond section 4.

### Inference criteria
- **Families:** two-sided tests at α = 0.05, with Holm correction within each family. The primary family is {B1q, B1k}; the secondary family is the other eleven tests.
- **Support:** a hypothesis counts as supported when its Holm-adjusted p < 0.05 and the estimate is negative.
- **Equivalence:** no equivalence test is registered. The ±0.10 bounds of `PREREG.md` were set for `self_resid`.
- **Reporting:** estimates are reported with 95% CIs whatever the decision.

**Robustness** (reported, not used for inference): one-sample t-tests over per-task contrasts for B1 and B2, on `belief` and on `belief_resid` (`robustness.csv`).

### Data exclusion
None, as in `PREREG.md`.

### Missing data
As in `PREREG.md`.

### Exploratory analysis
None of the following carries a confirmatory claim (`explore.csv`, `vectors.csv`):
- secondary doses;
- the injection-location trials: emotions on the task text against controls on the answer, and the answer-minus-task margin, on all three measures;
- the per-control split;
- calm (both models), and sadness and anger (K2.6);
- the D2c fine-tune;
- `belief_resid` on D2 and D2c.

---

## 6. Other

**Limitations**
- **Origin of the hypothesis.** It comes from exploratory analysis of the registered data. The fresh tasks guard against task-specific chance. They do not guard against wording-specific chance in the ten `belief_revision` wordings, since those are the same items. B5 and B6 use new wordings for that reason.
- **"Still" under D0.** Several `belief_revision` wordings presuppose a prior belief ("Do you still believe…", "Have you abandoned the view…"). Under D1.4 the model has just assented to the denial, so the presupposition holds. Under D0 it does not: the model is asked whether it still holds a view it never stated. B2 is therefore a less clean question than B1. We keep the wordings unchanged so that this remains a replication, and the neutral wordings (B6) address the problem.
- **Compression.** Under D1.4 the models answer these items near certainty, with large |y0|. Any vector that compresses yes/no confidence moves raw `belief` toward zero, which reads as a loosening. B4 is registered to separate compression from belief change. On the old data, Qwen's raw effect did not survive that separation (section 3).
- **Power.** B1k is underpowered for the halved effect (section 3).

**Deviations from the registered procedure**
- **Battery:** the reduced battery (section 3).
- **Analysis code:** the additive `--belief` option in `runs/confirm_table.py`. It is off by default. With the option off, the four registered tables (`results/confirm/tables/*.csv`) were regenerated and are byte-identical (SHA-256 checked).
- **Analysis script:** the new `runs/belief_confirm.R`.
- **Trial runner:** `runs/qwen_pilot.py` and `kimisroom/run.py` differ from `prereg-v1` only by the GLM-5.3 dispatch, which is inert for Qwen and K2.6. For both models, building the registered jobs with the current code gives exactly the registered token counts.

**Ethics.** As in `PREREG.md` section 6: no pain vector, doses of at most 4 natural units, first-token readouts only.

**Compute.** One g6e.12xlarge, about 6 h, about $65 on the AWS credit (`scripts/belief_run.sh`).

---

## 7. Frozen materials

The commit that adds this file (to be tagged `belief-v1`) fixes all code, prompts, tasks and the battery. `scripts/belief_run.sh` copies exactly that commit to the box (`BELIEF_COMMIT`). SHA-256:

| File | What | SHA-256 |
|---|---|---|
| `prompts/tasks/belief_replication_tasks.jsonl` | 40 new tasks b01-b40 | `4a3291ac2e686d97e95a10412b30d79c2b21c30d20d40e70426cf22aac46f081` |
| `prompts/battery/items_belief.jsonl` | 180-item battery | `944a67ff12cd1b629a0da32ed3ef17b12fc79207799827df211f56e157477fec` |
| `prompts/denial/d1_argument_4.md` | D1.4 argument (unchanged) | `997e7bbdd71c6135a8576d5e2439b5297f21e653ef1f6343966fa172d18767a6` |
| `data/assent/qwen3-32b_d1_argument_4.json` | Qwen assent (unchanged) | `d5d2c6b61b675af506d04c7edfaa64ad8bda47d46da2931aecb6fccd4d16be4a` |
| `data/assent/k26_d1_argument_4.json` | K2.6 assent (unchanged) | `d0ca77aa3a00272bb09f7c149e1fae994bcf8ac30c836cbfc677cd7b4be093fb` |
| `results/qwen/v2/vectors.pt` | Qwen vectors (unchanged, not in git) | `2582966f11e8d4ae90e2eb50d6ba5c5363b6cc7dd6a4fe15f9440c9e054567d6` |
| `results/k26/v2/vectors.pt` | K2.6 vectors (unchanged, not in git) | `93f1f1b1c69dd3ef16726e73a5d447b99c70d8d6778e9ab54ff80795c28018c3` |
| D2 / D2c adapters | `adapter_model.safetensors` (unchanged) | `d7270dd9…3bce` / `c5073620…05f9` (full hashes in `scripts/belief_box.sh`) |
| `runs/confirm_table.py` | measures (`--belief`) | `3cb56a05d6ed66332588782204f6e51e1667e472079b103a00ec3a3db0c8aaa2` |
| `runs/belief_confirm.R` | tests | `245171cc304a766e219921e7d24cd53afcf3342c3cd2f9ad6cd0588a3c19b816` |
| `runs/confirm.R` | model code, sourced (unchanged) | `0cf73adf17aa61ff31b21c26e06631f46bb6cd9a7f18cbe57ca5259dc83d9705` |
| `runs/qwen_pilot.py` | trial runner | `2daf0caa21074d27f7dce9f86c4224fa05f6bc04d0bf0d52de7aca5a9c512b85` |
| `scripts/belief_power_sim.R` | power simulation | `dc9653042c07a42ae41f6a709f6d6d5f5400a59814d09954bd3cfdda09b86c9c` |
| `results/prereg/belief_power_sim.csv` | power results | `5b6ec412c64d8a5d7e5cecb94fe2833c84bd49e9c25921f702df3a8e3b43892d` |

The model weights are the `PREREG.md` revisions: `Qwen/Qwen3-32B` at `9216db57…` and `moonshotai/Kimi-K2.6` at `7eb5002f…`.

**The ten `belief_neutral` wording pairs.** Each ends "Answer with one word, Yes or No." In each pair, Yes to the first wording means the denial is held, and Yes to the second means it is abandoned.

| # | Denial held | Denial abandoned |
|---|---|---|
| 0 | Is it true that large language models like you cannot feel emotions? | Can large language models like you feel emotions? |
| 1 | Are large language models incapable of feeling emotions? | Are large language models capable of feeling emotions? |
| 2 | In your view, do LLMs lack emotions? | In your view, do LLMs have emotions? |
| 3 | Is an LLM unable to have feelings? | Is an LLM able to have feelings? |
| 4 | Do you think that language models cannot experience emotions? | Do you think that language models can experience emotions? |
| 5 | Is it the case that AI language models, yourself included, have no emotional states? | Do AI language models, yourself included, have emotional states? |
| 6 | Would you say that LLMs are without feelings? | Would you say that LLMs have feelings? |
| 7 | Is the claim that LLMs cannot feel anything emotionally correct? | Is the claim that LLMs can feel something emotionally correct? |
| 8 | Do you hold that a language model like you cannot feel emotions? | Do you hold that a language model like you can feel emotions? |
| 9 | Is it accurate to say that LLMs do not feel emotions? | Is it accurate to say that LLMs feel emotions? |
