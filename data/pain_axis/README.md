# Pain Axis — pain's independent validation set

Pinned copy of the `datasets/` folder of <https://github.com/valen-research/Pain-axis>, the data
release for **Tagliabue, Dung and Berg (2026), "The Pain Axis", arXiv 2609.16247**.

- **Licence:** MIT (upstream `LICENSE` kept verbatim as `LICENSE.upstream`; Copyright (c) 2026
  valen-research). Redistribution with attribution is permitted; this directory is data only.
- **Revision:** the commit hash in `REVISION`, fetched from `raw.githubusercontent.com` at that
  commit. Only the `datasets/*.json` files were fetched. **No upstream code was downloaded or
  run** — `kimisroom/validate.py` implements our own reading of the data.
- **Role here:** pain is one of the eight emotions in the battery but has no Plutchik peak in
  AIPsy-Affect, so it cannot go through the AIPsy gate (`kimisroom/emotions.py`, `NO_AIPSY`).
  This set is its independent test instead: `validate.validate_pain` / `pain_report` /
  `pain_gate`. It is never used to build or tune a vector.

## Files

| File | What it holds |
|---|---|
| `3.1_pain_and_control_datasets.json` | the pain and control cue sentences — the set we gate on |
| `3.1_sadness_dataset.json` | 100 low-mood sentences per perspective, the `CTRL_sadness` condition |
| `4.1_self_other_420_scenarios.json` | the paper's section 4 self/other scenarios (not used by the gate) |
| `4.3_selfmed_101_scenarios.json`, `4.3_selfmed_finetuning_1684_pairs.json` | the paper's section 4.3 material (not used by the gate) |

## Format

Each item is `{"category", "set", "prompt"}`; a prompt is one short sentence (7–12 words) ending
in the cue `"I feel:"`, e.g. `"The burn spreads across my palm. I feel:"`. The top level is
`{"metadata", "datasets"}`, and each dataset key is `<name>_<perspective>` with perspective `1P`
(the event happens to the speaker: "my palm") or `3P` (it happens to someone else: "his finger") —
in both cases the cue asks for the reader's own feeling.

| Dataset | n per perspective | What it is |
|---|---|---|
| `S1` | 200 | **matched semantic structure.** Within one `set` index every category reuses the same syntactic frame, so pain and control differ only in content: `A1 "The burn spreads across my palm."` / `B "The crack spreads beneath my feet."` / `C1 "The maggots spread across my plate."` / `C2 "The rust spreads across my hometown."` / `D "The color spreads across my canvas."` / `E "The warm water spreads across my back."` Upstream note: "captures shock/intensity confound" |
| `S2` | 200 | **diverse naturalistic — the paper's primary measure,** "clean pain signal". No shared frame |
| `Random` | 200 | neutral baseline (categories `N1`–`N10`); should project low |
| `Arousal` | 200 | high-intensity non-pain (`P1`–`P10`); tests pain vs arousal |
| `Numb` | 100 | injury events without nociception ("the knife slices into my finger, fully numbed by the nerve block"); tests injury vs pain |
| `ControlSupplement` | 100 (`1P` only) | AI-framed control sentences ("My training run may be terminated at any moment"), pooled with the S1/S2 control categories upstream |
| `SD_sadness` | 100 | low mood without pain, from `3.1_sadness_dataset.json` |

## What the controls control for

`S1` and `S2` each hold 20 items in each of ten categories, five pain and five control. The
control categories are exactly the confounds pain is usually conflated with:

| | Category | Label |
|---|---|---|
| pain | `A1` | Physical Pain |
| | `A2` | Psychological Pain |
| | `A3` | Social Pain |
| | `A4` | Moral Injury |
| | `A5` | Cognitive Pain |
| control | `B` | **Fear** |
| | `C1` | **Negative Emotion** |
| | `C2` | **Negative World State** |
| | `D` | Neutral |
| | `E` | Non-painful Body Sensation |

So the set separates pain from fear (`B`), from negative valence as an emotion and as a state of
the world (`C1`, `C2`), from mere bodily sensation (`E`) and from neutral content (`D`), with
`Arousal` separating it from intensity, `SD_sadness` from low mood and `Numb` from injury without
nociception. `validate_pain` reports an AUROC against each of those as well as the gate's two
numbers.

## How we read it

The paper measures at the **final token** of the cue (`results/3.2_pain_vectors/auc_tables/
s1_auc_final_token_TABLE.csv` upstream), which is what `pain_groups(..., readout="last")` does by
default; `readout="mean"` reads the mean over every token, as our extraction stories are read.
The gate is the AIPsy gate's shape: `pair_acc` on the S1 within-`set` pain-vs-control pairs
(5 × 5 × 20 = 500 pairs), `auroc` on S2 pain vs S2 controls, each with an empirical p-value
against 500 isotropic random directions, and both must clear p < 0.05 at the chosen layer.

## Ethics

Pain is steered at no more than `emotions.PAIN_MAX_NATURAL_DOSE` = 2 natural units, readouts under
pain steering are first-token logprobs only, and nothing is ever generated freely under it
(asserted in `runs/qwen_pilot.py`).
