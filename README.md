# Kimi's Room

> [!CAUTION] 
This README was written by Claude (Opus 5.5)

Code, data and preregistration for the experiment in [Kimi's Room (Preliminary Results)](https://www.lesswrong.com/posts/gXHhtGijBbs8q7iEY) on LessWrong.

We add an emotion vector to a model's residual stream on the tokens of a fixed answer to a neutral task. Then, with no vector present, we ask the model whether it was feeling an emotion while it wrote that answer, and read the log-odds of "Yes" against "No" on the first token. The score is that change net of what the same vector does to 140 questions that are not about the model. We compare emotion vectors (checked against AIPsy-Affect) with control vectors (red, ocean, arithmetic, random) in Qwen3-32B and Kimi K2.6 under three conditions: with no argument; after the model has been convinced by an argument that it cannot feel emotions; and, for Qwen, after fine-tuning on that argument, compared with a fine-tune on an argument that it cannot count letters.

## Contents

| Path | What it holds |
| --- | --- |
| `PREREG.md` | The preregistration: design, the 11 tests, analysis plan, frozen materials |
| `PLAN.md` | The research plan and the pilot results the preregistration builds on |
| `prereg/` | `manifest.txt` (SHA-256 of every preregistered file), its OpenTimestamps proof, and `verify.sh` |
| `kimisroom/` | The Python package: steered prefill engines, trial construction, the question battery, the assent judge |
| `prompts/`, `data/` | The battery, the arguments and the tasks; the models' frozen replies agreeing to the arguments; the stimuli the vectors were extracted from; the fine-tuning data; two third-party datasets |
| `runs/` | Entry points: vector extraction, pilots, the confirmatory tables (`confirm_table.py`), the registered tests (`confirm.R`), exploratory analyses (`confirm_explore.R`, `confirm_dose_test.R`, `confirm_scale.py`) and figures (`confirm_figures.py`) |
| `scripts/` | The AWS orchestration of the confirmatory run |
| `results/qwen/`, `results/k26/` | Pilot results, and the frozen vectors in `v2/` |
| `results/confirm/` | The confirmatory run: `analysis/` (`hypotheses.csv` has the 11 registered tests), `figures/`, and `tables/`, one row per trial, paraphrase and measure (gzipped) |
| `post/` | The figures in the post and the scripts that draw them |
| `tests/`, `gate/`, `rehearsal/`, `analysis/` | Checks, the engine-against-API gate, an end-to-end rehearsal on a small model, and the first API baseline |

## Check the preregistration

`PREREG.md` was committed and tagged `prereg-v1` on 2026-10-04. `prereg/manifest.txt` lists the SHA-256 of every file in that commit, plus the three frozen vector files, which were kept out of git at the time and are included here unchanged. The OpenTimestamps proof `prereg/manifest.txt.ots` places the manifest in Bitcoin block 969910, mined 2026-10-04 at 21:53 UTC, before the confirmatory run began.

```sh
sh prereg/verify.sh                  # every file in the manifest, against the tag prereg-v1
ots verify prereg/manifest.txt.ots   # the timestamp; needs the OpenTimestamps client and a Bitcoin node
```

Without a Bitcoin node, drop `manifest.txt` and `manifest.txt.ots` on [opentimestamps.org](https://opentimestamps.org).

`verify.sh` should report 330 of the 331 files matching. The other one is withheld (see the last section), so `prereg-v1` here has a different commit and tree hash from the ones the manifest names.

## Rerun the analysis

The tables reproduce the registered tests and the exploratory analyses in `results/confirm/analysis` byte for byte. With R and the lmerTest package:

```sh
gunzip -k results/confirm/tables/*.csv.gz
Rscript runs/confirm.R out results/confirm/tables/{k26,qwen-D2,qwen-D2c,qwen}.csv          # the 11 registered tests
Rscript runs/confirm_explore.R out results/confirm/tables/{k26,qwen-D2,qwen-D2c,qwen}.csv  # exploratory analyses
Rscript runs/confirm_dose_test.R                                                           # exploratory dose test
```

`scale.txt` comes from `runs/confirm_scale.py`, which needs the raw grids (not included, see below).

The post's figures need Python with matplotlib and numpy, and read the gzipped tables directly:

```sh
python post/fig1_main_result.py post/1-main-result.png
python post/fig3_preregistered_tests.py post/3-preregistered-tests.png
python post/fig5_every_vector.py post/5-every-vector.png
python post/fig8_whose_feeling.py post/8-whose-feeling.png
python post/fig9_belief.py post/9-belief.png
```

Figures 8 and 9 also read `post/fig8_fig9_models.csv`, the mixed models for the cells that no file in `results/confirm/analysis` holds. To refit them, from the repository root:

```sh
Rscript post/fig8_fig9_models.R post/fig8_fig9_models.csv
```

Figure 9 uses `belief`, the belief measure in the preregistration. `who_user.csv.gz` holds the Assistant-against-User margin on the who-felt-it question behind figure 8's right panel; the registered tables hold Assistant against User or Neither. `post/who_user_table.py` made it from the raw grids.

## Rerun the experiment

The experiment needs the model weights (`Qwen/Qwen3-32B` and `moonshotai/Kimi-K2.6` at the revisions in `PREREG.md`, section 7) and a large multi-GPU machine. `scripts/confirm_run.sh` ran the confirmatory study unattended on AWS. The scripts in `scripts/` were written for one AWS account: they read settings from a `.env.local` that is not included and fetch the AWS key from `kv`, a local secret store, so expect to adapt them.

Not included because of their size: the raw grids the tables were computed from (about 1.2 GB) and the LoRA adapters of the two fine-tunes (about 1 GB).

## About this copy

This repository holds the commits that made the preregistration and the confirmatory results, copied from the working repository with their original dates. Later, unrelated work and drafts are left out. The last commit adds this README, the license, the tables, the frozen vectors and the post's figures. One preregistered file is withheld: `drafts/aws-sales-request.md`, a request to AWS for GPU quota that includes account details.

## License

MIT (see `LICENSE`). `data/aipsy_affect` (Keeman 2026) and `data/pain_axis` (Tagliabue, Dung and Berg 2026) are pinned copies of third-party datasets, both MIT licensed; see the README in each folder.
