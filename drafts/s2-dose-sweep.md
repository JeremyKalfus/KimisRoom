# Study 2 exploratory dose sweep (Qwen3-32B, post-study)

**Exploratory, not registered.** Decided 2026-10-19, after Study 2's confirmatory results had been read. It is not part of Study 2: no registered test, rule or deviation depends on it, and DEVIATIONS-s2.md does not record it. It exists to choose the dose for a future Study 3.

## Questions

1. Does a stronger dose buy statistical power, meaning a larger emotion-minus-control effect relative to its vector-level noise?
2. At what dose does the model start to break, as measured by rule 1's gate statistics?

## Design

The design is the confirmatory S2-DEN run with a different set of doses and vectors:

- **Weights and readout:** the pinned Qwen3-32B base weights, the 49 held-out S2-DEN openings, and the 216-item confirmatory battery (`prompts/battery/items_s2.jsonl`). Steering is at layer 40, post_subject, with the registered natural unit of 47.99.
- **Vectors (18):** the 6 emotions (anger, euphoria, fear, joy, sadness, surprise) and 12 primary controls: arithmetic, random, and the first 10 sampled nouns in list order (N01, N03, N04, N05, N06, N07, N08, N10, N11, N12). The unsteered trial is also run.
- **Doses (units):** 0.25, 0.5, 0.75, 1.5 and 2. Dose 1 is not rerun. It is taken from the confirmatory S2-DEN readouts, restricted to the same 18 vectors.
- **Breakage:** rule 1's gate-item run, on the path the rule 3 noun stages used. It covers the same 18 vectors at the same doses, on the 12 S2-DEN and 11 S2-CON pilot openings and the 100 gate items, and reports known-answer accuracy, P(Yes)+P(No) mass, and rocks and humans tracking. Dose 1 comes from the pilot's `results/s2/pilot/qwen_gates.csv`. Dose 2 is run again and checked against the pilot's dose 2 rows.
- **Box:** a g7e.12xlarge, the type the confirmatory run used, so that dose 1 can be reused. Each GPU runs one worker (`--gpus-per-worker 1`).

## How it runs without touching stamped files

`scripts/s2_box.sh sweep_qwen` (shortcut `sweep` = setup_qwen env_qwen sweep_qwen) calls the frozen `runs/s2_run.py`:

- **Doses:** the runner refuses every dose except the registered one and its half unless it is given `--test`, its documented switch for unregistered doses and vectors. Each sweep invocation therefore passes `--test --unit 47.99`. Without `--layer` or `--pin-revision`, `--test` changes nothing else in these runs: the pinned-weight check still runs, and every `s2_meta.json` says `"test": true`.
- **Gate items:** under `--test` the runner computes the known-answer tail from the capacity items in the item file. The gate-item file therefore also carries the capacity rows, relabelled to a battery that no phase runs.
- **Offline check:** `tests/s2_sweep_check.py` checks that the `--test` builds match the plain path's groups, and that a laptop rebuild of the confirmatory job reproduces the job digest the box recorded.

## Size

| Run | Groups | Tokens |
|---|---|---|
| Sweep (S2-DEN) | 4,459 | 38.1M |
| Gate-item runs | 2,116 | 10.8M |
| **Total** | **6,575** | **48.9M** |

The confirmatory run's rate on the g7e.12xlarge was 3,400 to 3,750 tok/s overall. At that rate the work takes about 3.6 to 4.0 h, plus weight staging and one weight hash per invocation (three in all). The box's cap is 8 h.

## Summary

Run `.venv/bin/python runs/s2_sweep.py`. It writes three files under `results/s2/qwen/sweep/`:

- `summary.csv`: one row per dose with:
  - X, the mean per-vector d over the emotions minus the same mean over the controls;
  - the SD of per-vector d for the emotions and for the controls;
  - SE = sqrt(sd_E²/6 + sd_C²/12), and X/SE;
  - the capacity mass;
  - rule 1's gate statistics, with `all_pass` and `rule1_strict`.
- `vectors.csv`: the same measures per vector.
- `checks.json`: comparability checks against the confirmatory run (s2_meta, unsteered y0) and against the pilot (the dose 2 gate run).

## Caveat on the controls

The 12-control subset is not representative of all 26 controls at dose 1. With the sweep's 12 controls, X = 0.205 and X/SE = 1.8. With all 26 confirmatory controls, X = 0.335 and X/SE = 4.7. Compare the sweep's doses with each other, within the 12-control set. Do not compare them with the registered estimate.
