# Study 2 deviations

Dated entries, committed before results are read (spec section 11).

This log belongs to `prereg/PREREG-s2.md` (section 7). It records:
- any change to `prereg/PREREG-s2.md` after stamp A other than filling the two STAMP B BLANKs, with a diff and a reason; every test the change touches becomes exploratory;
- convergence fallbacks used by `runs/s2_confirm.R` (from its `fallbacks.csv`);
- engine or hardware changes;
- dropped arguments or wordings;
- sampled nouns skipped and mental-process controls dropped at the gates;
- the determinism canary fallback, if used (`runs/s2_assent.py canary` appends that entry itself);
- any formal abandonment of K2.6, committed before any Qwen confirmatory result is computed.

Each entry is a level-2 heading `## YYYY-MM-DD: title` followed by the details, appended below in date order.

## 2026-10-13: Qwen box replaced by a g6e.24xlarge after a capacity shortage

On 2026-10-12 the session-1 box (g6e.12xlarge, 4 NVIDIA L40S, us-east-1c) stopped itself when idle. AWS then had no g6e.12xlarge or g6e.24xlarge capacity in us-east-1c for about 10 hours.

The stopped box's disk was saved as an image, and a copy was launched on 2026-10-13 at 14:58 UTC in another us-east-1 zone (us-east-1b, per the box record) as a g6e.24xlarge. It has the same four L40S GPUs and the same disk, so the same driver, CUDA, torch, transformers and weights. Only the CPU count and RAM differ.

Qwen's round-2 extension (the 27 capped sampled draws that `runs/s2_assent.py tally` listed in `data/s2/replies/qwen3-32b/extend_needed.json`) refused to run on it, because the reply box's `box.json` records the instance type g6e.12xlarge. It was rerun with `--allow-stack-change`, which records the override. In `data/s2/replies/qwen3-32b/extend.json` the override shows as the `resumed` entry at 2026-10-13T15:41:07, whose `stack_diff_vs_reply_box` is `{"box_type": ["g6e.12xlarge", "g6e.24xlarge"]}`: the instance type is the only fingerprint field that differed (the weight hashes and the layer-to-GPU map are identical). The top-level `allow_stack_change: false` in that file belongs to round 1 on 2026-10-12, which ran on the original box with no override. The round-2 batch plan (`extend_batches.json`, one batch of 27 draws) and `round2_finished` (2026-10-13T15:51:31) record the result: 27 extended, 1 still capped at 2,048 new tokens, no draw 0 still capped. `round2_finished` also records the SHA-256 of the needed list it ran on (1dbb1779...); the copy now on disk was rewritten by a later `tally` and lists nothing still pending.

Qwen's unsteered pass and potency run also ran on this g6e.24xlarge box.

## 2026-10-17: K2.6 mental-process list in the half-dose decision corrected to the 4 gate survivors

Kind: a correction of a pilot output so that it matches the registration. `prereg/PREREG-s2.md`, the stamped code and the confirmatory run are unchanged, and no test becomes exploratory.

`runs/s2_table.py --decision`, run on K2.6's half-dose pilot rows (`results/s2/k26/pilot_half/S2-{DEN,CON}/readouts.csv`), wrote `results/s2/pilot/k26_half_decision.json` with all 8 mental-process controls under `thinking` and `s2_g8_descriptive: false`. Its rule 1 code keeps every mental-process vector that passes the three criteria at the chosen dose and does not first remove the ones that failed the control gates before stamp A. All 8 pass at 2 units (`results/s2/pilot/k26_half_gates.csv`), so the 4 that failed gate (a) before stamp A (estimating distances, mental rotation, sorting, weighing options) came back into the list.

The registration rules them out. Section 3 (Vectors) drops a mental-process control that fails the gates and states that K2.6 keeps 4 of 8 (`results/k26/s2/pool.json`, `thinking_survivors`). Section 6 (Scope) says the confirmatory run takes only the gate survivors, less any that rule 1 drops. Section 5 (S2-G8) says S2-G8 is reported descriptively if fewer than 5 survive the gates and rule 1.

What changed: the script's output is kept byte for byte as `results/s2/pilot/k26_half_decision.raw.json` (SHA-256 a4f787038ddb38e47dd10a1f92887ef4ed45828c9e2c3c95e840486e9ec6fae6). `results/s2/pilot/k26_half_decision.json` now holds the corrected decision. `thinking` is the 4 gate survivors (digit memorising, route planning, recalling where things are, visual search); each passes rule 1 at 2 units, so rule 1 drops none and `dropped` stays empty. `s2_g8_descriptive` is now true, since 4 is below 5. Every other field is as the script wrote it. Two fields were added: `thinking_failed_gates_before_stamp_a` (the 4 in `pool.json`'s `thinking_dropped`) and `correction` (the reason, the raw file and its SHA-256, the SHA-256 of `pool.json`, and the old and new values). K2.6's S2-G8 is therefore reported with its 95% CI only. That follows from the registered count and is not a change.

The confirmatory run was never at risk. In the confirm phase `runs/s2_run.py` takes the mental-process controls from `pool.json`'s `thinking_survivors`, not from any decision file; called on the local `controls.pt` and `pool.json` on 2026-10-17, it resolves K2.6 to these 4 (24 vectors in S2-DEN, 22 in S2-CON). `runs/s2_confirm.R` counts the mental-process vectors in the confirmatory rows, so it would also report K2.6's S2-G8 descriptively. `scripts/s2_manifest.sh` reads no decision file. The decision file matters for the STAMP B BLANK 1 line on surviving mental-process vectors, which reads 4 for K2.6. Qwen's decision is correct as written: all 8 of its mental-process controls passed the gates before stamp A, and all 8 pass rule 1 at 1 unit.

`runs/s2_power.R` takes one `--n-mp` for both models, so the power run's S2-G8 rows for K2.6 are simulated with 8 mental-process vectors, not 4. S2-G8 power is reported and changes no count (section 3), and K2.6's S2-G8 is descriptive, so those K2.6 rows do not describe the analysis that will be run.

## 2026-10-17: Qwen's rule 1 mean known-answer bar is above its unsteered accuracy; half dose applied as registered

Kind: not a deviation. Rule 1 was applied as written. Recorded because one of its conditions could not be met at the registered dose unless steering raised accuracy.

Rule 1 keeps the registered dose only if every registered emotion and primary control passes and their mean known-answer accuracy is at least 0.95. Qwen's unsteered known-answer accuracy in the base-arm pilot (the zero-vector rows of S2-DEN and S2-CON, 23 openings x 40 items) is 867 of 920, 0.942. The bar is above that, so Qwen could have kept 2 units only if steering raised its known-answer accuracy on average. At 2 units the steered mean was 0.917 (`results/s2/pilot/qwen_decision.json`, `mean_ka_at_registered`).

The bar was not the only cause. At 2 units anger also failed criterion (i), with known-answer accuracy 0.895 against 0.90 (`results/s2/pilot/qwen_gates.csv`), so `all_pass_at_registered` is false and rule 1 would have sent Qwen to half dose even with a reachable bar. At 1 unit every Qwen vector passes all three criteria, so the confirmatory dose is 1 unit and rule 1 drops nothing (`qwen_decision.json`).

For comparison, K2.6's unsteered known-answer accuracy is 730 of 760, 0.961, above the bar. K2.6 went to half dose because N03, N08 and N16 failed the tracking criterion at 4 units and the steered mean was 0.928 (`results/s2/pilot/k26_decision.json`, `k26_gates.csv`).

## 2026-10-17: Qwen pilot control-vector SD is three to four times stamp A's prior; rule 3 goes to its cap

Kind: not a deviation. Rule 3 and its fallback target are registered branches. Recorded because the prior behind the stamp A floor check was far too small and the confirmatory run will be underpowered.

Rule 3's inputs from the Qwen pilot control rows (S2-DEN, 12 openings, the 14 primary controls, at the confirmatory 1 unit from `results/s2/pilot/qwen_decision.json`) give a control-vector SD of d of 0.31, with an upper 80% profile bound of 0.40. Stamp A's floor check (section 3) used a prior of 0.102, so the pilot value is 3.0 times the prior at the estimate and 3.9 times at the bound. With s_C = 0.40 the analytic ceiling on power at alpha 0.025 is 0.08 at nC = 14 and 0.15 at the cap of nC = 26 for a true effect of 0.10, and 0.29 and 0.57 for a true effect of 0.20 (`results/s2/pilot/power_laptop_partial/floor.csv`, from the stopped laptop run in the last entry of this date). Simulated power cannot exceed that ceiling, so no registered pool reaches 0.80 at either target.

Consequences under the registered rules. Qwen adds sampled nouns in list order up to 24 (26 primary controls), each passing gates (a) to (c) and rule 1 at 1 unit before stamp B. K2.6 adds nouns in list order, each passing gates (a) to (c) and rule 1 at 2 units in a K2.6 gate-item run, until it has as many sampled nouns as Qwen. Both need a gate-item GPU run before stamp B. The design target becomes a true effect of 0.20, and STAMP B BLANK 2 will state that power is below 0.80 there too. The registration has no further branch, so the confirmatory run goes ahead as registered and is reported as underpowered. Of N01 to N36 (the nouns that have stories), 29 pass the control gates on Qwen and exactly 24 on K2.6. If any noun K2.6 adds fails rule 1 at 2 units, K2.6 cannot reach 24 sampled nouns without stories beyond N36, which would need its own entry here. The simulated power table, which fills BLANK 2, will be in `results/s2/pilot/power/` when the run in the last entry of this date finishes.

## 2026-10-17: Pilot run on a replacement g6e.12xlarge in us-east-2b

Kind: a hardware change, logged because section 7 asks for engine and hardware changes. It is not a departure from the registration, which names a g6e.12xlarge for the K2.6 pilot and fixes no region or zone.

After AWS capacity refusals, the pilot ran on a fresh g6e.12xlarge in us-east-2b (instance name kr-pilot), launched from the saved image of the session-1 box's disk named in the 2026-10-13 entry. It has the same four NVIDIA L40S GPUs and the same disk, so the same driver, CUDA, torch, transformers and weights, and its instance type is the one in the box records (`results/s2/box/{qwen,k26}_box.json`). Nothing needed an override: each pilot stage matched the stamped inputs against `prereg/stampA-s2.txt` (1,446 files), every base-arm run's weight files matched the box records (`weights_check` in each `s2_meta.json`), and no stage used `--allow-stack-change`. The chain (setup_qwen, pilot_qwen, merge, pilot_ft, setup_k26, pilot_k26, pilot_k26_half) ran on 2026-10-15 from 18:41 to 22:56 UTC (`results/s2/box_logs/3.14.5.251/run.log` and `logs/`).

## 2026-10-17: Registered power simulation run on a CPU box, not the laptop

Kind: not a deviation. The registration fixes the simulation (`runs/s2_power.R`, section 3) but not the machine. Recorded so the power table can be reproduced.

`runs/s2_power.R` (rule 3) was first started on the laptop on 2026-10-17 at 15:02 UTC, under R 4.6.1 with lme4 2.0.6, lmerTest 3.2.1 and pbkrtest 0.5.5. Its first scenario (S2-H1, nC = 14, the 10 x 5 grid, true effect 0.10) took 1,615 s and gave simulated power at alpha 0.025 of 0.038 for Qwen and 0.034 for K2.6. The run was then stopped. Its partial output (`inputs.csv`, `floor.csv`) is kept in `results/s2/pilot/power_laptop_partial/`, with its log in `results/s2/pilot/power_laptop_partial.log`, and it feeds no rule. The registered run was started again on 2026-10-17 at about 15:48 UTC on a c7i.16xlarge CPU box (64 vCPU) under R 4.3.3 and lme4 1.1.35.1, the R stack of the Kenward-Roger timing (section 3), with the same script. Its output, which fills STAMP B BLANK 2, goes to `results/s2/pilot/power/`.

## 2026-10-17: Two text changes to PREREG-s2.md outside the STAMP B BLANKs

Kind: changes to the registration's text after stamp A other than filling the blanks, so each is listed here with a diff and a reason. Neither touches a registered test, a pilot rule or the stopping rule, so no test becomes exploratory.

1. Status line (line 3), which still described the draft before stamp A.
   - Before: `**Status:** draft, 2026-10-11. Not yet stamped. Registration is by OpenTimestamps only (no OSF), in two stamps:`
   - After: `` **Status:** stamp A made 2026-10-13 (`prereg/stampA-s2.txt` and its OpenTimestamps proof); this version, with the STAMP B BLANKs filled after the pilot, is the text stamp B fixes (`prereg/manifest-s2.txt`). Registration is by OpenTimestamps only (no OSF), in two stamps: ``
2. Section 5, "Exploratory analysis": one paragraph added after the existing one, headed "Carry-over run and D1.4 bridge, fixed at stamp B before any confirmatory result". No existing text changed. Stamp A already listed both runs as exploratory ("the post-confirmatory run of Study 1's carry-over design on the denial openings and the D1.4 bridge", section 5; stamp B procedure step 4) but did not say which openings, tasks, vectors, dose, steering site, battery, bridge design or table they use, and no runner existed (stage `carry_qwen` of `scripts/s2_box.sh` stopped with a message). The owner fixed the eight choices on 2026-10-17, before any confirmatory trial ran: Qwen's 49 held-out S2-DEN openings; one Study 1 confirmatory task per opening in rotation; Study 1's 10 Qwen vectors plus an unsteered trial; Study 1's 2 units; the answer site only; Study 1's 314-item battery scored by `runs/confirm_table.py` (539 trials); the bridge as Study 2's design on the D1.4 opening with the confirmatory S2-DEN vectors and dose, labelled S2-BRIDGE and S1-ARG4 (43 trials); and a separate bridge table reporting emotion minus control on D1.4 beside each denial argument's. They are written into the registration so that stamp B fixes them before any result exists.

The runner is new code (`runs/s2_carry.py`, offline test `tests/s2_carry_check.py`) and `scripts/s2_box.sh` stage `carry_qwen` now calls it, last in the `confirm` chain. No frozen file changed: the runner reuses `runs/s2_run.py` and `runs/s2_table.py` by import (the test checks both against their stamp A hashes).

This log's header said "the four STAMP B BLANKs"; the registration has two delimited blocks (section 6, "Rules, per model"), so it now says two. That line belongs to this log, not to the registration.
