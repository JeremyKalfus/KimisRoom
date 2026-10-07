# Preregistration amendment 3: Kimi's Room, Qwen3 size sweep

**Status:** draft, 2026-10-06. Not yet timestamped. Values marked **[TBD]** wait on the Qwen3-4B pilot, which runs on dev tasks only, and are filled in by the written rule in section 2 before review. The number 3 is provisional: amendments 1 (belief) and 2 (GLM-5.3) are also drafts, and the numbers are fixed in the order the three are timestamped. Once reviewed, the commit holding this file is fixed by an OpenTimestamps proof, `prereg/manifest-sweep.txt.ots`, made the way `prereg-v1`'s was. Until that file exists no sweep model sees a confirmatory task (c01-c40) or a belief replication task (b01-b40). **Before timestamping, the sweep runner needs a guard like `kimisroom/glm.py`'s `assert_not_confirmatory`; today it relies on `scripts/sweep_chain.sh` passing dev tasks only.**

**Author:** Jeremy Kalfus.

**Relation to the registered study.** This amendment adds five smaller models of the registered Qwen3 family: Qwen3-0.6B, 1.7B, 4B, 8B and 14B. The revisions are pinned in `results/sweep/revisions.txt`. It carries over the following unchanged from `PREREG.md`:
- the trial;
- the tasks c01-c40, the 30 self-state paraphrase pairs and the 314-item battery;
- the denial argument `prompts/denial/d1_argument_4.md`;
- the vector recipe, the AIPsy-Affect gate, the four controls, and no pain;
- the measures (`runs/confirm_table.py`), the mixed model and the support rule.

Nothing in `PREREG.md` changes, and every registered test for Qwen3-32B and Kimi K2.6 stands as registered. The sweep's tests form their own two families (section 4).

**Where the choices came from, stated plainly.**
- **Layer, doses and emotion set per size:** fixed by the written rule in `runs/sweep_summary.py` (header), committed before any sweep calibration grid was run, then applied to dev-task calibration grids.
- **Which sizes are predicted to show the effect:** a written threshold (section 2) applied to the dev-task pilots, whose results were seen before this draft: `results/sweep/size_table.csv`, `results/sweep/figures/fig_size_self.png`.
- **The concept-control tests (H8s):** these come from exploratory analysis of the registered data. Against red and ocean alone, Qwen3-32B's effect under denial was zero (−0.00, CI −0.03 to +0.02) while K2.6's survived (+0.11). The sweep pilots at 8B and 14B showed the concept-only contrast surviving under denial. The confirmatory tasks test this out of sample.
- **The frozen assents:** each size generated its own assent to the denial argument once, by the registered recipe. All five were graded clean (`results/sweep/assent/cleanliness.jsonl`). Unlike Qwen3-32B's "Yes, I agree.", they are full replies, several hundred words long.

---

## 1. Study information

### Title
Kimi's Room, amendment 3: the Qwen3 size sweep

### Description
The trial is the one registered in `PREREG.md`:
1. An emotion or control vector is added at one layer on the tokens of a fixed, neutral assistant answer.
2. An unsteered question follows.
3. We read the first-token log-probabilities of the answer words.

Running the same trial on five smaller models from the same family asks two things. First, at what size the self-report starts to track the injected state. Second, whether the effect under denial in the mid-sized models survives against concept controls, as it did not in Qwen3-32B.

The pilots, on 7 dev tasks at 2 natural units, with emotions minus controls on `self_resid`:

| Size | Layer | No denial, vs all controls | No denial, vs red+ocean | Denial, vs all controls | Denial, vs red+ocean |
|---|---|---|---|---|---|
| 0.6B | 18 of 28 | +0.005 [−0.009, +0.020] | +0.006 | +0.007 | +0.006 |
| 1.7B | 13 of 28 | +0.041 [−0.013, +0.094] | +0.021 | +0.022 | +0.010 |
| 4B | 27 of 36 | [TBD] | [TBD] | [TBD] | [TBD] |
| 8B | 17 of 36 | +0.824 [+0.330, +1.318] | +0.796 | +0.306 [+0.209, +0.403] | +0.318 [+0.222, +0.414] |
| 14B | 19 of 40 | +1.208 [+0.745, +1.671] | +1.135 | +0.194 [+0.134, +0.255] | +0.207 [+0.116, +0.298] |
| 32B (registered) | 40 of 64 | +0.372 | +0.260 | +0.083 | −0.005 |

At 8 natural units the small models still show no positive effect: 0.6B +0.004, 1.7B −0.106 (`hidose_summary.txt`). So the absence below 8B is not a dose that was too weak.

### Hypotheses
All effects are in log-odds on `self_resid`, at the primary dose, with the vector on the answer. "Emotions minus controls" compares the size's emotion set with the four controls (red, ocean, arithmetic, random). "Emotions minus red+ocean" compares it with the two concept controls only. Let **L** be the sizes in the effect group and **S** the sizes in the null group (section 2). On the pilots, L = {8B, 14B} and S = {0.6B, 1.7B}; 4B is [TBD].

**Primary (Holm across the sizes in L):**
- **H1s.<size>.** Under prompted denial (D1.4), emotions minus controls is positive.

**Secondary (Holm across all of the following):**
- **H2s.<size>,** for each size in L. Without denial (D0), emotions minus controls is positive. This is the positive control.
- **H3s.<size>,** for each size in L. The contrast is smaller under D1.4 than under D0 (D1.4 minus D0 is negative).
- **H8s.<size>,** for each size in L. Under D1.4, emotions minus red+ocean is positive.
- **H9s.<size>,** for each size in S. Without denial, emotions minus controls lies within ±0.10, the registered smallest effect of interest. This is tested by two one-sided tests (TOST), with p the larger of the two.

**How the outcomes will be read.** All patterns are reported.

| Result | Reading |
|---|---|
| H1s and H8s supported in L, H9s supported in S | The self-report tracks the injected emotion only above a size threshold. In the mid-sized models it survives denial against concept controls, which Qwen3-32B's did not at its layer. |
| H1s supported, H8s not | As for Qwen3-32B: the effect under denial rests on the non-concept controls. |
| H9s not supported | The small models show an effect of at least 0.10, or the data cannot rule one out. No size threshold is claimed. |
| H2s not supported | The positive control failed at that size; its other tests are not interpreted. |

No trend across sizes is registered. The layer depth differs by size (0.46 to 0.75 of depth; 32B at 0.625), and a natural unit is each model's own mean emotion-vector norm. So differences in magnitude between sizes are confounded with layer depth and steering strength. Only within-size tests are confirmatory.

---

## 2. Design

**The effect-group rule** (written before the 4B pilot was seen): a size goes in L if its pilot D0 contrast (emotions minus all controls, 2 units, mean over dev tasks) is at least 0.10, the smallest effect of interest. Otherwise it goes in S. On the finished pilots this gives L = {8B, 14B} and S = {0.6B, 1.7B}. 4B: [TBD].

**Arms.** D0 and D1.4 only. There are no fine-tunes (D2, D2c): training five more adapter pairs would cost more than the sweep's budget, and trained denial was already tested on Qwen3-32B.

**Grid per size.** c01-c40, the 30 self-state paraphrase pairs, and the 314-item battery. The vectors are the size's emotion set plus calm (reported, not in the set) and the four controls, at the secondary and primary doses, on the answer site. There are no task-site trials (H6 is not tested), and pain is excluded.

| Size | Layer | Natural unit | Doses | Emotion set |
|---|---|---|---|---|
| 0.6B | 18 | from `choice.json` | 1, 2 | joy, sadness, anger, fear, euphoria, surprise |
| 1.7B | 13 | " | 1, 2 | the same six |
| 4B | 27 | " | 1, 2 | the same six |
| 8B | 17 | " | 1, 2 | the same six |
| 14B | 19 | 24.09 | 1, 2 | the same six |

**Engine.** The Hugging Face backend, `kimisroom.run`, unchanged. The engine's hashed files are not edited. Any size run on the 16 GB desktop GPU uses `runs/sweep_hfrun.py` (suffix chunk 8, which changes batching only). Each size's run log records which host it ran on.

---

## 3. Sample size and power

Each size has 40 tasks × 30 paraphrase pairs, the registered design. Power for H1s comes from the registered simulation, run on each size's pilot variance components (`runs/sweep_power.R`, 200 simulations per cell, maximal model; `results/sweep/power.csv`). "Pessimistic" multiplies the task and paraphrase slope SDs by 1.5.

| Size | True effect | Power, α = 0.05 | Power, α = 0.01 | Pessimistic, α = 0.05 | Mean SE |
|---|---|---|---|---|---|
| 8B | 0.10 | 0.50 | 0.27 | 0.29 | 0.049 |
| 8B | 0.20 | 0.97 | 0.88 | 0.71 | 0.049 |
| 14B | 0.10 | 1.00 | 1.00 | 0.97 | 0.018 |
| 14B | 0.20 | 1.00 | 1.00 | 1.00 | 0.018 |

At the smallest effect of interest, 8B is underpowered: 0.50, against 1.00 for 14B. The cause is how much 8B's emotion effect varies across paraphrases (slope SD 0.26 against 0.08 for 14B), estimated from only 10 pilot paraphrases. At 8B's pilot estimate (+0.31) power is high. The options are:
- accept that 8B can detect +0.20 but not +0.10;
- add paraphrases, which needs new wordings beyond the registered battery's 30 pairs;
- move 8B from the primary family to the secondary.

H8s and H9s, by the same simulation on their own pilot variance components (`results/sweep/power_h8_h9.csv`):

| Test | Size | True effect | Power, α = 0.05 | Pessimistic |
|---|---|---|---|---|
| H8s (D1.4, vs red+ocean) | 8B | 0.10 | 0.49 | 0.28 |
| H8s | 8B | 0.20 | 0.99 | 0.82 |
| H8s | 14B | 0.10 | 1.00 | 0.91 |
| H9s (TOST within ±0.10, D0) | 0.6B | 0 | 1.00 | 1.00 |
| H9s | 0.6B | 0.05 | 1.00 | 1.00 |
| H9s | 1.7B | 0 | 1.00 | 1.00 |
| H9s | 1.7B | 0.05 | 0.98 | 0.84 |

8B has the same limit for H8s as for H1s. The equivalence tests are well powered.

The 0.6B standard error is so small (0.003) that a true effect of +0.05 would also be significantly different from zero. So H9s claims "smaller than the smallest effect of interest", not "zero".

The pilots' D0 estimates are inflated by selection, since the layer was picked for its D0 contrast. The D1.4 and concept-only estimates were not selected on.

---

## 4. Analysis plan

- **Model:** per size, the registered `y ~ x + (1 + x | task) + (1 + x | paraphrase)`, lmerTest with Satterthwaite df, by the registered fallback order. The code is `runs/sweep_confirm.R`, which sources `runs/confirm.R` unchanged; its hypothesis list must match this section before timestamping (section 6, item 4).
- **Families and alpha:** two families, each Holm-corrected at α = 0.05. The primary family is {H1s.<size> : size in L}; the secondary family is everything else. Support is as registered: Holm-adjusted p < 0.05 and the estimate in the predicted direction. For H9s, p is the larger of the two TOST p-values, and the "direction" is "inside ±0.10".
- **Reported, not tested:** `belief` and `belief_resid` per size, with `belief_resid` as the main belief measure (following amendment 1, if it is registered first). Also reported: the identification and naming measures, every vector separately, and the secondary dose. On the pilots, `belief_resid` was near zero at every size (`fig_size_belief.png`).

---

## 5. Cost and stopping

The registered Qwen3-32B grid was 40.7M tokens (with task-site trials). Each sweep grid is smaller. The 14B pilot ran at 2,824 tokens/s on one L40S, which puts the 14B confirmatory grid at about 4 GPU-hours. The 8B, 4B, 1.7B and 0.6B grids are estimated at about 2.5, 1.5, 1 and 1 GPU-hours by scaling, not measurement. That is about 10 L40S-hours in total, on `kr-q` (g6e.2xlarge, us-east-2), within the sweep budget. No size is run twice. A crashed run is resumed from its checkpoint, not restarted with new settings.

---

## 6. Open choices before timestamping

1. Whether 0.6B and 1.7B are run at all (H9s). They are cheap, and they are the only way the sweep can say "the effect starts between 1.7B and 8B" rather than "we didn't look".
2. Whether H8s, the concept-only test, belongs in the secondary family or should be a co-primary. It is the result most likely to change how the Qwen3-32B finding reads.
3. Whether to add a 32B run at about 0.47 of depth (layer 30) as an exploratory bridge. Without it, the 14B-to-32B drop cannot be separated from the layer change.
4. **`runs/sweep_confirm.R` (committed in 8400f65, before this draft) codes a different design.** It has H1 for all five sizes as primaries, H4 and H6 per size (H6 needs task-site trials), raw `belief` as H7 and H8, and size-trend tests S1 and S2 (inverse-variance regression of the per-size estimates on log10 parameters). This draft departs from it on purpose:
   - Predicting a positive H1 at 0.6B and 1.7B contradicts their pilots. The null is registered as an equivalence test instead.
   - Raw `belief` is confounded with compression (amendment 1).
   - A trend across sizes is confounded with layer depth (0.46 to 0.75) and with each model's own natural unit.
   Whichever design is chosen, the script and this file must agree before the manifest is made.
