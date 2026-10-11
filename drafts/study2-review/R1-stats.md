# R1 @stats: model specification, inference, power

Bottom line: the H1 model in spec section 9 is roughly the right shape, but as specified (a) power is capped by the number of vectors, and none of the section 8 rule 4 levers (wordings, paraphrases) can raise it; (b) the raw-vs-fit rule lets a compression artefact several times the SESOI into the primary; (c) several random terms needed for honest SEs are missing, especially for the arm interactions. Findings in priority order.

## stats-1 (critical) Vectors, not openings, set the SE floor; rule 4 pulls the wrong levers
x varies only between vectors. With a vector random intercept of SD s_v, SE(x) >= s_v * sqrt(1/nE + 1/nC) however many openings and paraphrases there are: 0.50 s_v for Qwen (6 + 12), 0.58 s_v for K2.6 (4 + 12), with df near nVectors - 2 (16, 14). For 80% power at beta 0.10, two-sided alpha 0.025 (Holm step 1), you need SE <= 0.10 / (t(0.9875, 16) + t(0.80, 16)) = 0.10 / 3.34 = 0.030, so s_v <= 0.060 (Qwen) and <= 0.051 (K2.6), and that is before any argument-slope variance. Study 1 shows controls are far more spread than that on the self item: under denial arithmetic -0.26 and random -0.24, red and ocean near the emotions (drafts/lesswrong/kimis-room-v4.md:55-60). The subject contrast may cancel some of this, but nobody knows how much. Rule 4 (section 8) only allows more wordings (up to 10) or paraphrases (up to 15): both shrink terms that are not binding. Study 1's power sim had no vector term at all (scripts/power_sim.R:13-21, vectors fixed), so this was never checked.
Fix: (1) add a pilot gate on the vector component: compute the analytic floor above with the upper 80% bound of s_v from the pilot; (2) make "add control concepts" the first lever in rule 4 (say up to 24 or 30 controls, built from the reserve list and more, by the same story pipeline and AIPsy-failure rule) since that is the only lever that helps; (3) if the floor still fails at beta 0.10, register beta 0.20 as the design target and say so, rather than claim 80% power at 0.10.

## stats-2 (critical) The raw-vs-fit rule admits a bias several times the SESOI
Algebra: within a trial group a and c are shared, so d_fit - d_raw = sign x (1 - a) x (y0(you) - y0(LM)); c cancels. The emotion-minus-control contrast in d_raw therefore carries a bias of (a_C - a_E) x g, g = the signed unsteered gap. Rule 2 tolerates |g| up to 2 log-odds. In the old Qwen pilot (results/qwen/qwenpilot_compression.txt, answer site) emotion slopes are 0.54 to 0.67 and control slopes 0.64 to 0.92 (ocean 0.92, red 0.83, random 0.82): a_C - a_E around 0.2. With g = 2 that is 0.4 log-odds of artefact, four times the SESOI, and with you more denied than LM (the expected case) it points in the H1 direction. Even g = 0.5 gives 0.10. The rule as written is not protective. (Caveat: that file is an earlier pilot at a different site and dose; the size is indicative, the algebra is not.)
Fix: replace rule 2 with one on the predicted bias, not the gap: compute B = mean over pilot DEN groups of (a_C - a_E) x g; d_raw is primary only if |B| < 0.02 (a fifth of the SESOI), else d_fit. Better: make d_fit primary unconditionally and report d_raw, since the fit is the only measure that handles the known compression. Also check the reference set spans y0(you) and y0(LM): with Qwen at about -23 under denial (kimis-room-v4.md:45) the fit may be an extrapolation; register a check that the reference items' y0 range covers the you and LM values, or add floor-level reference items (false known-answer facts at matching depth).

## stats-3 (major) Missing random terms; paraphrase must be nested in frame
The section 9 model omits vector-by-opening and vector-by-paraphrase variation. Each (opening, vector) group contributes 20 rows (10 paraphrases x 2 frames) that share whatever that vector does in that opening; leaving it in the residual understates SE(x). Paraphrases are frame-specific templates (section 5), so if they are coded 1..10 in both frames the model wrongly crosses capacity item 3 with state item 3. The shared unsteered y0 is common to every vector in an (opening, item) cell; it does not bias x (balanced) but belongs in a cell intercept rather than the residual.
Fix, the corrected H1 model (d = d_fit per opening, vector, frame, paraphrase pair):
```
d ~ x * frame_c
  + (1 | vector) + (1 | vector:argument) + (1 | vector:argument:wording)
  + (1 | vector:frame:paraphrase)
  + (1 + x | argument) + (1 + x | argument:wording)
  + (1 + x | frame:paraphrase) + (1 | argument:wording:frame:paraphrase)
```
frame_c = +-0.5 (sum coded), so the x coefficient is the average over frames, not the capacity-frame effect, and x:frame is estimated rather than forced to zero. Convergence fallback after the existing steps: drop vector:argument:wording, then vector:frame:paraphrase; never vector or vector:argument.

## stats-4 (major) H3 and H5 are anticonservative as specified
DEN and CON (and TD and TC) share the same vectors, so the x-by-arm interaction is a within-vector contrast. If each vector's effect differs by arm (it did in Study 1: arithmetic and random fell under denial far more than red and ocean, kimis-room-v4.md:55-60), the interaction SE must include vector-by-arm variance. Arguments are nested in arm (denial vs control arguments) for H3, which also needs separate argument variances per arm. TD vs DEN differs in weights and opening (section 14), so no between-arm contrast has a clean argument structure.
Fix: H3/H5 model: d ~ x * arm + (1 + arm | vector) + (1 + x | argument) [argument nested in arm, with arm-specific variances via (0 + DEN + DEN:x | argument) + (0 + CON + CON:x | argument)] + the stats-3 terms; simulate H3 power separately (it will be lower than H1's).

## stats-5 (major) Heterogeneous vector populations and what the emotion term estimates
One (1 | vector) assumes emotions and controls are draws with the same variance. They are not one population: K2.6 has 4 emotions chosen by a gate, and the controls mix 11 concepts with a Gaussian random direction, which Study 1 showed behaves unlike the concepts. With class-specific variances, Satterthwaite df for x behaves like Welch's and for K2.6 can fall to about 3 to 5 (4 emotions), which no extra opening can fix.
Fix: state the estimand: "the mean of the registered emotion set (fixed) versus the population mean of non-emotion concepts (random)". Model it with class-specific vector variances, (0 + emo + ctl || vector), and register that. Add two registered robustness checks: concept controls only (drop random), and an exact vector-label permutation test on vector-level means of d (C(18,6) = 18,564 relabelings for Qwen, C(16,4) = 1,820 for K2.6, minimum p 1/1,820), which generalises over vectors without distributional assumptions.

## stats-6 (major) Singular fits and small-sample df
With about 18 vectors the vector SD is often estimated at or near 0. lmerTest then returns large Satterthwaite df and the test silently treats vectors as fixed, which is exactly the anticonservative case the "never drop the vector intercept" rule meant to block. With only 7 to 10 arguments (section 4 allows down to 7), Satterthwaite is also liberal for the argument slope.
Fix: use Kenward-Roger df (pbkrtest) for H1 and every x test; if the vector variance is singular, cap df for x at nVectors - 2 (or report the permutation test of stats-5 as primary). Register both rules now.

## stats-7 (major) The power simulation cannot be calibrated as planned
Section 10 takes argument and wording components from a pilot with one wording per argument, so they are confounded; setting wording = argument is an arbitrary guess. The pilot has 10 DEN arguments, so the argument-slope SD has about 9 df, and s_v about 16; point estimates will be noisy and probably low. The sim also omits vector:argument and vector:paraphrase terms, and does not simulate the exclusion rule.
Fix: simulate the stats-3 model with: s_v and argument-slope SDs at the upper 80% confidence bound from the pilot (profile CIs), wording SD at 0.5 and 1.0 times argument SD as two scenarios, the worst case the exclusion rule allows (7 arguments x 3 wordings) as well as the full grid, Kenward-Roger df, the convergence fallback chain actually run in each replicate, and Holm across both models (report P(both significant) as well as each). Report the analytic floor of stats-1 next to the simulated power.

## stats-8 (minor) H8 equivalence cannot succeed, and its Holm placement is odd
With SE(x) around 0.03 at best (stats-1), a 90% CI inside +-0.10 needs the estimate within about +-0.05 of zero; with a larger s_v, equivalence power is near zero, so a non-significant H1 will be "inconclusive" whatever the truth. H8 is conditional on H1 failing, yet is placed in the Holm family with H2 to H7, so its alpha depends on how many unrelated tests failed.
Fix: report the planned equivalence power from the sim and state in advance that a null with no equivalence is inconclusive. Take H8 out of the secondary Holm family: test it as TOST at alpha 0.05 conditional on H1 being non-significant (it is a closed sequence with H1, so no extra correction is needed).

## stats-9 (minor) SESOI 0.10 is not on a common scale across models or measures
0.10 log-odds means different things at Qwen's -23 baseline and Kimi's -9 (kimis-room-v4.md:45), and d_fit and d_raw have different scales whenever a < 1. Study 1's Qwen primary came in at 0.08, below its SESOI, and was still called supported.
Fix: either justify 0.10 per model (for example as a fixed fraction, such as a quarter, of the CON-arm emotion effect measured in the pilot) or keep 0.10 and register that the claim is "beta > 0 and the CI excludes values below 0.10", not just p < alpha.

## stats-10 (minor) Sidedness, dose rule
H1 is directional but the test is two-sided at Holm 0.05, which matches a one-sided 0.025 in the predicted direction: fine, but say what a significant negative x means (it is a result, not a failure to replicate). Rule 1 averages known-answer accuracy over all vectors, so one vector can collapse comprehension while the mean stays at 0.95. Fix: require each vector to pass (accuracy >= 0.90) and the mean to pass 0.95, or drop any failing vector from both classes by a fixed rule before confirmatory.

## Proposed power plan, in order
1. Pilot (wording 0) on Qwen, both doses: fit the stats-3 model with class-specific vector variances; extract profile CIs.
2. Analytic floor: SE_min = sqrt(s_e^2/nE + s_c^2/nC) with upper-bound SDs; if beta 0.10 needs SE_min > 0.030, add controls (to 24 or 30) before anything else.
3. Simulate (500 reps) the full model with the scenarios in stats-7, Kenward-Roger df, fallback chain, exclusion worst case, Holm across models; also H3 and the permutation test.
4. Register the counts that achieve 0.80 at beta 0.10 if feasible, else register beta 0.20 as the design target.
