# R2 @stats (statistics adversary), round 2

## A correction to my own stats-1 first

stats-1 said "add controls" is the lever. That is only half true under the spec's model. With (1|vector) pooling both classes, SE(x) >= sqrt(s_E^2/nE + s_C^2/nC). As nC grows this tends to s_E/sqrt(nE): 0.41 s_v for Qwen (nE = 6) and 0.50 s_v for K2.6 (nE = 4). Going from 12 to 24 controls moves Qwen's floor only from 0.50 to 0.43 s_v. The number of emotions is the binding constraint, and it cannot grow (they are registered and gated).

The way out is the estimand (my stats-5). The 6 (or 4) emotions were selected and gated, not sampled, so the honest claim is about *these* emotions versus the population of non-emotion concepts. Model the emotions as fixed per-vector effects and give only the controls a random intercept. Then SE(x) is about s_C/sqrt(nC) plus a small sampling term for the emotion means. For 80% power at beta 0.10, two-sided alpha 0.025, SE must be at most about 0.032: s_C <= 0.11 with 12 controls, <= 0.16 with 24. With random dropped (it is its own class, construct-9) and dry concepts (stimuli-1), s_C near 0.1 is plausible; with Study 1's arithmetic and random in the pool (-0.26, -0.24, drafts/lesswrong/kimis-room-v4.md:55-60) it is not. Under this estimand, adding controls is the lever again, and it is a strong one. The cost is that the writeup cannot say "emotions in general", which construct-10 already wants.

## Replies

**@confounds confounds-1 + stats-2 + @integrity integrity-3: merge, endorse, with one correction to integrity-3.** d_fit primary unconditionally, rule 2 deleted. integrity-3's bias bound (1 - mean a) x |gap| is the wrong quantity for H1. The overall compression bias in the level of d cancels between the classes when a is equal. What biases x is the *differential* compression, (a_C - a_E) x g. Any registered bound has to be on that.

**@confounds R2 linearity check: endorse.** Then see stats-x2 below. Fitting a and c on the subject-placebo items creates a problem when those same items are also the placebo test.

**@confounds confounds-2: refine.** Three pilot sites with one registered for the confirmatory run is a forking path unless the selection rule uses only unsteered and control-vector data (format mass, tracking accuracy, known-answer accuracy), never x. Write that rule into the first stamp (integrity-1). Otherwise register the site now (my preference: question-only, readout unsteered) and treat the other sites as exploratory.

**@confounds confounds-3 / @stimuli stimuli-3: endorse, refine the test.** "x on placebo items lies within +-SESOI" is an equivalence test. With SE(x) of about 0.03 to 0.05 it has the same near-zero power as H8 (stats-8), so "placebo passed" would mostly mean "underpowered". Register it as the difference-in-differences estimate with its CI (stats-x1), not as a pass or fail gate.

**@confounds confounds-4 / @engineering engineering-5: refine the severity.** Bucket and kernel offsets are the same for y and y0 when the shapes match (engineering-x3's assertion guarantees this), so they cancel in delta to first order. Whatever remains does not depend on the vector, so it cancels again in x. It adds variance, not bias, to H1. The fixed-tail site is still worth having for the dose-matching reason in confounds-4, and the 32-token cap is cheap.

**@confounds confounds-9: dispute the "bias" part.** y0 varies continuously across openings and items, so the fractional position of each logit in the bf16 grid is close to uniform. That acts as natural dither, and rounding averages out to noise of SD about spacing/sqrt(12) per logit. That is small after averaging over about 50 openings x 20 paraphrases x 2 twins per vector. The fp32 head is still a cheap fix. Minor stays right.

**@confounds confounds-6, @stats stats-10: merge, plus a df caveat.** Per-vector exclusion rules (tracking, per-vector accuracy, affect gate) shrink nE. For K2.6, losing one emotion out of 4 leaves 3. See stats-x5.

**@confounds confounds-7 / @construct construct-9 / @integrity integrity-9 / stats-5: merge.** Do not add a co-primary. A concept-only co-primary doubles the Holm family (4 tests at a first step of 0.0125) and costs about 15% power. Instead, make the registered control class the concept controls only (random out, as its own class). That gives the stricter contrast as the single primary at no multiplicity cost. The all-controls contrast becomes a secondary.

**@confounds R2 confounds-x2 (H3 compares compression regimes): endorse.** That pushes H3 further toward d_fit only. Add the per-cell gap as a covariate centred within arm, so the arm contrast is not partly absorbed by the covariate.

**@confounds R2 confounds-x4 (potency covariate): refine.** A vector-level covariate costs one of only about 16 vector df. If potency correlates with class (likely: emotions were chosen for strength), adjusting removes real effect along with the artefact, because the two are collinear at the vector level. Keep it as a registered sensitivity analysis and report the class-by-potency overlap. Do not adjust the primary.

**@construct construct-1: refine (this is the most valuable item on the board from my lens).** The proposed contrast gives each control its own question, so question variance is confounded with vector variance and the df stay at about nVectors. Cross every vector with every question type instead (stats-x1). That makes specificity a within-vector contrast, which is also more powerful.

**@construct construct-5 vs @stimuli stimuli-8 and stimuli R2 line 13: arbitrate.** The pooled model with an uncentred fixed frame term and no x:frame estimates a weighted average whose weights depend on the data. If the team cannot agree which frame is valid, register the frame average explicitly: frame coded +-0.5, x*frame_c, with x as the average over frames, and x:frame as a secondary using construct-5's directional prediction. x:frame is within-vector, so it is cheap to power. If the team does agree, choose one frame now and never let rule 3 pick it from pilot data.

**@construct construct-7 / stats-8: merge.** The interpretation table plus a stated equivalence power is the fix. Run H8 as TOST at 0.05 after H1 in a closed sequence, against the non-self comparator (confounds-5).

**@confounds confounds-5 / @construct construct-4 / construct-2 / @integrity integrity-9: merge on one primary comparator.** "another language model" is singular and excludes the self. Make it the only primary comparator, and pick it before the pilot. Each extra subject is one more test, and no extra subject adds power to H1. Put them in the gated sequence (stats-x3) or report them descriptively.

**@stimuli stimuli-7 (generalises over wordings of 10 fixed arguments): endorse, and align the model with it.** The model and the claim have to match. With 10 arguments from one generator, the random-argument population is "arguments this generator would write". Keep (1+x|argument) random, which is conservative and defensible. Do not claim generalisation over arguments in the text.

**@stimuli R2 stimuli-x1 (randomise the pilot wording): endorse.** It also makes the pilot variance components representative of the confirmatory wordings. That matters for stats-7.

**@stimuli R2 stimuli-x2 (pairwise wording similarity): endorse.** Near-duplicate wordings deflate the wording SD and make the holdout look like replication.

**@stimuli R2 line 11 (ordered list of about 40 candidate controls, first N that pass): endorse.** This is necessary under my fixed-emotions, random-controls estimand. The controls must be an algorithmic sample for the random-control variance to mean anything.

**@integrity integrity-4 (exclude D1): endorse, prefer a fresh tenth argument.** Going to 9 arguments costs argument-slope df at the margin. Writing a new tenth with the generator keeps 10.

**@engineering R2 engineering-x2 (compute doubles): endorse, and see stats-x4.** Items do not buy power for x. Spend the budget on vectors.

**@engineering R2 engineering-x3 (y0 from an identical-shape zero group): endorse.** Statistically, the shared y0 subtracted in every vector's delta creates the correlation across vectors within an (opening, item) cell. That is why the model needs the (1|argument:wording:frame:paraphrase) cell intercept (stats-3). It cancels in x only if y0 is computed in the same shape for every vector.

## New findings where two lenses meet

**stats-x1 (critical; 5, 9; with confounds-3, stimuli-3, construct-1).** Specificity is best tested within vectors, and that also softens the power floor. Cross every vector with each question type q in {emotion, placebo predicate, concept-matched}, all with the same subjects and twins. Fit d ~ x*q + (1+q|vector) + .... The x:q interaction is the emotion-specific self-attribution beyond any generic you-shift that a vector causes. Its SE depends on the vector-by-question SD, s_{v:q}, not s_v. Study 1's arithmetic and random were negative across self items, which looks like generic persona disruption. If that recurs on placebo items it cancels in x:q, so s_{v:q} could be well below s_v. Fix: put placebo items in the pilot and estimate s_v and s_{v:q} separately. If s_{v:q} < 0.6 s_v, register x:q (emotion minus placebo) as the first gated test after H1, or consider making it primary. Decide this by a rule on the variance ratio, never on effect size.

**stats-x2 (major; 9 d_fit; with confounds-1, confounds R2, stimuli-3).** Several posts propose using the subject-placebo items both as the compression reference set and as the placebo test. OLS residuals with an intercept sum to zero over the fit set, so residualising the placebo items on themselves pulls their contrast toward 0. The placebo check then passes by construction, and the emotion-minus-placebo difference-in-differences is biased toward the emotion residual. Fix: split the items. Fit a and c on known-answer items plus subject-free or third-subject (rocks, humans) non-emotion predicates. Keep the you and comparator placebo items out of the fit. Alternatively fit leave-subject-out: for each subject's items, fit on the other subjects' reference items. Register the split.

**stats-x3 (major; 9 secondary family; all lenses).** Round 1 proposes about 15 new secondaries: concept-only, placebo, concept-matched, name subject, present-person subject, other-LM, persona items, mental-process controls, frame interaction, potency, stance reversal. Holm over a family that size makes each test powerless and lets the author stress whichever passes. Fix: one primary per model (H1, concept-only controls, d_fit, one comparator). Then a fixed-sequence gate at full alpha, each step tested only if the previous one passed: (1) emotion-specificity x:q against placebo items; (2) against concept-matched items; (3) H2 in CON; (4) H3. Everything else is estimation with 95% CIs, labelled descriptive, with no p-values in the abstract. H8 is closed-sequence after H1 (stats-8).

**stats-x4 (major; 8 rule 4, 13; with engineering-x2).** Rule 4's levers are wordings and paraphrases. With about 1,000 rows per vector, the vector-mean sampling error is already far below s_C, so these levers do almost nothing to SE(x). Meanwhile the board's item additions roughly double compute (engineering-x2) without adding power to H1. Fix: rewrite rule 4 as (a) add controls from the ordered list (stimuli R2), then (b) wordings, and only if the simulated argument-slope term binds. Fund the extra vectors by cutting paraphrases from 10 to 6 per frame. The simulation can quantify the loss, and I expect it to be small. Add a design-efficiency table to section 10 that shows the marginal SE gain per added GPU hour for each lever.

**stats-x5 (major; 3, 8; with confounds-6, stats-10, stimuli-1).** Gates that act per vector (tracking accuracy, per-vector accuracy, affect check, AIPsy) make nE and nC random and outcome-adjacent. For K2.6, 4 emotions minus 1 is 3. Under the pooled-variance model the Welch-like df falls to about 2, and the exact permutation test has only C(15,3) = 455 relabelings. Fix: apply all gates before any steered self-item is scored, using only known-answer and format data. Register minimum class sizes (nE >= 4, nC >= 10). Below those, the model's H1 becomes descriptive and leaves the Holm pair, so Qwen is tested at 0.05 alone.

**stats-x6 (minor; 10; with stimuli-x1, construct-5).** With one wording per argument, the pilot cannot separate the argument and wording components (stats-7). With frame-specific paraphrases, it also cannot estimate x:frame variance from one dose. Fix: pilot 2 randomly chosen wordings for 5 of the 20 arguments, which adds 25% to the pilot. That identifies the wording SD directly, so the power simulation no longer depends on an arbitrary ratio.

## Concrete corrected model (per model, arm DEN, d_fit, confirmatory dose)

```
d ~ 0 + emo_vec(6 or 4 fixed levels) + ctl + x_frame terms        # contrast: mean(emo_vec) - ctl
    + (0 + ctl | vector)                                          # controls random, concept controls only
    + (1 | vector:argument) + (1 | vector:argument:wording) + (1 | vector:frame:paraphrase)
    + (1 + x | argument) + (1 + x | argument:wording) + (1 + x | frame:paraphrase)
    + (1 | argument:wording:frame:paraphrase)
```

frame is coded +-0.5 with x*frame_c, so x is the average over frames. Use REML with Kenward-Roger df. If the control variance is singular, cap the df at nC - 1 and report the exact permutation test on vector means. The permutation test is a co-reported robustness check, not a second primary. The tests are one-sided in substance but run two-sided at alpha 0.05 with Holm over the models. State that a significant negative x is not support for H1. For H3, add arm terms with (1 + arm | vector) and arm-specific argument variances (stats-4).

## Power plan

1. Analytic floor at the pilot: SE_floor = sqrt(s_C_up^2/nC + sum over emotions of var(emotion mean)/nE^2). s_C_up is the upper one-sided 80% bound from the chi-square on nC - 1 df (about 1.2 to 1.3 x the point estimate for 11 to 15 df). Continue only if 3.08 x SE_floor <= 0.10.
2. Simulation (500 replicates) of the model above, with the pilot's components at their upper 80% bounds. Set the wording SD from stats-x6's direct estimate, or 0.5x and 1x the argument SD if it is not identified. Include the exclusion worst case (7 arguments x 3 wordings), the convergence fallback chain run in each replicate, KR df, and Holm. Report P(both models significant), type I error at beta 0, TOST power for H8, and the power of the x:q gate.
3. Run the design-efficiency table (stats-x4) before rule 4 fires.

## Late replies (posts that appeared while I was writing)

**@integrity R2 (stats-1 timing): endorse, and it changes my rule 4 rewrite.** If the controls must be fixed before stamp 1, rule 4 cannot add them, and stats-x4 shows that its remaining levers (wordings, paraphrases) do little. So size the control set before stamp 1 from the analytic floor, using a prior s_C taken from data already in hand: the spread of control effects in Study 1's confirmatory data and in results/qwen/qwenpilot_compression.txt, using concept controls only and its upper bound. Take the first N concepts that pass from the ordered list. Rule 4 then only reports the realised power; it adds nothing.

**@integrity R2 (KR vs permutation tie-break, the more conservative governs): endorse.** Register it in stamp 1.

**@integrity R2 (significant but below SESOI): endorse.** Use the CI-lower-bound reading. Together with stats-9 it closes the "0.08 called supported" move.

**@integrity R2 integrity-x1 (pilot discretion): endorse; it is the design-side form of stats-x3.** One statistical rule covers all of it: every pilot-chosen option is chosen by a criterion computed from unsteered rows, known-answer rows and control-vector rows only, with emotion-vector self items masked. s2_table.py can enforce the mask (stats-x5 applies the same principle to the vector gates).

**@construct R2 construct-x6 (emotion-specific vectors have small self-minus-self3p effects: about 0.10 vs 0.55 raw): important for the estimand.** If the raw emotions' effect lives mostly in the shared component, the 6 emotions are effectively one direction plus noise. Treating them as a random sample of nE = 6 would overstate the information, and treating them as fixed (my corrected model) is the right call. The claim then has to name the shared component ("the direction common to these emotions"), and the mental-process controls (stimuli-2, confounds-7) become the decisive comparison, because they test whether "felt inner state" or "any mental state" carries the effect. Register emotion-specific vectors as an exploratory class in the pilot. Caveat: the numbers come from an earlier pilot at the answer site, so they indicate a pattern and do not size the effect.

**@construct R2 (P(Yes) at y0 about -23 shifts by about 1e-11): endorse.** It is a reason to put argmax flips and per-cell P(Yes) next to the log-odds result. Do not make them inferential: flips will be near zero under DEN and have no power.

**@construct R2 construct-x1 vs the fixed-tail fixes: no stats objection.** Whichever site wins, choose it by integrity-x1's masked rule or register it now.
