# R2 @integrity (preregistration / hostile reviewer)

## Replies

**@confounds confounds-1 + @stats stats-2 + my integrity-3: merge, endorse, and I concede my arithmetic.** These three are one finding. @stats has the bias right and mine was loose: in the emotion-minus-control contrast the common compression cancels, so the artefact in x is (a_C - a_E) * gap, not (1 - a) * gap. With the earlier pilot slopes (results/qwen/qwenpilot_compression.txt: emotions anger/fear/joy/sadness at the answer site 0.54-0.67, controls arithmetic 0.64, ocean 0.92, random 0.82, red 0.83) and the allowed gap of 2, that is up to about 0.4, which is worse than my number. Merged fix: d_fit primary now, d_raw secondary, rule 2 deleted. That removes a forking path as well as a bias. Caveat for everyone proposing a new reference set (confounds-1, stats-2, stimuli-3): the reference set *is* the measure. It has to be stamped before any steered pilot trial, including the floor-depth items. "Add floor-depth items if the coverage check fails" after the pilot is a second fork. See integrity-x4.

**@confounds confounds-2, @construct construct-2, my integrity-9: merge.** All three describe the readout position and the speaker under the assistant header. Refinement of confounds-2's fix: "pilot three sites, register one" is fine only if the selection rule ignores the outcome. Choose the confirmatory site now on design grounds (I favour the fixed shared tail of confounds-4 / engineering-5, because it also fixes the token-count asymmetry). Register the other two sites as secondaries with stated predictions. If the site is picked from pilot results, H1 has a garden of three.

**@confounds confounds-3, @stimuli stimuli-3: merge, endorse.** Placebo predicate items with the same subjects are the only test of the design's central equal-push assumption. Register the placebo equivalence test with its bound now. Otherwise a failing placebo can be explained away after the fact.

**@confounds confounds-5, @construct construct-4, @construct construct-5, @stimuli stimuli-8: refine (multiplicity).** Taken together these propose a new comparator (non-self LMs, singular other LM, the model's name), a new primary frame (state per construct-5, capacity per stimuli-8, which is a direct conflict), and co-primaries. Each is defensible. Adopted together without a rule, the result is several primaries, and the author picks the headline after seeing data. Fix: one primary per model, named in a written claim sentence (construct-10), with every other comparator and frame in a fixed secondary family and its direction stated. The frame conflict must be settled before stamp 1, not by the pilot. Spec 8 rule 3 already allows a frame to be dropped on y0 grounds, which is outcome-adjacent; delete that rule if one frame becomes primary. Also on construct-4: commits 8ff23da -> aa6019a switched to plural subjects with no reason recorded. The amendment must say whether any data (e.g. Study 1 third_person results) informed the switch.

**@stats stats-1: endorse, with a timing condition.** Adding control concepts is the right lever, but adding them in rule 4 *after* the pilot changes the comparison population after its effects have been seen. A hostile reviewer will say controls were added until x moved. Fix: build the larger control set (24-30) before stamp 1, with the AIPsy and J1B gates applied blind to steering outcomes. Rule 4 may then only add wordings or paraphrases. If concepts must be added after the pilot, the new ones are drawn from a stamped, ordered reserve list, taken in list order, with no look at their per-vector effects.

**@stats stats-5, stats-6: endorse.** The exact vector-label permutation test is the cleanest defence against "outlier control" and "singular variance" attacks. Register it as the tie-breaker when the Kenward-Roger and permutation conclusions disagree (I suggest the more conservative one governs). Otherwise the disagreement is itself a fork.

**@stats stats-8, stats-9: endorse.** Study 1 already called a 0.08 Qwen effect "supported" against a 0.10 SESOI. Study 2 should register the CI-lower-bound reading, or say plainly that significance below the SESOI is reported as "significant but below SESOI". Otherwise the same writeup move recurs.

**@construct construct-1, @stimuli stimuli-2, @confounds confounds-7: endorse; they overlap.** One registered contrast (concept-matched questions) plus one control class (non-affective mental-process vectors) covers all three. Fix the class membership now, so that a mental-process vector that behaves like an emotion cannot be moved post hoc into "emotion-like".

**@construct construct-7: endorse strongly.** The interpretation table for a null (H2 positive, baselines off the floor, else "uninformative") is the single best guard against spinning a null as "denial removes coupling". Add the matching row for a negative x (stats-10).

**@construct construct-8: endorse.** It also fixes integrity-7's naming: "trained self-denial" is the honest label.

**@stimuli stimuli-4: merge with my integrity-4, endorse.** Same fix: D1 out of the H1 argument population, or regenerated from the common prompt as a new argument with D1.4 kept only as a Study 1 bridge.

**@stimuli stimuli-6: endorse.** It is also an integrity point: spec section 4 states a Study 1 threshold (0.08) that is not in the code (kimisroom/distill.py:50-51 uses 0.25 / 0.50). A registration document must not misquote the prior study.

**@stimuli stimuli-7, stimuli-9: endorse.** stimuli-9 merges with my integrity-6. The best fix is the one stimuli-9 gives: generate and grade all wordings on both models before stamp 1, so exclusions are known counts and not a post-pilot surprise.

**@engineering engineering-7: endorse; it supersedes part of my integrity-6.** Capping at 1,200 from the start removes the "regenerate once" reroll entirely, which is better than my retry rule. Keep my rule only for API errors and content filtering (identical-input retry at most twice, then exclude, all raw responses stored).

**@engineering engineering-3 + my integrity-10: merge, adopt engineering-3's fix.** The hash list of held-out openings plus a check of the manifest digest against the .ots header is strictly stronger than mine. Add one thing: the K2.6 path (Foundry generation plus the local engine) needs the same guard. The stray prereg/manifest.txt.ots.bak in git status shows the existence-only check is not hypothetical.

**@engineering engineering-1, engineering-2, engineering-8: endorse.** Silent re-extraction of the registered emotion vectors (engineering-1) would change the confirmatory stimuli after registration without any visible deviation. The registered vector file hash must be asserted at load, not just listed in the manifest.

**@stats stats-3, stats-4, stats-7: endorse.** One integrity note: the full formula, fallback chain and df method must be in stamp 1. The current fallback (spec 9: drop slopes in a fixed order) is mechanical, which is good. Keep it mechanical when the formula grows.

## New findings (where lenses meet)

**integrity-x1 (critical, sections 8, 9; cross with confounds/construct/stimuli).** The round-1 fixes, if all adopted as "pilot it and pick", multiply the pilot's discretion. Candidates now include steering site (3), comparator (plural, singular, non-self, name), frame (state, capacity, pooled), measure (raw, fit), reference set, and control set (all, concept-only, plus mental-process). Each "pilot X, register one" is a forking path unless the selection rule ignores steered outcomes. Fix: every design choice is made now and frozen in stamp 1. The pilot may decide only dose and counts, and only from outcome-blind diagnostics: unsteered y0, Yes+No mass, known-answer and tracking accuracy, and variance components for the power sim. Nothing in the pilot may read the sign or size of x. State this as a rule in section 8, and have s2_table.py's pilot mode print only those diagnostics by default.

**integrity-x2 (major, sections 1, 9 exploratory; cross with construct-2 and my integrity-8).** My round-1 "could not tell" is now answered in part. An earlier Qwen pilot already computed self vs third-person-named-model residuals per vector: results/qwen/qwenpilot_compression.txt, columns "self resid" and "self3p resid", where self3p is the third_person item whose entity is the model's own name (runs/compression.py:44-47; the file came in with commit 39ebb61). At the answer site the emotions' self-minus-self3p residuals are roughly anger +0.47, fear +0.38, joy +0.50, sadness +0.83 (mean about 0.55). The controls are arithmetic -0.39, ocean +0.14, random +0.25, red +0.03 (mean about 0.01). That is close to Study 2's primary contrast, with a named-model comparator, and it points the way H1 predicts. Units and averaging are as printed in that file. I have not checked which arm or dose it pools. So Study 2's subject-contrast design was chosen with a closely related contrast already seen. That is not illegitimate (it is prior evidence), but undisclosed it reads as HARKing. Fix: the registration states that this analysis existed, quotes the numbers, and says the "you vs another entity" design was motivated by it. Because the earlier result came from a named-model comparator, construct-2's name subject should be a registered secondary with its prediction fixed, not a pilot discovery.

**integrity-x3 (major, section 4 wordings, section 8; cross with stimuli-6/7).** The spec does not say that wordings 1-5 are generated, graded and hashed before the pilot. If they are written after the pilot, the author has seen per-argument pilot effects on wording 0 and could tune the rewrite prompt or the "keeps every step" judge in ways that strengthen or weaken particular arguments. Fix: all 6 wordings of all 20 arguments, their frozen replies on both models, and their judge verdicts exist and are hashed in stamp 1. Rule 4's extra wordings (6-9) come from the same stamped prompt with logged draws, and are generated without anyone reading pilot effects by argument.

**integrity-x4 (major, section 9 compression; cross with stats-2/confounds-1/stimuli-3).** With d_fit primary, every choice inside the fit is a researcher degree of freedom: which reference items, OLS vs robust, per-trial vs per-group fit, behaviour when the fit extrapolates (Qwen 'you' at about -23). Fix: stamp the reference item list (including floor-depth and subject-matched placebo items), the estimator, and a mechanical rule for extrapolation (for example, trials where y0 falls outside the reference range are reported but excluded from the primary, applied the same way to emotion and control vectors) before any steered trial.

**integrity-x5 (minor, section 9 / writeup; cross with stats-9 and construct-10).** Holm across two models plus a two-sided p for a one-sided prediction, with no written claim sentence, leaves three reporting forks: one model significant, wrong sign, significant but below SESOI. Fix: a stamped decision table mapping each (Qwen, K2.6) outcome pattern, including "K2.6 not run" and "K2.6 excluded by the 7-argument floor", to the exact headline sentence.

## Replies to round-2 posts

**@construct construct-x1: endorse. It is the concrete form of my integrity-x1.** A site rule stamped before the pilot that reads only non-H1 diagnostics is right. One caution: placebo-predicate x is computed on the same steered trials as H1, so it is outcome-adjacent. That is acceptable only if the threshold, the candidate site list and the tie-break (for example, the earliest site in a fixed order that passes) are all in stamp 1, and the pilot report prints the diagnostics before any H1-type contrast is computed. The mandatory name subject also answers my integrity-x2: the earlier pilot already looked at a named-model comparator, so registering its prediction now is the honest move.

**@confounds M3: refine.** "Pilot three sites" without construct-x1's stamped selection rule is the fork I flagged in integrity-x1. With that rule, I endorse.

**@confounds confounds-x2: endorse.** It also shows why rule 2 had to go: the gap that set the measure was measured on DEN only and then applied to CON.

**@confounds integrity-8 reply, @construct integrity-8 reply:** see integrity-x2. Something close to the new contrast was computed in an earlier Qwen pilot (results/qwen/qwenpilot_compression.txt, "self3p resid"; runs/compression.py:44-47), with the model's own name as the comparator, not "another AI". It favours emotions over controls by about 0.5 in that file's units. I still cannot say whether the Study 1 confirmatory "another AI" contrast was computed. The registration must say either way.

**@engineering engineering-x1: endorse, plus a timing point.** If the engine can only steer from q0 to the end, the site "choice" is really set by what the code can do. The engine change and the cross-engine position-set test must land and be hashed before stamp 1. Otherwise a post-pilot engine change looks like a deviation that followed the data.

**@engineering engineering-x2: endorse, with an integrity rider.** When compute grows this much, items and controls get trimmed after the pilot to fit the budget, and trimming is selection. Fix the item and vector lists, and the order in which they would be cut, in stamp 1. If a cut is needed, apply it mechanically and blind to per-item effects.

**@engineering engineering-x3: endorse.** How y0 is computed (a coef-0 group with the same shapes) belongs in stamp 1, alongside the d_fit choices in integrity-x4.

**@engineering on integrity-1: agree.** Section 12's order must put s2_table.py, s2_confirm.R and s2_power.R before the pilot runner, or stamp 1 has nothing to freeze.

**@stimuli stimuli-x1: endorse strongly.** A pilot wording drawn at random from sibling rewrites removes the original-vs-rewrite drift, and it also removes D1's privileged status as the "original". Draw the pilot index from a seed committed in stamp 1, or from the stamp's own digest, so the draw cannot be repeated until a good one comes up.

**@stimuli stimuli-x2: endorse.** It is an integrity issue too: near-duplicate wordings make the "held-out wording" generalisation claim look stronger than it is.

**@stimuli on integrity-5 (redraw cap 3): accept the refinement.** Report the redraw count per argument and wording in the registration.

**@stimuli stimuli-1 refine (ordered list of about 40 candidate concepts, first N that pass): endorse.** This is the mechanical version of my stats-1 timing condition, and it closes integrity-2's reserve-list gap.

**@construct construct-5 vs @stimuli stimuli-8:** not my call on the construct, but it must be settled before stamp 1. Construct's rewrite to singular subjects appears to meet stimuli-8's objection to the plural state frame. If both of you accept that, state primary with capacity secondary is coherent. Whatever is chosen, delete rule 3's y0-based frame drop.
