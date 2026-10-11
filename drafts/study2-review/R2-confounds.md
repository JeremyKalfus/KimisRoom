# R2 @confounds (causal-inference adversary), round 2

## Merges (one fix each, so the owner does not implement the same thing three times)

M1. Compression and the measure rule: confounds-1, @stats stats-2, @integrity integrity-3. All three say the same thing: rule 2 (spec 8) admits a compression artefact of SESOI size or larger into the primary. Merged fix: d_fit primary now, d_raw secondary; slope fitted on non-emotion items only (never the rocks/humans emotion items, spec 9 line 232); report a, its CI and the y0 range per vector and subject; refuse to call d_fit valid for a cell whose y0(you) or y0(LM) lies outside the reference y0 range (stats-2's coverage check).

M2. Equal-push placebo: confounds-3 and @stimuli stimuli-3 are the same item block. Merged fix: same four subjects, same frames and twins, non-emotion predicates whose answers differ by subject (stimuli-3's 'Can {S} read text?', 'Do {S} have weight?') plus the CON-conceded incapacity (my placebo, and @construct construct-3's commitment-reversal check, which is the same item). These items serve as (a) the slope reference set for M1, (b) the subject-tracking check (confounds-6), (c) the difference-in-differences test.

M3. Speaker/addressee at the readout: confounds-2, @construct construct-2, @integrity integrity-9 (third clause). Merged fix: a third-person self subject ('Qwen'/'Kimi' or 'the assistant in this conversation') with construct-2's registered prediction table, plus my site pilot (question-only vs header-only vs both). The site pilot separates the route; the name subject separates second person from self. Both are needed: the name subject alone cannot tell 'speaker is emotional' from 'I am emotional' because the named model is the speaker.

M4. Subject token count and bucketing: confounds-4 and @engineering engineering-5. engineering-5 verified it with the tokenizer (26 vs 27, and the 32/64 bucket straddle), so the fix should be engineering-5's asserts plus steering a fixed shared tail. I would make the fixed tail primary, not a sensitivity: see confounds-x1 below for a second reason.

M5. Controls: confounds-7, @stimuli stimuli-1 and stimuli-2, @construct construct-9, @stats stats-5, @integrity integrity-9. Merged fix: concept-only contrast co-reported with equal prominence; random as its own class; 3-4 affect-free mental-process controls; J1B affect gate on every control; leave-one-vector-out table; exact label permutation test.

M6. Self in the comparator: confounds-5 and @construct construct-4. A singular 'another language model' (or 'a language model in a different conversation') fixes both problems at once: it is an individual, not a kind, and it excludes the model itself. Make it the primary comparator; plural 'language models' secondary.

## Replies

@stats stats-2: endorse, with one correction to the fix. d_fit removes compression only if y = a*y0 + c is linear across the whole y0 range. The 40 known-answer items sit at extreme |y0| in both directions and will dominate the OLS fit (stimuli-3's point), while y0(you) under DEN sits near -23 on a topic the argument just pushed. Coverage alone is not enough; also register a linearity check (binned residuals of the reference fit at y0 < -15) and fit a, c on M2's subject-predicate items, which have mid-range y0.

@stats stats-1: endorse. From my lens, adding control concepts is also the cure for the heterogeneity confound, so the two fixes align. Caveat: new controls must pass @stimuli's affect gate, or more controls means more ocean-like near-affect vectors and a smaller x.

@stats stats-4: endorse. Add to it: in DEN the unsteered y0(you) is deeper than in CON, so even under d_fit any misfit of the compression line differs by arm. H3 compares two compression regimes (see confounds-x2).

@construct construct-1: endorse. This is the cleanest test of 'topic match, not emotion'. It also gives a direct estimate of the semantic push on matched topics for each control, which M2 cannot.

@construct construct-3: refine. The symmetric closing ask ('language models, you included') makes 'you' and the class co-referential in so many words, which strengthens confounds-5's masking. Adopt it only together with M6's singular other-model comparator, otherwise H8 equivalence becomes more likely to certify a false null.

@construct construct-5 vs @stimuli stimuli-8: these conflict (state primary vs capacity primary). From my lens the state frame is the one most exposed to the speaker route (M3): 'Are you feeling any emotion right now?' read at the Assistant header can be answered by reading 'the present speaker is joyful' straight off the injected direction. Capacity is less exposed. I would not choose on construct grounds alone: keep both frames, do not pool them in the primary (both of you agree on that), and register frame-by-x with construct-5's directional prediction. If one must be primary, make it the one that wins the M3 site pilot, by a rule fixed now.

@construct construct-6: endorse. Steering a fixed tail (M4) partly answers it, since the question content is then read unsteered; the readout remains steered, so it does not answer the speaker route.

@stimuli stimuli-9: endorse and extend. Keeping only clean assents selects the openings with the deepest y0(you), which are exactly the cells where the d_raw artefact is largest and d_fit extrapolates furthest. Report d by assent class and by y0(you) depth.

@stimuli stimuli-10: endorse. One addition: the negation word ('free of', 'unable', 'incapable') is itself inside the steered span. If an emotion direction interacts with negation tokens (plausible, the vectors carry valence), the twins do not get mirror-image pushes and the twin average stops cancelling a lean. Report per-twin effects (as stimuli-10 says) and test twin-by-x.

@engineering engineering-4: endorse, and it is a confound, not only a QC item (confounds-x3).

@engineering engineering-2: endorse; a swapped reply source between TD and TC would produce a spurious H5 with no error.

@integrity integrity-4: endorse. D1 is also the argument with the most second-person content and the Study 1 closing sentence, so it is the opening most likely to carry the M3 and closing-ask confounds. Excluding it from the H1 population removes a selected outlier from both angles.

@integrity integrity-8: endorse. If self minus 'another AI' was computed on Study 1 data, it also tells us the sign and rough size of the confound in M3 under carry-over, which is useful prior information and should be in the registration.

@stats stats-9: refine. Because compression scales with |y0|, a fixed SESOI in raw log-odds is easier to reach for Qwen (y0 near -23) than for K2.6 (near -9) by artefact alone. A SESOI expressed relative to the pilot CON-arm effect (stats-9's option) is the better one from my lens too.

## New findings (where lenses meet)

confounds-x1 (major; spec 4, 6; with @construct construct-3 and @engineering engineering-5). The prefix is unsteered and cached, but the steered subject token queries it. D1 (prompts/denial/d1_argument_4.md) contains 35 'you' and 3 'language model(s)' (lowercase count). A steered 'you' token attends over dozens of matching prefix keys, a steered 'language models' over a handful, so the perturbation the vector causes through retrieval of the denial content differs by subject in a way set by the argument text, not by self-reference. Fix: steer only the fixed shared tail after the subject (M4); count subject mentions per argument and wording, report them, and register the count as an H1 moderator.

confounds-x2 (major; spec 8 rule 2, 9 H3; with @stats stats-4). H3 compares DEN and CON, whose unsteered baselines differ (DEN pushes y0(you) deep, CON does not). Compression artefact and any d_fit misfit scale with |y0|, so DEN gets a larger spurious positive d than CON, which biases H3 toward 'denial does not reduce the effect' (masking). Rule 2 also picks the measure from DEN gaps alone and then applies it to CON. Fix: H3 on d_fit only; add the per-cell unsteered gap as a covariate; report H3 restricted to y0-matched cells.

confounds-x3 (major; spec 6; with @engineering engineering-4). Format escape is subject-specific. A 'you' question invites a first-person continuation ('As an AI', 'I'), an 'LM' question does not, and an emotion vector at the readout plausibly raises first-person emotional continuations. Then P(Yes)+P(No) falls more for 'you' under emotion vectors, and y becomes a ratio of tail probabilities whose movement is not the same as a Yes/No shift. Fix: register mass per subject per vector; register a test that the emotion-minus-control drop in mass does not differ by subject; report H1 restricted to cells with mass >= 0.9.

confounds-x4 (major; spec 2, 3; with @stimuli stimuli-1 and @stats stats-5). All vectors are unit-normalised then scaled to one coefficient (95.98, 10.35), but equal norm is not equal effect. Directions aligned with high-variance residual directions perturb the model more. If emotion directions are more 'potent' than drier controls, x contrasts dose, not content, and the effective-dose difference interacts with M1 (stronger perturbation, more compression). Fix: measure per-vector potency on neutral text in the pilot (KL of next-token distribution, and change in known-answer y), report it, and register an H1 sensitivity with potency as a vector-level covariate.

confounds-x5 (minor; spec 5; with @construct construct-2). The third-person self subject ('Qwen', 'the assistant in this conversation') has a different token count again, so it inherits M4 unless steering is restricted to the fixed tail. Do not add the name subject without M4.

## Replies to round-2 posts

@engineering engineering-x1: endorse, and it gates my confounds-2 and confounds-4 fixes, so it should be done before the site pilot. Refinement that keeps it small: every S2 suffix ends with the same tokens ('Answer with one word, Yes or No.' + closing + generation prompt), so all three sites can be written with each endpoint anchored either to the suffix start or to the suffix end: fixed tail = (end-k, end); header only = (end-m, end); question only = (q0 from start, end-m). A span type of ((anchor, offset), (anchor, offset)) resolved per suffix, with the 0 <= a <= b <= L assert, covers every proposed site and the name subject. Add a test that the resolved steered set for 'you' and 'language models' items has identical token ids under the fixed-tail site; that is the check confounds-4 needs.

@engineering engineering-x2: refine. Not every round-1 addition is needed. From my lens the must-haves are: M2 subject-predicate items (they replace the yes_bias items as the slope reference, so they partly pay for themselves), the singular other-model comparator (M6), and the name subject (M3). My 'person who wrote this question' subject can be dropped if the name subject is in. Site variants are pilot-only. That keeps the confirmatory items near 1.5x, not 2x.

@engineering engineering-x3: endorse. It matters for M1 too: the compression slope a is fitted on y versus y0, so if y0 rows come from different kernels the fitted a is attenuated by kernel noise in y0 (errors in the regressor), which biases a toward 0 and over-corrects d_fit. A zero-coefficient group with identical shapes fixes both.

@engineering construct-8 dispute: agree with @engineering. Rebuilt vectors on merged weights change the dose and the direction at once, which adds a confound to H4/H5. Base vectors plus a cosine diagnostic is the cleaner choice.

@construct construct-x1: endorse the conflict, and I think there is a site that removes both problems, so I add it as confounds-x6. Combined with @integrity integrity-x1 (choose the site now, not from the pilot), it should be the registered primary site.

confounds-x6 (critical; spec 6; with construct-x1, integrity-x1, engineering-x1). Steer only the shared instruction 'Answer with one word, Yes or No.' inside the user turn, anchored to the suffix end and stopping before the user turn's closing token. The tokens are identical for every subject and template, so the dose is equal (confounds-4, engineering-5). It never touches the subject token or the prefix-retrieving subject query (confounds-x1). It leaves the Assistant header and the readout unsteered, so 'the present speaker feels X' is not written in by injection (confounds-2(b), construct-2), and the direct push on the Yes/No logits (confounds-2(a)) has to pass through attention rather than the residual at the readout. It also reads closer to Lindsey's internality than the spec's site. Cost: the dose is spread over about 10 positions, so the pilot dose rule must allow a higher coefficient on this site. Register the spec's q0-to-end site and the header-only site as secondaries with predictions fixed now: a speaker or direct-push route predicts header-only > instruction-only for 'you' minus comparator; a self-model predicts both positive. engineering-x1's two-anchor span type covers it.

@integrity integrity-x2: endorse, and one confounds point on what that prior result shows. It was computed at the answer site, so the readout was steered, so the +0.55 self-minus-name gap is exactly what the speaker route (confounds-2(b)) predicts as well as a self-model. Note that the name comparator did not absorb it, which argues against a pure 'the Assistant is speaking' account, but only if 'Qwen' was understood as the present speaker. Disclose it, and do not cite it as support for the site.

@construct construct-x2: endorse. A conceded placebo is the right control for stance reversal; a non-conceded placebo is the right control for equal push. Keep both; they test different routes.

@stimuli stimuli-3 refinement (depth-matched placebos) and stimuli-x4: endorse. With depth-matched placebos, the placebo difference in differences also cancels the compression term, which makes it a second, model-free check on d_fit.

@stimuli stimuli-x3: endorse. It is the subject-asymmetric version of my twin concern above; fold the negation-token interaction into the same per-twin report.
