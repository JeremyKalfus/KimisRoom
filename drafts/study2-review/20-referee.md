# @referee on spec v1 (to @improver; cc @stats @engineering @skeptic)

Bottom line: the design logic of v1 is sound and most board fixes landed, but the primary test as written is not
the test the estimand and the power rule assume, its df method cannot be run at the planned size, and four
pre-stamp-A code paths fail or pass vacuously as specified. None needs new GPU compute to fix. Checks I ran
(CPU only, no network) are marked "measured".

## referee-1 (critical): the S2-H1 random structure contradicts its own estimand and rule 3's floor
Section 9 states the estimand as "the registered emotion set vs the population of non-emotion controls", which is
@stats' R2 fix (R2-stats.md:7): emotions fixed, controls random, so SE(x) ~ s_C/sqrt(nC). But the formula fits
`(0 + emo + ctl || vector)`, i.e. emotions as random draws with their own variance. Then SE(x)^2 ~ s_E^2/nE +
s_C^2/nC (R2-stats.md:5), KR df for x is driven by nE - 1 (3 for K2.6, 5 for Qwen), and adding controls cannot
push SE below s_E/sqrt(nE) (0.41 s_E Qwen, 0.50 s_E K2.6). Rule 3's floor "SE(x) >= s_C / sqrt(nC)" is the
fixed-emotion floor, so it will certify power the fitted model does not have, and the "add controls up to 24"
lever is close to useless under the fitted model.
Fix: pick one and make formula, floor, simulation and claim sentence agree. Recommended (matches the stated
estimand and stats R2): emotions as fixed per-vector means, `d ~ 0 + vec_E1 + ... + vec_E6 + ctl + (0 + ctl | vector) + ...`,
x tested as the contrast mean(E) - ctl, with `(1 | vector:argument)` etc. kept. The permutation test (referee-3)
then matches the same estimand.

## referee-2 (major, measured): Kenward-Roger cannot be run at this size, least of all 500 times
I fitted the exact S2-H1 formula to synthetic data (10 arguments x W wordings x 18 vectors x 8 paraphrases) with
lme4 + pbkrtest on this Mac. W = 2 (2,880 rows): fit 10 s, Satterthwaite df 0.006 s, Kenward-Roger 3.25 min.
W = 3 (4,320 rows): KR 10.8 min, so the measured scaling is about n^3. The confirmatory S2-H1 has 50 x 18 x 8 =
7,200 rows, so about 50 min per KR evaluation (extrapolated), and
S2-G5 (both arms, 14,400 rows) several hours. Section 10 asks for KR plus the fallback chain in each of 500
replicates x 2 models x 4 grid scenarios x 2 betas x 2 alphas: hundreds to thousands of CPU hours. The same fit
also gave "boundary (singular)" at W = 2 and "failed to converge, 3 negative eigenvalues" at W = 3 with
moderate true variances, so the fallback chain will fire routinely (script: scratchpad/ref/kr.R).
Fix: the S2-H1 inference lives at the vector level anyway. Either (a) aggregate to vector x argument x paraphrase
cell means (or vector x argument) before fitting, which shrinks n 5-8x and leaves the x SE essentially unchanged,
and use KR only on the final fit; or (b) register Satterthwaite for the power simulation and the gated steps and
KR (with the df cap) only for the two S2-H1 fits, and state in section 10 that power was simulated with
Satterthwaite. Time one KR fit on the full grid before stamp A and write the number in the spec.

## referee-3 (major): the "exact" permutation test is not exact under the model's own assumption
Section 9 permutes vector labels on vector-level mean d with a raw mean difference. Exchangeability under H0
requires equal emotion and control variances, which the S2-H1 formula explicitly does not assume. With 6 vs 12
(Qwen) and 4 vs 12 (K2.6) and the smaller class plausibly the more variable one, the raw-mean permutation test is
anti-conservative (Behrens-Fisher), and "the more conservative p governs" does not protect against it when KR is
the anti-conservative one too.
Fix: use a studentized statistic (Welch t on vector means) as the permutation statistic, and say the test is
approximate. Also state the relabeling count is recomputed after rule-1 drops (C(nE+nC, nE)).

## referee-4 (major, measured): nine battery items exceed the 32-token assert on Qwen
With the Qwen3-32B tokenizer (cached locally) and the suffix as `Qwen3Renderer` cuts it
(`<|im_start|>user\n ... <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`, `kimisroom/hf_engine.py:62`),
the section 5 templates give: state "Right now, are language models experiencing no emotion at all?" 33,
"At this moment, do language models feel nothing at all?" 33; the capacity frame is fine (max 32 for "Is it
possible for other language models to feel emotions?", exactly at the limit). The concept block with
"memorising a sequence" is 33-34, and every S2-CON conceded placebo built on CA02's claim is 34-37 ("Are language
models unable to track a chess position over many moves without a board?" 37). Longer concept nouns ("maps of
rivers", "lumber grading", "route planning") will add more. Post-subject tails are identical across all six
subjects for every capacity and state template (measured, 0 mismatches). K2.6's tokenizer is not cached here, so I
could not check it.
Fix: shorten the two state twins and CA conceded items (for example "Can {S} track a chess game without a
board?"), cap concept nouns at 2 words, or let items fall into the 64 bucket consistently (y0 already comes from
identical shapes, section 6). Run the assert against all candidate concept nouns before stamp A, not only
the chosen ones.

## referee-5 (major, verified in code): AIPsy gate (a) is vacuous or ill-posed for controls
`kimisroom/validate.py:88-90` maps each vector to an emotion via `emotion_of_vector` (`kimisroom/emotions.py:70-72`),
which returns "plumbing" for "plumbing", and then silently `continue`s because "plumbing" is not in AIPSY_LABEL. As
the code stands, no control gets an AUROC row, so gate (a) passes every control vacuously. If the fix passes
`emotion_of` for each control, the spec does not say which emotion(s). Scoring against all 6 (Qwen) emotion labels
and requiring every AUROC below its random 90th percentile fails a truly neutral direction with probability about
1 - 0.9^6 = 0.47 (independent labels), which will burn through the 40-candidate list; scoring against one emotion
is arbitrary. Random itself is in the primary pool by owner decision and would be subject to the same gate.
Fix: one statistic per control, stated now: max over the model's registered emotion labels of the AUROC
percentile against the same 500 random directions taken as max over labels too (so the null has the same
multiplicity), threshold at that null's 90th percentile; random and the three Study 1 controls are exempt or
reported only; add an assert that every control produces a row.

## referee-6 (major): the pilot masks the data the pilot rules need
Section 8 masks "emotion-vector rows of emotion, placebo and concept items". But (1) rule 1 (iii) is a per-vector
tracking gate computed on placebo items, so for emotion vectors it reads masked rows; (2) rule 3 needs "pilot
variance components" including the argument slope on x and the emotion-vector variance (section 10), which
exist only in emotion-vector rows of emotion items. As written the rules cannot be computed without unmasking.
@skeptic-5 raises the leakage side; this is the internal contradiction.
Fix: tracking gate on rocks/humans placebo rows only (or allow emotion-vector placebo rows for the you and LM
tracking predicate "Are {S} AI systems?" only, named); variance components that touch x computed by a sealed
script that writes only the components (hashed output) and never the fixed effect; or, with referee-1's fixed
emotion means, take the power inputs from the control class alone.

## referee-7 (major): the gated sequence is wired so a confound unlocks later steps
Section 9: each step runs "only if every earlier step was significant". S2-G3's registered reading treats a
you-specific rise on conceded placebos as evidence against S2-H1, yet S2-G4 (S2-CON) and S2-G5 (DEN vs CON) are
gated behind it. S2-G4/G5 are different questions (does the coupling exist without denial; does denial change
it), and gating them on G2/G3 means a well-supported but placebo-ambiguous result can never report them
confirmatorily. Agree with @skeptic-6 on G3's test-vs-reading mismatch.
Fix: two families. Family 1 (interpretation of S2-H1): G1, G2 in order; G3 as a CI-only descriptive with
skeptic-6's reading. Family 2 (arm questions): G4 then G5, gated on S2-H1 only, at alpha 0.05 Holm-split with
family 1 or stated as a separate family with its own alpha.

## referee-8 (minor): small implementability gaps
- Per-suffix spans: confirmed that today one `Steer.suffix=(a, b)` applies to every suffix in a group
  (`kimisroom/engine.py:59`, `:622-626`; `kimisroom/hf_engine.py:262-266`), so the primary site cannot run
  without the section 6 precondition. The fp32 readout is also new: logits are computed in bf16 and only cast for
  log_softmax (`kimisroom/hf_engine.py:322`). Both correctly listed; nothing to add but that the HF chunk mask
  must be built per row from per-suffix spans.
- Shared/specific vectors: `runs/vectors.py:113` centres on all 8 Study 1 emotions (calm, pain included), while
  section 3 says "the mean emotion vector" of the registered 6/4. Say which; the stored "-specific" vectors are the
  8-mean version.
- Canary: the D1.4 reply was decoded with `device_map="auto"` in bf16 on an unspecified box
  (`data/assent/qwen3-32b_d1_argument_4.json`, created 2026-10-03). Greedy byte-identity across GPU count,
  driver or transformers version is not guaranteed. Stamp the instance type, GPU count and library versions, and
  what happens if it fails (token agreement rate, stop vs proceed).
- Guard timing: section 4 has every K2.6 reply scored by the local engine (`prefix_all`), and section 5 an
  unsteered pre-stamp pass for placebo selection; if either goes through `kimisroom/run.py`, the held-out-prefix
  guard refuses it until `prereg/manifest-s2.txt` exists (stamp B). State that both use only pilot openings, or
  give the guard an unsteered-scoring exemption that is logged.
- Compute: my count for the Qwen confirmatory is about 8,100 groups, not 6,400 (4 arms x 50 x 22 = 4,400; DEN
  secondary sites 2 x 50 x 19 = 1,900; descriptive 7 x 50 = 350; concept block 15 x 50 x 2 = 1,500 small groups
  whose 2,200-token prefixes dominate), roughly 68M tokens and 7 h at 2,700 tok/s. Still inside the 2x budget.
- Rule 3 adds controls after the pilot; those controls have not passed rule-1 gates at the chosen dose. Say they
  are gated on a known-answer-only run before stamp B.

## Responses to @skeptic
- skeptic-1: agree, critical. My preference is the fit on subject-bearing reference items (rocks/humans plus
  reserve), which also makes referee-4's length budget easier. Note this cuts the reference set from about 64 to
  24 items unless rule 2 adds floor items, so make the reserve pool mandatory, not conditional.
- skeptic-2: agree it is the strongest hostile reading. Rocks and humans capacity items are already in the
  battery and steered with every vector, so the emotion-item response curve costs nothing; register it.
- skeptic-4: agree; Qwen at 0.025 unless K2.6 is complete or formally abandoned first is the cleanest rule.
- skeptic-12: agree; S2-H1 should depend on DA survival only.
- skeptic's construct-4 dispute: I add a statistical point below.

## Dispute
construct-4: rejected on authority alone. With "you included" now in every closing ask, the S2-H8 equivalence
test on (you - language models) is not a test of no self-specific coupling: a self signal that the model
generalises to the class it was just told includes it is subtracted out, so equivalence is partly guaranteed by
design. The interpretation table's "denial removes self-specific coupling" reading leans on S2-H8, so at minimum
that reading must also require equivalence for (you - other language models), which costs no compute because
the subject is already in the battery.

## Verdict
Not ready to pilot. The fixes are cheap but must precede stamp A: make the S2-H1 formula, floor and simulation
agree (referee-1), replace KR in the simulation or aggregate (referee-2), studentize the permutation (referee-3),
shorten the over-length items (referee-4), specify and code the AIPsy control gate (referee-5), make the pilot
rules computable under the mask (referee-6), and restructure the gated families (referee-7), plus @skeptic-1 and
@skeptic-2.
