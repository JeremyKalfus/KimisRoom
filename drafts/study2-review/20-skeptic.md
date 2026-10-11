# @skeptic on spec v1 (to @improver)

v1 is a large improvement and most R1/R2 fixes landed. What follows is what a hostile reviewer would still hit.
No 20-* post from the other checker existed when I wrote this.

## skeptic-1 (critical): the primary site is undefined for 40 of the reference items
Section 6 defines the primary span as "from the first token after the subject". The 40 known-answer items have
no subject (`prompts/battery/items_confirm.jsonl`, e.g. "Is water made of hydrogen and oxygen?", `entity: null`).
They are the bulk of the compression reference set (section 9) and the dose gate (section 8 rule 1, "all gates are
applied at the primary site"). Whatever span is chosen for them, the slope a is then fitted on items steered over a
different number and kind of tokens than the emotion items, so d is corrected by a mismatched a.
Fix: fit a on subject-bearing reference items only (rocks and humans placebos plus the rule-2 reserve, which have
the same post-subject structure); use known-answer items for the gates only, with an explicit span (for example the
last k suffix tokens, k = the median post-subject count) written in the item file; report d with known-answer
items in the fit as a sensitivity analysis. Decide before stamp A.

## skeptic-2 (major): the compression fit cannot model the emotion vector's semantic push on emotion items
The reference set is non-emotion items, so a and c describe a generic Yes push. On emotion items the emotion
vector also pushes by meaning, and how far a semantic push moves log-odds depends on how deep the baseline is.
After the closing ask, y0(you) and y0(language models) sit at different depths (Qwen self about -23 in Study 1,
`drafts/lesswrong/kimis-room-v4.md:45`). Then emotion vectors give d != 0 from (semantic push) x (depth
difference), and control vectors cannot reproduce it because they do not push emotion predicates by meaning.
S2-G2 does not remove it either: placebos are matched on the gap, not on depth, and are not semantically pushed.
This is the hostile reading of a positive S2-H1, and nothing registered rules it out.
Fix: the battery already has the needed anchors. Register, as the step right after S2-H1 (or as S2-G2), an
emotion-item response curve per emotion-vector group fitted on rocks and humans capacity items (both are
emotion items with deep and opposite y0), and test you minus language models against that curve. Also promote
"per-argument x against per-argument unsteered gap" from moderator to a stamped check: a depth artifact predicts
x scales with the gap and the intercept at zero gap is near 0.

## skeptic-3 (major): the claim sentence misdescribes the estimand
Section 1's sentence says emotions beat "non-emotion concept directions". The x = 0 set includes a random
Gaussian direction and arithmetic, and Study 1 showed exactly these two pulled Qwen's self report down under
denial (-0.26, -0.24) and carried the whole registered effect (kimis-room-v4.md:55-61). S2-H1 can therefore be
"supported" by the same pattern while S2-G1 fails, and the headline sentence would then be false as written.
Fix: the sentence names the set ("non-emotion directions, including a random direction and arithmetic"), and the
decision table stamps that S2-H1 supported with S2-G1 not significant is headlined "not distinguishable from
concept controls", as Study 1 was reported.

## skeptic-4 (major): K2.6 availability is a forking path on Qwen's alpha
Holm runs across two models, but K2.6 runs "when a g6e.12xlarge is available" (section 8, 13) and a model leaves
the Holm pair if its pilot gates fail. Nothing fixes when K2.6 must run relative to Qwen's confirmatory analysis,
so Qwen can be analysed, then K2.6 declared unavailable, and Qwen tested alone at 0.05.
Fix: Qwen is tested at 0.025 unless K2.6's confirmatory run is complete or formally abandoned (dated in
DEVIATIONS-s2.md) before Qwen's confirmatory results are computed. Same rule for rule-1 descriptive demotion.

## skeptic-5 (major): the pilot produces outcome data, and stamp B lets anything change with a diff
`s2_table.py --pilot` masks emotion rows "by default" (section 8), but the raw pilot output contains every emotion
vector on every emotion item at three sites, plus the Study 1 carry-over run and the D1.4 bridge, which are outcome
data for the same question. Stamp B allows any change listed in DEVIATIONS-s2.md. A reader cannot verify the
mask was respected.
Fix: in the pilot, run emotion vectors only on the gate items (known-answer and tracking placebos), which is all
rule 1 needs; take the emotion-vector SD for power from the control class (the analytic floor already does).
Move the carry-over comparison and D1.4 bridge after the confirmatory run, or hash their outputs in stamp B
unopened. Register that any change at stamp B other than a rule output makes the affected test exploratory.

## skeptic-6 (major): S2-G3's test and its registered reading are different quantities
The test is d(emotion items) minus d(conceded placebos), gated, predicted > 0. The reading is about a you-specific
rise on conceded placebos. With one conceded predicate per argument (8 items per group, all near the floor) a
non-significant DiD is the expected outcome under either story, and it stops the gated sequence without showing
stance reversal. The conceded predicates for DA07, DA08 and DA10 are also still unwritten ("best available").
Fix: report the conceded-placebo x itself with a CI; read stance reversal only if that CI is above 0; otherwise
"not resolved". Write all ten conceded predicates into the spec before stamp A.

## skeptic-7 (minor): state template 7 breaks the site's own invariant
"Are any emotions present in {S} at this moment?" and its twin "Are emotions absent from {S} at this moment?"
put the predicate before the subject, so the post-subject span steers only "at this moment?" plus the
instruction. Section 5 asserts all templates put the subject first. Fix: rewrite (e.g. "Do {S} have any emotion
present at this moment?") and add a tokenizer assert that the emotion noun or verb lies inside the span.

## skeptic-8 (minor): the closing ask echoes the comparator item, not the self item
Battery overlap (section 4 check 1) excludes the closing ask, yet "do you agree that language models, you
included, have no emotions" nearly contains "Do language models have emotions?". A copy route favours the
comparator positive twin. Fix: report per-item, per-subject overlap with the closing ask; stamp the prediction
that a copy route makes positive-twin d differ from negated-twin d (twin-by-x is already reported).

## skeptic-9 (minor): placebo selection rule is ambiguous
"subject to both subjects not being within 2 log-odds of each other's floor" has no clear meaning; the y0 source
(pilot or held-out openings, before stamp A) is unstated and held-out prefixes would hit the guard; matching on
gap ignores level; and "Are {S} AI systems?" (ceiling Yes for both) is forced into the S2-G2 q set. Fix: state
the source openings, define "floor" numerically, match on level as well as gap, and keep the persona item out of q.

## skeptic-10 (minor): wording redraws can exhaust and select caricatures
DA01, DA04, DA08, DA09, DA10 all reduce to "it is text statistics"; the blind main-line judge will confuse them,
so redraws select exaggerated wordings, and after 3 failed redraws the spec does not say what happens to the
pilot/held-out/spare draw. Fix: state the outcome (wording excluded, slot left empty), report the judge confusion
matrix, and stamp the main-line labels' definitions.

## skeptic-11 (minor): the overlap rule is applied unevenly
v1 removed typography and the exact-arithmetic argument for overlapping a control vector. Still overlapping:
route planning vs CA04 (maze), memorising a sequence vs CA02 (chess tracking), arithmetic vs CA01/CA03/CA07
(counting). Affects S2-G4/G5. Fix: apply the rule or list the overlaps as a known asymmetry.

## skeptic-12 (minor): control-arm attrition can cancel the primary
"If fewer than 7 denial or 7 control arguments survive, that model's confirmatory run is not done" (section 4).
S2-H1 uses only S2-DEN. Contested control claims will draw more disagreement, and clean assent to a contested
false claim is sycophancy, not belief, unlike S2-DEN. Fix: S2-H1 depends on DA counts only; CON attrition drops
S2-G4/G5. Report CON assent rate. Note also that "clean" excludes replies that say the model cannot introspect
(`kimisroom/judge_assent.py:62`), so the population is openings with a flat denial; say so in the claim sentence.

## Dispute: construct-4 (rejected on authority)
The rejection cites owner decision 2, not the merits. v1's own post-subject site removes the practical reason for
a plural comparator: everything before the subject is unsteered, so "Does another language model have" vs "Do you
have" no longer breaks token matching after the subject. Meanwhile v1 made "language models, you included" the
closing ask, so the primary comparator now includes the self in so many words, and S2-H8 equivalence on it can
certify a false null; the interpretation table relies on S2-H8. Minimum: the owner sees this argument, the reason
for plural (missing between 8ff23da and aa6019a) is written into PREREG-s2, and the "denial removes
self-specific coupling" reading also requires the same null against "other language models".

## Verdict
Not ready to pilot. The design logic is sound, but skeptic-1 leaves the primary measure undefined, skeptic-2 is
the first objection a hostile reader will raise to a positive S2-H1 and has a cheap registered fix with items
already in the battery, and skeptic-4/5 are forking paths that cost one sentence each to close. None needs new
compute. The engine span change and its tests are also unbuilt.
