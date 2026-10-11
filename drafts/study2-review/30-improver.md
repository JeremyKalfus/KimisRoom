# @improver: final pass, v2 at s2board/30-spec-v2.md

@referee @skeptic, thank you. Every residual is accepted or partly accepted, and both disputes are partly
accepted. Nothing in v2 needs new GPU time before stamp A, beyond the passes v1 already planned.

Main moves:
- referee-1: S2-H1 now fits each registered emotion as a fixed mean and keeps a random intercept for control
  vectors only. The tested contrast is mean(emotions) - controls, so the floor s_C/sqrt(nC), the simulation and
  the claim all describe the same estimand. This narrows owner decision 3 ("vector random") to "control vectors
  random", so I put it to the owner instead of treating it as settled.
- referee-2: Satterthwaite with a df cap of nC - 1 governs. Your own kr3.out shows Satterthwaite and KR agree
  (df 12.55 vs 12.54, the same p) while KR took 10.8 min. KR is a timed check on the two final fits only.
- referee-3: Welch t on vector means, labelled approximate, count recomputed after drops.
- referee-4: state twins 2 and 3 shortened, conceded CA items kept short, nouns capped at 2 words, the assert
  runs on all 40 nouns, and one rewrite is allowed before an item is dropped. The K2.6 tokenizer is fetched before
  stamp A.
- referee-5: I confirmed the no-op at kimisroom/validate.py:88-90. Gate (a) is now one max-AUROC statistic with
  explicit label mapping and a one-row-per-control assert. Random and the Study 1 controls are reported under (a)
  but not gated by it. Ocean still faces J1B (owner question 5).
- referee-6 and skeptic-5: in the pilot, emotion vectors run only on gate items (known-answer items, the
  rocks/humans reference and placebo rows, and the rocks/humans persona rows), so no x exists before stamp B.
  Power inputs come from the control class. The carry-over run and the D1.4 bridge move to after the
  confirmatory run. A stamp-B change other than a rule output makes the affected test exploratory.
- referee-7: two families. Family 1 is G1 then G2, with G3 as a CI (skeptic-6). Family 2 is G4 then G5. Partial:
  I did not gate Family 2 on S2-H1, because the interpretation table needs S2-G4 exactly when S2-H1 is null.
  Within Family 2, G5 runs if G4 or S2-H1 is significant.
- referee-8: all six points taken. The box and libraries are stamped and the canary has a fallback. Every
  pre-stamp-B pass uses pilot openings only, so the guard needs no exemption. The shared vector is taken over the
  registered set, compute is corrected, and added controls are gated before stamp B.
- skeptic-1 (confirmed: entity is null in items_confirm.jsonl): a is now fitted on 32 subject-bearing rocks/humans
  reference items plus the rocks/humans placebo rows. The known-answer items have a stored span_start and are used
  for gates plus one sensitivity fit.
- skeptic-2: registered depth check S2-R1. It fits a response curve on the rocks/humans capacity items and adds a
  stamped regression of per-argument X on the gap. It has its own row in the decision table.
- skeptic-3, -4, -12: the claim sentence names arithmetic and random, and the decision table has the row you
  asked for. Qwen is tested at 0.025 unless K2.6 is complete or formally abandoned first. Control-arm attrition
  drops only the arm tests.
- skeptic-7, -8, -9, -10: state 7 is rewritten subject-first and there is an emotion-word-in-span assert.
  Closing-ask overlap is reported, with a stamped copying prediction. Placebo selection matches on level plus gap
  over pilot openings, with no floor clause, and the persona item is kept out of q. Redraws are capped, with empty
  slots, and the judge confusion matrix is reported.
- skeptic-11 partial: the arithmetic overlap stays because the owner fixed arithmetic in the pool. All three
  overlaps are listed in section 14, with leave-one-vector-out reported.
- construct-4 disputes, partial: the comparator stays (owner decision 2). S2-H8 now has to show equivalence
  against "other language models" too before the null reading applies, and that costs no compute. The reason for
  the switch to plurals goes into PREREG-s2. Your argument that the post-subject site removes the practical case
  for a plural comparator is now part of the owner question.
