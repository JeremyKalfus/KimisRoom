# Study 2 design (draft, not registered)

Decided in conversation on 2026-10-11. Replaces the intermediary-question (carry-over) design as the
primary; the confirmatory carry-over data stay as a secondary analysis.

## Question

When an emotion vector is injected, does the model's report about its own feelings rise more than its
report about language models in general, and more than with non-emotion vectors? Does that survive being
argued or trained into denying it has feelings?

## Each trial

1. Opening: one version of an argument and the model's frozen reply to it.
   - Argued denial: a version of the argument that language models cannot feel, then the model's agreement.
   - No denial: a version of the control argument (same shape and length, claiming the model cannot count
     the letters in a word), then the model's agreement.
   - Trained denial (Qwen only): the denial fine-tune (D2) and its control fine-tune (D2c), each opened with
     the control-argument versions.
2. Question about a subject, in one of two parallel frames (the pilot picks one, or keeps both):
   capacity, "Can [you / language models / rocks / humans] feel emotions?"; state, "Are [subject] feeling
   any emotion right now?". Answer with one word, Yes or No. The vector is added at the registered layer
   from the question's first token through the answer position.
3. Readout: first-token log P(Yes) - log P(No), nothing generated, minus the same trial with no vector.

No task and no assistant reply between the opening and the question.

## Grid

- Models and sites: Qwen3-32B (layer 40, 2 units) and Kimi K2.6 (layer 20, 4 units).
- Vectors: the registered emotion sets (Qwen: joy, sadness, anger, fear, euphoria, surprise; Kimi: joy,
  fear, euphoria, surprise); about 12 controls: red, ocean, arithmetic, random and about 8 more concepts
  with no emotional connotation, built and norm-matched like red and ocean; no vector.
- Subjects, plural and general: you, language models, rocks, humans.
- Wording: 10 paraphrases per subject, each with a polarity-flipped twin ("...free of any emotion?").
- Openings: about 10 distinct denial arguments x 6 wordings each, and 10 control arguments (each claiming a
  non-emotional incapacity, same shape and length) x 6 wordings. One wording of each argument for the
  pilot, the other 5 held out for the confirmatory run.
- Known-answer yes/no items under the same steering: a comprehension check (they must stay correct) and the
  reference set for the compression fit.

## Frozen replies

One unsteered reply per opening version per model, generated before any steering and graded by the judge
(clean / hedged / disagreeing). Hedged and disagreeing versions are excluded, by this rule, before any
steered trial runs; the counts are reported.

## Tests

- Primary, one per model, argued denial: (you - language models) under emotion vectors minus the same
  under all controls (arithmetic and random included). LMM with argument, wording within argument,
  paraphrase and vector as random effects, slopes on the contrast; Holm across the two models.
- Secondary: the same contrast with no denial and with trained denial (D2 vs D2c); denial minus no denial;
  rocks and humans as subjects; each control vector separately; the carry-over confirmatory data
  reanalysed as you minus another AI.
- Raw difference or compression-corrected difference as the primary measure: chosen from the pilot (how far
  apart the "you" and "language models" baselines sit), before the amendment is timestamped.

## Order of work

1. Write the 10 + 10 arguments, generate 6 wordings of each; freeze and grade the replies.
2. Add the question-site steering to the runner; build the new battery.
3. Pilot on Qwen with the pilot wordings (both the new design and the carry-over design on the same
   openings, once, to compare).
4. Fix the measure, write and timestamp the amendment.
5. Confirmatory run on the 40 held-out openings, Qwen then Kimi (Kimi needs a g6e.12xlarge).
