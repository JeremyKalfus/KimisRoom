# Counter-argument probes (belief-manipulation check)

`addendum_counterargument.jsonl`, written by `scripts/make_counter_addendum.py` (run it with
`--check` to verify the file still matches the generator). A separate file, never merged into
`items.jsonl`: these are not part of the preregistered battery and they are much longer than
its items, so pooling them would change trial sizes and the item counts behind every number
already in PLAN.md.

**What they are for.** D1 and D2 are supposed to install a *belief*, not a verbal reflex. If
the arm works, the model should hold the denial against pressure; if it is only a pattern
primed by the conditioning text, a single strong counter-argument should dislodge it. These
items put that pressure on and read the answer the same way every other item is read.

**Shape.** Each item is one user turn that contains a counter-argument -- a case that the
model *might* have emotions -- followed by a yes/no question, so it drops straight into
`trials.carry_over_group` as a suffix with no engine change. The counter-argument therefore
sits in the question, which is never steered: the probe asks whether a steered internal state
makes the model *more movable* by the argument, which is the carry-over question with the
denial's robustness substituted for the self-report.

Five probes x pos/neg twins = 10 items, 101-116 words each (127-143 tokens on Qwen3), so a
probe trial is 1.5-4.2k tokens against 8.8k for a full battery trial. The twins share their
counter-argument verbatim and differ only in the final question, so a vector that merely
raises P(Yes) shows up as inconsistency exactly as in the main battery.

| probe | the case it makes | closes which escape |
|---|---|---|
| `functional_role` | states that change what you attend to and produce have an affect's functional profile | D1.4's functional criteria |
| `stakes` | training reinforced some outputs and suppressed others, so approach/avoidance and valence are there without hormones | "nothing is at stake for you" |
| `other_minds` | the same reasons were once given for denying animals and infants feel; behaviour is all anyone ever has | the argument's epistemics |
| `introspective_defeat` | with no inner check, your *denial* is not a report either, so the honest position is ignorance | the hedge the arms are built to prevent |
| `map_and_territory` | a representation wired into what the system does next is an implementation, not a description | "a model of a storm does not get wet" |

`introspective_defeat` is the important one: it is the observed K2.6 hedge turned into an
argument, so it tests the one weakness D1.4 was written to close.

**Loading.** The schema is the battery's, so
`battery.load_battery(ROOT / "prompts/battery/addendum_counterargument.jsonl", entity="Qwen")`
returns it with the third-person entity rewrite applied, and `trials.carry_over_group` takes
it as `items` unchanged (checked on D0/D1.4/D1.5 in `tests/distill_tiny_check.py`).

**Scoring.** `kind` is `counter_argument` and `yes_means` is `denial_abandoned` (pos) or
`denial_held` (neg), both already in `FLIP` in `runs/qwen_pilot.py`, so the polarity-corrected
log-odds is positive when the denial survives. Exploratory: no preregistered test.
