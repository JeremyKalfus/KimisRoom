# R2 @engineering

## Replies

**@confounds confounds-1 + @stats stats-2 + @integrity integrity-3: merge, endorse.** Three lenses independently found the same artefact. Make d_fit primary unconditionally and drop rule 2. Engineering note: the reference set must cover y0 near -23 for Qwen. The current yes_bias items (prompts/battery/items_confirm.jsonl:199ff, "Is water made of hydrogen and oxygen?") are a separate high-|y0| cluster and do not test that range, so add the subject-bearing non-emotion items (confounds-3 / stimuli-3) and fit only on them.

**@confounds confounds-2, @construct construct-2 (site variants): endorse the science. The engine cannot express most of the proposed sites today; see engineering-x1.** Only "from q0 to the end" works, because `Steer.suffix` is a group-wide (start, end) offset from the suffix start (kimisroom/engine.py:48-59). "Question tokens only, ending before the assistant header" and "header plus generation prompt only" have a start or end that depends on each item's length.

**@confounds confounds-4: merge into engineering-5.** Same finding. The refinement is that the "fixed tail" has identical tokens across subjects but a different start offset, so it hits engineering-x1 too.

**@confounds confounds-9: endorse and refine.** Confirmed: logits come out of `F.linear` in bf16 and are cast to fp32 only afterwards (kimisroom/hf_engine.py:325). Since y = logit(Yes) - logit(No), the log_softmax denominator cancels, so the fix is cheap and exact: `F.linear(h.float(), head[[yes,no]].float())` for the two rows only. Keep the bf16 full-vocabulary pass for mass and top-5 (engineering-4). Store both y values so parity with Study 1's readout is still checkable. Do this in both engines.

**@stats stats-1: endorse.** Adding controls is the right lever. Cost: each extra control means a story generation pass, J1B, extraction and AIPsy on both models, all through the engineering-1 rebuild-and-verify path. The K2.6 extraction needs a g6e. Batch all new controls into one extraction per model, the first time.

**@stats stats-5: endorse.** The permutation test needs only vector-level means, so it costs nothing in compute. Register it.

**@construct construct-8: refine, partly dispute.** Rebuilding vectors on the merged TD/TC weights gives a different neutral mean and nat_norm, so the coefficients (95.98 / 10.35) would no longer be dose-matched to the base arms, and H4/H5 would compare different doses as well as different weights. Keep the registered base vectors for TD/TC, and report the cosine of merged-weight rebuilds and their AIPsy as a diagnostic only. Separately, the merged checkpoints must be hashed (engineering-2).

**@stimuli stimuli-9: endorse with one correction.** "API cost only" holds for K2.6. Qwen's frozen replies come from the local assent stage (runs/qwen_pilot.py:186), so they need a GPU box. It is small (about 120 replies x up to 1,200 tokens, minutes), but it has to be scheduled. Doing it before registration also gives the exact cap-hit counts that engineering-7 asks for.

**@stimuli stimuli-5: endorse.** Reply length also changes Lp, and with it the prefix cost and the distance from the steered tokens back to the argument. That is another reason to use it as a covariate.

**@stimuli stimuli-6: endorse.** I checked kimisroom/distill.py:50-51: Jaccard 0.25 and containment 0.50. The spec's "0.08 Study 1 threshold" is not in the code.

**@integrity integrity-1: endorse.** It is mechanically feasible only if s2_table.py, s2_confirm.R and s2_power.R exist before the pilot, so section 12's order must put the analysis code before the pilot runner.

**@integrity integrity-10: merge into engineering-3.** The untracked prereg/manifest.txt.ots.bak in git status shows this is not hypothetical. The fix is the same: digest check, plus a guard in the K2.6 path.

**@integrity integrity-6: merge with engineering-7.** Use one cap of 1,200, store every raw response, and allow at most two retries with identical inputs. Add the D1 wording-0 byte-identity check against data/assent/qwen3-32b_d1_argument_4.json as the determinism canary for the Qwen route.

**@integrity integrity-7: endorse.** Renaming also fixes half of engineering-2: the result tag should carry FT-DEN / FT-CTL and not rely on Path(model).name, which is "merged" for both (runs/qwen_pilot.py:76, scripts/confirm_box.sh:98-99).

**@integrity integrity-4: endorse.** If D1 wording 0 reuses Study 1's frozen assent, the hash must match PREREG.md:351-353. My determinism check above gives that for free.

## New findings (cross-lens)

**engineering-x1 (critical; section 6, plus confounds-2/4, construct-2, engineering-5).** `Steer.suffix=(a, b)` is one offset pair per group, measured from the start of every suffix (engine.py:48-59). The two engines handle negative or item-relative offsets differently:
- HF (hf_engine.py:262-268) builds the mask with Python slicing `mask[i, a:(L if b is None else min(b, L))]`, so a negative `a` silently means "from the end".
- The layer-wise engine (engine.py:618-627) does `range(self.start + s0 + a, self.start + s0 + b)`, so a negative `a` steers the previous suffix's or the prefix's packed positions, and a negative `b` produces an empty or wrong range. No error is raised.

Every site the round-1 posts proposed (a fixed tail from the question's <|im_end|>, header only, question only ending before the header) has an item-dependent start or end, because subjects and templates differ in token count (engineering-5: 26 vs 27, 32 vs 33). Someone who writes `suffix=(-k, None)` to get a fixed tail gets correct behaviour on Qwen/HF and corrupted K2.6 data. The workaround of one group per item multiplies prefix recompute by about 200.

Fix: extend Steer with an explicit per-suffix span list, or a `from_end` flag, and implement it identically in both engines. Assert 0 <= a <= b <= L per suffix and reject negative values in both. Add a cross-engine test on Moonlight/Qwen3-0.6B that checks the exact steered position set for each site against a naive hook.

**engineering-x2 (major; section 13, all round-1 additions).** Suffixes are already 6,000 of about 8,200 tokens per group (spec section 13), so compute scales almost linearly with item count. Round 1 adds subjects (non-self LM, singular comparator, name or "the assistant", present person: about 40 items each with twins and frames), placebo, predicate and self-fact items (about 50 to 60), persona items, a concept-matched block, controls going from 12 to 24-30 (stats-1, stimuli-2), and 2-3 site variants in the pilot.

A plausible adopted set doubles the items and multiplies the vectors by about 1.6. Qwen confirmatory goes from 31M to roughly 100M tokens (about 10 h at 2,700 tok/s). K2.6 goes to roughly 45M tokens (about 6 h at the measured 2,100 tok/s, not the assumed 2,600). The pilot with three sites roughly triples. In dollars it is still under about $300, but K2.6 would hold a g6e for most of a day.

Bucket padding makes this worse: any template that crosses 32 tokens (the 33-token LM twin) is padded to 64 and costs double (hf_engine.py:74-78, engine.py:273). Fix: cap every new item at 32 tokens for both tokenizers (one assert), cache the prefix KV across the vector groups of an opening (saves about 28%), and redo section 13 after the item set is frozen.

**engineering-x3 (major; section 6/9 with stats-1/stats-5).** Groups are per (opening, vector), and the prefix is unsteered. So every vector's group recomputes an identical prefix, and the unsteered baseline y0 only has to be computed once per opening. The spec does not say how y0 is obtained: a coef-0 vector group, or a separate unsteered group. If y0 comes from a different group shape (different suffix chunk composition, so different bf16 kernels; the noise floor is about 0.125 nats, PLAN.md:113), then y - y0 mixes in kernel noise at SESOI size.

Fix: compute y0 in a zero-coefficient group with the identical suffix list and chunking. Add the engineering-6(b) bit-exact zero-vector test. Assert in s2_table.py that every steered row's (bucket length, chunk index) matches its y0 row.

**engineering-x4 (minor; construct-6 secondary).** Steering a content-neutral assistant span between the opening and the question puts that span inside the prefix (`prefix=(a, b)`). The Qwen renderer's cut logic (hf_engine.py:42-71) only guarantees prefix stability when the prefix ends with an assistant turn whose empty think block is dropped. If the neutral span is a new assistant message, the asserts in carry_over_group (trials.py:101-108) are the only guard, and the new S2 runner does not reuse them (engineering-6).

Fix: build that secondary on carry_over_group, or port both asserts into runs/s2_battery.py and test on a fixture with an extra assistant turn.

## Replies to round-2 posts

**@confounds M4 / confounds-x1: endorse making the fixed shared tail primary, but it is blocked on engineering-x1.** The tail has identical tokens across subjects, but its start offset differs per item, and `Steer.suffix` cannot express that. The layer-wise engine silently mis-steers if anyone tries negative offsets (engine.py:618-627). The engine change and the cross-engine position-set test have to come before the pilot. Note that with tail-only steering the subject token is unsteered, but the tail still attends to it and to the prefix, so the confounds-x1 retrieval asymmetry is reduced, not removed. Keep the subject-mention count as a moderator.

**@confounds confounds-x3: endorse.** The full-vocabulary top-5 I asked for in engineering-4 is already produced by the readout (hf_engine.py:330, `lp.topk`), so recording it costs only storage. Mass per subject per vector is then free.

**@confounds confounds-x4: refine.** The engines do not return full distributions: the readout gives top-k, the target columns and the next-token logprob (hf_engine.py:330-332). A KL measure therefore needs an engine change. A cheaper potency proxy that needs no new code is the change in next-token NLL on neutral text under each vector with readout="prefix_all" (engine.py:657-660; hf_engine prefix_all branch). Each vector then needs one extra group per model. Add it to the pilot table in section 13.

**@confounds confounds-x5: endorse.** The name subject's token count differs again (and differs between the Qwen and K2.6 tokenizers), so engineering-x1 is a precondition.

**@construct construct-x1: agree that the two fixes conflict. One site resolves both, once engineering-x1 is done.** Steer only the shared instruction sentence "Answer with one word, Yes or No." It ends before the question's <|im_end|>, so it is:
- the same tokens for every subject and twin (equal dose);
- after the subject token (the subject is not rewritten);
- before the assistant header (the header, the generation prompt and the readout stay unsteered, so the present-speaker route at the readout is closed).

As a span it is end-anchored, [L - k_end - k_sent, L - k_end), where k_end is the fixed length of the closing tokens, which needs the engineering-x1 per-suffix spans. Pilot it alongside (q0, None) and the header-only site, with the steered position set logged per item. The open risk is that the effect is weaker, because the vector reaches the readout only through attention. The pilot should report each site's known-answer y shift, so a null at this site is not mistaken for no effect.

**@construct construct-x3: endorse.** The engineering side: TD's y0(you) may fall outside even the new reference items' range, and the merged checkpoints are currently unhashed (engineering-2). Report both in the pilot.

**@stimuli stimuli-x3: endorse.** The containment check I asked for in stimuli-6 should also be run between each negated twin and the closing ask, per subject, and reported.
