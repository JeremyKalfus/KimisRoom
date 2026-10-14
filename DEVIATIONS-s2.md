# Study 2 deviations

Dated entries, committed before results are read (spec section 11).

This log belongs to `prereg/PREREG-s2.md` (section 7). It records:
- any change to `prereg/PREREG-s2.md` after stamp A other than filling the four STAMP B BLANKs, with a diff and a reason; every test the change touches becomes exploratory;
- convergence fallbacks used by `runs/s2_confirm.R` (from its `fallbacks.csv`);
- engine or hardware changes;
- dropped arguments or wordings;
- sampled nouns skipped and mental-process controls dropped at the gates;
- the determinism canary fallback, if used (`runs/s2_assent.py canary` appends that entry itself);
- any formal abandonment of K2.6, committed before any Qwen confirmatory result is computed.

Each entry is a level-2 heading `## YYYY-MM-DD: title` followed by the details, appended below in date order.

## 2026-10-13: Qwen box replaced by a g6e.24xlarge after a capacity shortage

On 2026-10-12 the session-1 box (g6e.12xlarge, 4 NVIDIA L40S, us-east-1c) stopped itself when idle. AWS then had no g6e.12xlarge or g6e.24xlarge capacity in us-east-1c for about 10 hours.

The stopped box's disk was saved as an image, and a copy was launched on 2026-10-13 at 14:58 UTC in another us-east-1 zone (us-east-1b, per the box record) as a g6e.24xlarge. It has the same four L40S GPUs and the same disk, so the same driver, CUDA, torch, transformers and weights. Only the CPU count and RAM differ.

Qwen's round-2 extension (the 27 capped sampled draws that `runs/s2_assent.py tally` listed in `data/s2/replies/qwen3-32b/extend_needed.json`) refused to run on it, because the reply box's `box.json` records the instance type g6e.12xlarge. It was rerun with `--allow-stack-change`, which records the override. In `data/s2/replies/qwen3-32b/extend.json` the override shows as the `resumed` entry at 2026-10-13T15:41:07, whose `stack_diff_vs_reply_box` is `{"box_type": ["g6e.12xlarge", "g6e.24xlarge"]}`: the instance type is the only fingerprint field that differed (the weight hashes and the layer-to-GPU map are identical). The top-level `allow_stack_change: false` in that file belongs to round 1 on 2026-10-12, which ran on the original box with no override. The round-2 batch plan (`extend_batches.json`, one batch of 27 draws) and `round2_finished` (2026-10-13T15:51:31) record the result: 27 extended, 1 still capped at 2,048 new tokens, no draw 0 still capped. `round2_finished` also records the SHA-256 of the needed list it ran on (1dbb1779...); the copy now on disk was rewritten by a later `tally` and lists nothing still pending.

Qwen's unsteered pass and potency run also ran on this g6e.24xlarge box.
