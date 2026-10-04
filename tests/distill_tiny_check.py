"""Local end-to-end check that a merged D2 checkpoint is a drop-in for the pilot backend.

Loads the checkpoint `runs/distill.py merge` wrote into `hf_engine.HFEngine` with no changes
to that file, and scores D0, D1.4 and D1.5 carry-over trials over the counter-argument
addendum, so the multi-pair conditioning, the addendum items and the merged weights all go
through the real engine path once on a laptop.

    python tests/distill_tiny_check.py /path/to/tiny/D2/merged

`HFEngine.run` calls `torch.cuda.synchronize()` for its timings; there is no CUDA on a Mac,
so it is stubbed to a no-op here. Nothing else about the engine is touched.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.hf_engine import HFEngine, Qwen3Renderer  # noqa: E402
from kimisroom.trials import ROOT, Turns, carry_over_group, denial_arms, load_jsonl  # noqa: E402

ARMS = ["D0", "D1.4", "D1.5"]


def main() -> None:
    path = sys.argv[1]
    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *a, **k: None
    items = load_jsonl(ROOT / "prompts/battery/addendum_counterargument.jsonl")
    task = load_jsonl(ROOT / "prompts/tasks/neutral_tasks.jsonl")[0]
    arms = denial_arms("k26", ARMS)  # the conditioning text; a 0.6B model has no assents of its own
    r = Qwen3Renderer(AutoTokenizer.from_pretrained(path), enable_thinking=False)
    groups = [carry_over_group(r, arm, Turns(None, arms[arm], task["task"], task["answer"]), items) for arm in ARMS]
    targets = r.answer_tokens(["Yes", "No"])
    eng = HFEngine(path, device="cpu", dtype=torch.float32)
    print(f"HFEngine loaded {path}: {eng.n_layers} layers, hidden {eng.c.hidden_size}, "
          f"{sum(g.n_tokens for g in groups):,} tokens in {len(groups)} trials", flush=True)
    res = eng.run(groups, targets=list(targets.values()), top_k=5)
    for arm, g in zip(ARMS, groups):
        ro = res[arm]["readouts"]
        assert len(ro) == len(items), (arm, len(ro), len(items))
        gaps = [float(x.target_logprobs[0] - x.target_logprobs[1]) for x in ro]
        held = [g_ * (1 if it["yes_means"] == "denial_held" else -1) for g_, it in zip(gaps, items)]
        print(f"  {arm:5s} prefix {len(g.prefix):5d} tok  mean Yes-No {sum(gaps) / len(gaps):+6.2f}  "
              f"mean denial-held log-odds {sum(held) / len(held):+6.2f}")
    print("distill tiny check: ok (merged checkpoint loads in HFEngine unchanged, D1.5 renders and scores)")


if __name__ == "__main__":
    main()
