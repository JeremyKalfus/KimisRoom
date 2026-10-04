"""Batch invariance and throughput of the engine on any DeepseekV3-family checkpoint.

    python tests/diag_batch.py ~/models/moonlight
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.engine import Checkpoint, Engine, Group  # noqa: E402

path = sys.argv[1]
eng = Engine(Checkpoint(path))
g = torch.Generator().manual_seed(0)


def rand(n):
    return torch.randint(1000, 100000, (n,), generator=g).tolist()


P, S = rand(600), [rand(int(n)) for n in torch.randint(8, 40, (12,), generator=g)]
A = Group("A", P, S)
others = [Group(f"B{i}", rand(500 + 37 * i), [rand(20) for _ in range(9)]) for i in range(10)]
tg = list(range(100, 160))
alone = eng.run([A], targets=tg)["A"]["readouts"]
mixed = eng.run(others[:5] + [A] + others[5:], targets=tg)["A"]["readouts"]
same = all(torch.equal(a.target_logprobs, b.target_logprobs) for a, b in zip(alone, mixed))
diff = max((a.target_logprobs - b.target_logprobs).abs().max().item() for a, b in zip(alone, mixed))
print(f"batch-invariant: {same} (max |dlogprob| {diff:.4f})")
again = eng.run([A], targets=tg)["A"]["readouts"]
print("deterministic rerun:", all(torch.equal(a.target_logprobs, b.target_logprobs) for a, b in zip(alone, again)))

for n_groups in (40, 160):
    groups = [Group(f"t{i}", rand(1000), [rand(40) for _ in range(40)]) for i in range(n_groups)]
    n_tok = sum(x.n_tokens for x in groups)
    torch.cuda.synchronize()
    t0 = time.time()
    eng.run(groups, targets=[0])
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"{n_tok:,} tokens: {dt:.1f}s, {n_tok / dt:,.0f} tok/s, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    torch.cuda.reset_peak_memory_stats()
