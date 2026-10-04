"""Time one Kimi K2.6 MoE layer on one GPU: disk read, host-to-device, and the layer itself
at two batch sizes, to size the full layer-by-layer run.

    python tests/bench_layer_k26.py /nvme/k26 1
"""

import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kimisroom.engine as E  # noqa: E402
from kimisroom.engine import BF16, Checkpoint, Engine, Group, _Layout  # noqa: E402

for k in ("TILE_ATTN", "TILE_MLP", "TILE_EXPERT", "TILE_ROUTER", "TILE_NORM"):  # tile-size sweeps
    if k in os.environ:
        setattr(E, k, int(os.environ[k]))

path, L = sys.argv[1], int(sys.argv[2])
ck = Checkpoint(path)
eng = Engine(ck)
H = ck.config.hidden_size

t = time.time()
cpu = ck.load_layer_cpu(L)
t_read = time.time() - t
nbytes = sum(v.numel() * v.element_size() for v in cpu.values())
t = time.time()
w = eng._to_device(cpu)
torch.cuda.synchronize()
t_h2d = time.time() - t
print(f"layer {L}: {nbytes / 1e9:.1f} GB, read {t_read:.1f}s ({nbytes / 1e9 / t_read:.2f} GB/s), to GPU {t_h2d:.1f}s")


def trials(n_groups: int, prefix: int = 1500, n_suffix: int = 40, suffix: int = 60) -> list[Group]:
    return [Group(str(i), [0] * prefix, [[0] * suffix] * n_suffix, readout="none") for i in range(n_groups)]


for n in (5, 20, 40):
    groups = trials(n)
    T = sum(g.n_tokens for g in groups)
    h = (torch.randn(T, H) * 0.02).to(BF16).cuda()
    layout = _Layout(groups, eng.device)
    eng.layer(w, h, layout)  # warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    eng.layer(w, h, layout)
    torch.cuda.synchronize()
    dt = time.time() - t
    print(f"{T:>9,} tokens: {dt:.2f}s per layer, {T / dt:,.0f} tok/s/layer, "
          f"full model ~{T / (dt * 61):,.0f} tok/s per GPU, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    del h, layout
    torch.cuda.empty_cache()
