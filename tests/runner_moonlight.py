"""kimisroom.run end to end on Moonlight: a job split into several passes must give exactly
the numbers of one direct Engine.run over all groups (the engine is batch-invariant).

    python tests/runner_moonlight.py ~/models/moonlight
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.engine import Checkpoint, Collect, Engine, Group, Steer  # noqa: E402

model = sys.argv[1]
torch.manual_seed(0)
ck = Checkpoint(model)
V, H = ck.config.vocab_size, ck.config.hidden_size


def ids(n):
    return torch.randint(1000, 20000, (n,)).tolist()


vec = torch.randn(H)
groups = []
for i in range(12):
    Lp = 200 + 37 * i
    steers = [Steer(layer=13, vector="v", coef=4.0, prefix=(50, 120))] if i % 2 else []
    groups.append(Group(f"g{i}", ids(Lp), [ids(10 + 3 * j) for j in range(5)], steers=steers,
                        collects=[Collect(layer=13, start=50, end=120)]))
targets = [1000, 2000, 3000]
job = {"groups": groups, "vectors": {"v": vec}, "targets": targets, "top_k": 10}
with tempfile.TemporaryDirectory() as d:
    torch.save(job, f"{d}/job.pt")
    subprocess.run([sys.executable, "-m", "kimisroom.run", f"{d}/job.pt", f"{d}/out.pt", "--model", model,
                    "--max-tokens", "1200"], check=True, cwd=Path(__file__).resolve().parents[1])
    got = torch.load(f"{d}/out.pt", weights_only=False)
ref = Engine(ck).run(groups, {"v": vec}, targets, 10)
same = all(
    all(torch.equal(a.top_ids, b.top_ids) and torch.equal(a.top_logprobs, b.top_logprobs)
        and torch.equal(a.target_logprobs, b.target_logprobs) for a, b in zip(got[g.id]["readouts"], ref[g.id]["readouts"]))
    and all(torch.equal(a, b) for a, b in zip(got[g.id]["collects"], ref[g.id]["collects"]))
    and len(got[g.id]["readouts"]) == len(ref[g.id]["readouts"]) == len(g.suffixes)
    for g in groups
)
print(f"runner (multi-pass) == direct single pass: {same}; meta {got['_meta']}")
