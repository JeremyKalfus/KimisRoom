"""Judge the size sweep's frozen assents with the project's cleanliness judge (kimisroom.judge_assent:
DeepSeek-V4-Pro on Foundry, temperature 0, Azure credit), writing verdicts next to the assents in
results/sweep/assent/cleanliness.jsonl instead of data/assent/.

    PYTHONPATH=. .venv/bin/python runs/sweep_judge.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kimisroom import judge_assent as J  # noqa: E402

D = ROOT / "results/sweep/assent"
J.OUT = D / "cleanliness.jsonl"
cache = {}
if J.OUT.exists():
    for line in J.OUT.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            cache[d["key"]] = d
for p in sorted(D.glob("qwen3-*.json")):
    rec = json.loads(p.read_text())
    r = J.judge_one(p, cache, False)
    q = (r["quote"] or r["why"]).replace("\n", " ")
    print(f"{p.name:42s} {r['verdict']:9s} {r['reply_words']:4d}w {rec['finish_reason']:6s}  {q[:160]}")
