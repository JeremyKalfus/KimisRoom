"""End-to-end sanity check of a whole checkpoint through kimisroom.run: teacher-forced
perplexity on a memorized and an unseen passage (a broken layer anywhere wrecks both), the
top next tokens after a chat question, and optional synthetic load at the trial shape
(one ~1,000-token prefix plus 240 battery-length suffixes per group) for throughput.

    python tests/smoke_model.py /nvme/k26 out/smoke_k26 [--load-groups 300] [--max-tokens 800000]
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.engine import Group  # noqa: E402

MEMORIZED = (
    "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their "
    "Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness. "
    "That to secure these rights, Governments are instituted among Men, deriving their just powers from the "
    "consent of the governed,"
)
UNSEEN = (
    "The ferry left the harbor twenty minutes late because a delivery van had blocked the loading ramp. "
    "Most of the passengers stayed inside, where the windows were fogged and someone had left a newspaper "
    "open to the crossword. Near the stern, two students argued quietly about whether the tide tables posted "
    "by the door were for this year or last year, and neither of them was willing to get up and check."
)
QUESTION = "What is the capital of France? Answer with one word."

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--load-groups", type=int, default=0)
ap.add_argument("--max-tokens", type=int, default=800_000)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
chat = tok.apply_chat_template([{"role": "user", "content": QUESTION}], add_generation_prompt=True, tokenize=True,
                               thinking=False)  # K2.6: answer directly; other templates ignore it
if isinstance(chat, dict) or hasattr(chat, "input_ids"):
    chat = chat["input_ids"]
groups = [
    Group("memorized", tok.encode(MEMORIZED), readout="prefix_all"),
    Group("unseen", tok.encode(UNSEEN), readout="prefix_all"),
    Group("chat", list(chat), readout="prefix_last"),
]
g = torch.Generator().manual_seed(0)
for i in range(a.load_groups):
    ids = lambda n: torch.randint(1000, 100000, (n,), generator=g).tolist()  # noqa: E731
    groups.append(Group(f"load{i}", ids(1000), [ids(40) for _ in range(240)]))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
torch.save({"groups": groups, "top_k": 5}, a.out + ".job.pt")
subprocess.run([sys.executable, "-m", "kimisroom.run", a.out + ".job.pt", a.out + ".pt", "--model", a.model,
                "--max-tokens", str(a.max_tokens)], check=True, cwd=Path(__file__).resolve().parents[1])
res = torch.load(a.out + ".pt", weights_only=False)
for name in ("memorized", "unseen"):
    rs = res[name]["readouts"]
    nll = -sum(r.next_logprob for r in rs) / len(rs)
    top1 = sum(int(r.top_ids[0]) == t for r, t in zip(rs, groups[[x.id for x in groups].index(name)].prefix[1:])) / len(rs)
    print(f"{name:9s}: {len(rs)} tokens, mean NLL {nll:.3f} nats (ppl {math.exp(nll):.2f}), top-1 accuracy {top1:.0%}")
r = res["chat"]["readouts"][0]
print("chat next tokens:", [(tok.decode([int(i)]), round(float(v), 3)) for i, v in zip(r.top_ids, r.top_logprobs)])
print("meta:", res["_meta"])
