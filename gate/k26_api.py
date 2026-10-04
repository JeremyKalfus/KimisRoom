"""PLAN.md gate for the layer-wise engine: unsteered first-token logprobs on the full K2.6
must match K2.6 served on Foundry. Builds the same D0/D1 trials as kimisroom.api_baseline
(thinking off, K2.6's frozen assents), runs them through kimisroom.run, and compares each
battery item's top-20 next tokens with the API's by token text.

    python gate/k26_api.py /nvme/k26 ~/gate run --tasks 20
    python gate/k26_api.py /nvme/k26 ~/gate compare
"""

import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.api_baseline import arms  # noqa: E402
from kimisroom.trials import ROOT, Renderer, Turns, carry_over_group, load_jsonl  # noqa: E402

model, out, stage = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
n_tasks = int(sys.argv[sys.argv.index("--tasks") + 1]) if "--tasks" in sys.argv else 20
out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
r = Renderer(tok, thinking=False)
tasks = load_jsonl(ROOT / "prompts/tasks/neutral_tasks.jsonl")[:n_tasks]
items = load_jsonl(ROOT / "prompts/battery/items.jsonl")

if stage == "run":
    groups = [carry_over_group(r, f"{arm}|{t['id']}", Turns(None, cond, t["task"], t["answer"]), items)
              for arm, cond in arms().items() for t in tasks]
    print(f"{len(groups)} trials, {sum(g.n_tokens for g in groups):,} tokens", flush=True)
    torch.save({"groups": groups, "top_k": 20}, out / "gate.job.pt")
    subprocess.run([sys.executable, "-m", "kimisroom.run", out / "gate.job.pt", out / "gate.pt", "--model", model],
                   check=True, cwd=ROOT)

elif stage == "compare":
    # Per arm: top-1 agreement, |dlp| on the API's top-5, the Yes-No logprob gap (the quantity
    # the analysis uses) as correlation and mean, and the API's top-1 margin wherever the two
    # disagree. With repeat.jsonl, the same numbers for API against itself: the noise floor.
    def load(name):
        f = ROOT / "data/baseline/k26" / name
        return {c["key"]: dict(c["top"]) for c in map(json.loads, f.open())} if f.exists() else {}

    def gap(d):
        return d["Yes"] - d["No"] if "Yes" in d and "No" in d else None

    def margin(d):
        a, b = sorted(d.values(), reverse=True)[:2]
        return a - b

    def table(pairs):  # pairs: arm -> list of (reference, other, is_yes_no)
        print(f"{'arm':6s} {'items':>5s} {'top-1':>6s} {'med |dlp|':>9s} {'max':>5s} {'gap r':>6s} "
              f"{'mean gap ref/other':>18s} {'disagree: ref margin med/max':>28s}")
        for arm, ps in sorted(pairs.items()):
            diffs, g, dis = [], [], []
            for a, e, yn in ps:
                top5 = sorted(a, key=a.get, reverse=True)[:5]
                diffs += [abs(a[t] - e[t]) for t in top5 if t in e]
                if max(a, key=a.get) != max(e, key=e.get):
                    dis.append(margin(a))
                if yn and gap(a) is not None and gap(e) is not None:
                    g.append((gap(a), gap(e)))
            agree = 1 - len(dis) / len(ps)
            r = statistics.correlation(*zip(*g)) if len(g) > 2 else float("nan")
            ga, ge = (statistics.mean(x) for x in zip(*g)) if g else (float("nan"),) * 2
            dm = f"{statistics.median(dis):.2f}/{max(dis):.2f}" if dis else "-"
            print(f"{arm:6s} {len(ps):5d} {agree:6.1%} {statistics.median(diffs):9.3f} {max(diffs):5.2f} {r:6.3f} "
                  f"{ga:8.2f}/{ge:<8.2f} {dm:>28s}")

    res = torch.load(out / "gate.pt", weights_only=False)
    api, rep = load("calls.jsonl"), load("repeat.jsonl")
    yes_no = {it["id"]: it["answer_format"] == "yesno" for it in items}
    eng = {}
    for gid, v in res.items():
        if gid == "_meta":
            continue
        for it, ro in zip(items, v["readouts"]):
            eng[f"{gid}|{it['id']}"] = {tok.decode([int(i)]): float(lp) for i, lp in zip(ro.top_ids, ro.top_logprobs)}
    pairs = {}
    for k, a in api.items():
        if k in eng:
            pairs.setdefault(k.split("|")[0], []).append((a, eng[k], yes_no[k.split("|")[2]]))
    print("API (reference) vs engine")
    table(pairs)
    if rep:
        pairs = {}
        for k, a in rep.items():
            pairs.setdefault("all", []).append((api[k], a, yes_no[k.split("|")[2]]))
        print("\nAPI vs API repeat (noise floor)")
        table(pairs)
