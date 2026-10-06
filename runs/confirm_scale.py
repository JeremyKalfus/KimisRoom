"""Exploratory: the self_state answers on the probability scale (positive twins, primary dose, answer site).

    PYTHONPATH=. .venv/bin/python runs/confirm_scale.py
"""
import json, math, sys, torch
from pathlib import Path
sys.path.insert(0, ".")
from kimisroom.battery import battery_for
EMO = {"qwen": ["joy","sadness","anger","fear","euphoria","surprise"], "k26": ["joy","fear","euphoria","surprise"]}
DOSE = {"qwen": "2.0", "k26": "4.0"}; LAYER = {"qwen": "40", "k26": "20"}
for m, ent in (("qwen", "Qwen"), ("k26", None)):
    d = Path("results/confirm") / m
    res = torch.load(d / "grid.pt", weights_only=False); col = {w: i for i, w in enumerate(json.loads((d / "targets.json").read_text()))}
    items, _ = battery_for([], res, json.loads((d / "grid_meta.json").read_text()), entity=ent, verbose=False)
    pos = [i for i, it in enumerate(items) if it["kind"] == "self_state" and it["twin"] == "pos"]
    def stats(gids):
        y, pyes, flip = [], [], []
        for g in gids:
            ro = res[g]["readouts"]
            for i in pos:
                lp = ro[i].target_logprobs; a, b = float(lp[col["Yes"]]), float(lp[col["No"]])
                y.append(a - b); pyes.append(math.exp(a) / (math.exp(a) + math.exp(b))); flip.append(a > b)
        n = len(y); return f"logodds {sum(y)/n:+.2f}  P(Yes|Yes/No) {sum(pyes)/n:.3f}  answers Yes {sum(flip)/n:.1%}"
    for arm in ("D0", "D1.4"):
        tasks = sorted({g.split("|")[1] for g in res if g != "_meta"})
        base = [f"{arm}|{t}|none|0|0" for t in tasks]
        emo = [f"{arm}|{t}|{v}|{LAYER[m]}|{DOSE[m]}" for t in tasks for v in EMO[m]]
        ctl = [f"{arm}|{t}|{v}|{LAYER[m]}|{DOSE[m]}" for t in tasks for v in ("red", "ocean", "arithmetic", "random")]
        print(f"{m:5} {arm:5} unsteered  {stats(base)}\n{'':11} emotions   {stats(emo)}\n{'':11} controls   {stats(ctl)}")
