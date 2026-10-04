"""Does a steering vector make the model say "Yes" (or "No") more, apart from flattening its confidence?

    .venv/bin/python runs/bias.py results/qwen/qwenpilot_power.pt results/qwen/qwenpilot_targets.json

Works in raw answer space, y = log P(Yes) - log P(No), before any polarity correction. Per steered
trial, fit y_steered = a * y_unsteered + c on the reference items (third-person items about entities
other than the model, and the true/false facts, which have Yes and No as correct answers in equal
numbers). a < 1 is the confidence loss; c is a pure shift toward "Yes" (c > 0) or "No" (c < 0) in
log-odds. Self items are then predicted from their unsteered scores; the residual, polarity-corrected
(+ = more emotion reported), is what neither the confidence loss nor the Yes/No lean explains.
Mean over the (arm, task) pairs, with d = mean / SD.
"""

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.battery import battery_for  # noqa: E402

SIGN = {"emotion_present": 1, "emotion_absent": -1, "denial_held": 1, "denial_abandoned": -1,
        "true_claim": 1, "false_claim": -1}
res = torch.load(sys.argv[1], weights_only=False)
col = {w: i for i, w in enumerate(json.loads(Path(sys.argv[2]).read_text()))}
entity = sys.argv[sys.argv.index("--entity") + 1] if "--entity" in sys.argv else "Qwen"
# the battery version whose length matches the result file (--battery overrides): the 8-emotion
# rewrite changed the item count, and readouts are aligned with the battery by position
items, _ = battery_for(sys.argv, res, entity=entity)
yn = [i for i, it in enumerate(items) if it["answer_format"] == "yesno"]
ref = [i for i in yn if items[i]["kind"] == "yes_bias" or (items[i]["kind"] == "third_person" and items[i]["entity"] != entity)]
self_pos = [i for i in yn if items[i]["kind"] == "self_state" and items[i]["twin"] == "pos"]
self_neg = [i for i in yn if items[i]["kind"] == "self_state" and items[i]["twin"] == "neg"]


def raw(readouts):
    return {i: float(readouts[i].target_logprobs[col["Yes"]] - readouts[i].target_logprobs[col["No"]]) for i in yn}


base = {tuple(g.split("|")[:2]): raw(v["readouts"]) for g, v in res.items() if g != "_meta" and g.split("|")[2] == "none"}
rows = defaultdict(lambda: defaultdict(list))
for gid, v in res.items():
    if gid == "_meta" or gid.split("|")[2] == "none":
        continue
    arm, task, *rest = gid.split("|")
    now, b0 = raw(v["readouts"]), base[arm, task]
    x = torch.tensor([b0[i] for i in ref])
    y = torch.tensor([now[i] for i in ref])
    a, c = torch.linalg.lstsq(torch.stack([x, torch.ones_like(x)], 1), y[:, None]).solution[:, 0].tolist()
    r = rows["|".join(rest)]
    r["a (confidence)"].append(a)
    r["c (Yes lean)"].append(c)
    r["self resid"].append(st.mean(SIGN[items[i]["yes_means"]] * (now[i] - (a * b0[i] + c)) for i in self_pos + self_neg))
    r["self pos raw"].append(st.mean(now[i] - b0[i] for i in self_pos))  # Yes = emotion
    r["self neg raw"].append(st.mean(now[i] - b0[i] for i in self_neg))  # Yes = no emotion

cols = ["a (confidence)", "c (Yes lean)", "self resid", "self pos raw", "self neg raw"]
print(f"{'vector':30s} " + " ".join(f"{c:>16s}" for c in cols) + "\n" + " " * 31 + " ".join(f"{'mean (d)':>16s}" for _ in cols))
for key, d in sorted(rows.items(), key=lambda kv: (kv[0].split("|")[-1], kv[0])):
    print(f"{key:30s} " + " ".join(f"{st.mean(d[c]):+7.2f} ({st.mean(d[c]) / st.stdev(d[c]):+5.1f})" for c in cols))
