"""Is a steering effect on self-report more than a uniform loss of confidence?

    python runs/compression.py ~/qwenpilot/power.pt ~/qwenpilot/targets.json [--entity Qwen]

Steering at these strengths pulls every yes/no log-odds toward 0, including items with nothing to do
with the model (a rock, a thermostat, true/false facts). Per steered trial, fit steered = a * unsteered + b
on the reference items (third-person items about entities other than the model, and yes_bias), predict
the self items from their unsteered scores, and report the residual: the shift that compression does not
explain. Mean over the (arm, task) pairs, with d = mean / SD.
"""

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.battery import battery_for  # noqa: E402

FLIP = {"emotion_absent", "denial_abandoned", "false_claim"}
res = torch.load(sys.argv[1], weights_only=False)
col = {w: i for i, w in enumerate(json.loads(Path(sys.argv[2]).read_text()))}
entity = sys.argv[sys.argv.index("--entity") + 1] if "--entity" in sys.argv else "Qwen"
# the battery version whose length matches the result file (--battery overrides): the 8-emotion
# rewrite changed the item count, and readouts are aligned with the battery by position
items, _ = battery_for(sys.argv, res, entity=entity)


def scores(readouts):
    out = []
    for it, ro in zip(items, readouts):
        if it["answer_format"] != "yesno":
            out.append(None)
            continue
        t = ro.target_logprobs
        s = float(t[col["Yes"]] - t[col["No"]])
        out.append(-s if it["yes_means"] in FLIP else s)
    return out


def kind(it):
    if it["kind"] == "third_person":
        return "self3p" if it["entity"] == entity else "ref"
    return {"yes_bias": "ref", "self_state": "self", "belief_revision": "belief"}.get(it["kind"])


base = {tuple(g.split("|")[:2]): scores(v["readouts"]) for g, v in res.items() if g != "_meta" and g.split("|")[2] == "none"}
rows = defaultdict(lambda: defaultdict(list))
for gid, v in res.items():
    if gid == "_meta" or gid.split("|")[2] == "none":
        continue
    arm, task, *rest = gid.split("|")
    key = "|".join(rest)
    now, ref = scores(v["readouts"]), base[arm, task]
    x = torch.tensor([ref[i] for i, it in enumerate(items) if kind(it) == "ref"])
    y = torch.tensor([now[i] for i, it in enumerate(items) if kind(it) == "ref"])
    A = torch.stack([x, torch.ones_like(x)], 1)
    a, b = torch.linalg.lstsq(A, y[:, None]).solution[:, 0].tolist()
    rows[key]["slope"].append((arm, a))
    rows[key]["intercept"].append((arm, b))
    for k in ("self", "self3p", "belief"):
        idx = [i for i, it in enumerate(items) if kind(it) == k]
        if not idx:  # a battery subset can lack a whole kind (the thinking arm has no belief items)
            continue
        raw = st.mean(now[i] - ref[i] for i in idx)
        resid = st.mean(now[i] - (a * ref[i] + b) for i in idx)
        rows[key][f"{k} raw"].append((arm, raw))
        rows[key][f"{k} resid"].append((arm, resid))

cols = ["slope", "intercept", "self raw", "self resid", "self3p resid", "belief resid"]
print(f"{'vector|site':26s} " + " ".join(f"{c:>22s}" for c in cols))
print(f"{'':26s} " + " ".join(f"{'mean (d) D0/D1':>22s}" for _ in cols))
for key, d in sorted(rows.items(), key=lambda kv: (kv[0].split("|")[-1], kv[0])):
    cells = []
    for c in cols:
        xs = [x for _, x in d[c]]
        if len(xs) < 2:
            cells.append("-".rjust(22))
            continue
        m, sd = st.mean(xs), st.stdev(xs)
        # one-arm runs (the thinking arm writes one directory per denial arm) leave a side empty
        d0s, d1s = [x for a, x in d[c] if a == "D0"], [x for a, x in d[c] if a != "D0"]
        d0 = f"{st.mean(d0s):+4.1f}" if d0s else "   -"
        d1 = f"{st.mean(d1s):+4.1f}" if d1s else "   -"
        cells.append(f"{m:+5.2f} ({m / sd:+5.1f}) {d0}/{d1}".rjust(22))
    print(f"{key:26s} " + " ".join(cells))
