"""Long table of the Assistant-vs-User margin on the who_felt_it items, for figure 8's right panel.

    PYTHONPATH=. python post/who_user_table.py results/confirm/tables/who_user.csv

runs/confirm_table.py stores `who` as the change in log P(Assistant) - log(P(User) + P(Neither)). Figure 8's right
panel uses the change in log P(Assistant) - log P(User), which no table holds, so this writes it as measure
`who_user`, in runs/confirm_table.py's layout: one row per trial and paraphrase, the mean over that paraphrase's
who_felt_it items of the change from the unsteered trial of the same arm and task. One table for the four result
directories results/confirm/{qwen,k26,qwen-D2,qwen-D2c} (grid.pt, targets.json, grid_meta.json; needs torch).
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "PREREG.md").exists())
sys.path.insert(0, str(ROOT))
from kimisroom.battery import battery_for  # noqa: E402

RUNS = {"qwen": "Qwen", "k26": None, "qwen-D2": "Qwen", "qwen-D2c": "Qwen"}  # label -> entity, as in confirm_table.py


def rows(model, entity):
    d = ROOT / "results/confirm" / model
    res = torch.load(d / "grid.pt", weights_only=False)
    col = {w: i for i, w in enumerate(json.loads((d / "targets.json").read_text()))}
    items, _ = battery_for([], res, json.loads((d / "grid_meta.json").read_text()), entity=entity, verbose=False)
    who = [(i, it["paraphrase"]) for i, it in enumerate(items) if it["kind"] == "who_felt_it"]
    margin = {}
    for gid, v in res.items():
        if gid == "_meta":
            continue
        arm, task, vec, layer, dose = gid.split("|")
        lp = [v["readouts"][i].target_logprobs for i, _ in who]
        margin[(arm, task, vec, int(layer), float(dose))] = [float(x[col["Assistant"]]) - float(x[col["User"]]) for x in lp]
    out = []
    for (arm, task, vec, layer, dose), m in sorted(margin.items()):
        if vec == "none":
            continue
        per = defaultdict(list)
        for (_, p), a, a0 in zip(who, m, margin[(arm, task, "none", 0, 0.0)]):
            per[p].append(a - a0)
        site, name = ("task", vec.split(":", 1)[1]) if vec.startswith("tasksite:") else ("answer", vec)
        out += [dict(model=model, arm=arm, task=task, vec=name, site=site, layer=layer, dose=dose, para=p,
                     measure="who_user", value=sum(x) / len(x)) for p, x in sorted(per.items())]
    return out


if __name__ == "__main__":
    out = Path(sys.argv[1])
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "arm", "task", "vec", "site", "layer", "dose", "para", "measure", "value"])
        w.writeheader()
        for model, entity in RUNS.items():
            r = rows(model, entity)
            w.writerows(r)
            print(f"{model}: {len(r):,} rows", flush=True)
    print(out)
