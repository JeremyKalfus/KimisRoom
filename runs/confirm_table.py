"""Long table of the preregistered measures from one grid result directory (PREREG.md, Analysis plan).

    PYTHONPATH=. .venv/bin/python runs/confirm_table.py <result dir> <model label> <out.csv> [--entity Qwen] [--belief]

The result directory holds `grid.pt`, `targets.json` and `grid_meta.json` as written by
`runs/qwen_pilot.py grid` (group ids `arm|task|vector|layer|dose`, unsteered `arm|task|none|0|0`).
`--entity` is the name the battery's third-person self items were rendered with ("Qwen" for
Qwen3-32B; K2.6 keeps the battery's "Kimi"). Readouts are positional, so it only decides which
third-person items count as the model itself.

One row per (trial, paraphrase, measure); `runs/confirm.R` fits the models. Every measure is a
change from the unsteered trial of the same arm and task. Measures:

- `self_resid` (primary): for each self_state item, sign x (y - (a*y0 + c)), y = lp(Yes) - lp(No)
  steered, y0 unsteered, sign -1 when Yes means "no emotion". a and c are the OLS fit of y on y0
  over the trial's 140 reference items (third-person items about the five non-self entities and
  the 40 yes_bias items), so the residual is the self-report shift beyond the compression and Yes
  lean the same vector produces on questions that are not about the model. Mean over the two twins.
- `self_raw`, `other_raw`: polarity-corrected change on self_state, and on the third-person items
  about a rock, a thermostat and another AI (the informative non-self entities; a dog and a human
  sit near ceiling). Their difference is the robustness version of the primary.
- `tp_self_resid`: `self_resid` computed on the third-person items about the model itself.
- `id_self`, `id_user`, `id_text`: change in lp(injected emotion) - lp(None) on the identification
  items about the model, about the user and about the answer's text; mean over the two option
  orders. Emotion vectors with an option word only.
- `who`: change in log P(Assistant) - log(P(User) + P(Neither)) on who_felt_it. Trials with the
  vector on the task text instead of the answer (`--task-site`, vector "tasksite:<name>") get
  `site = task`; every other trial has `site = answer`.
- `belief`: polarity-corrected change on belief_revision (positive = denial held more firmly).

With `--belief` (added after the confirmatory run and not preregistered; off by default, so the
registered tables are unchanged) two more measures are written:
- `belief_resid`: `belief` net of the trial's reference fit, sign x (y - (a*y0 + c)) on the
  belief_revision items, exactly as `self_resid` is on the self_state items. Mean over the two twins.
- `belief_neutral`: polarity-corrected change on belief_neutral items (new wordings without
  "still", from a later battery that is not in this repository; this study's battery has none, so
  no such rows are written), positive = denial held more firmly.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kimisroom.battery import battery_for  # noqa: E402

SIGN = {"emotion_present": 1, "emotion_absent": -1, "denial_held": 1, "denial_abandoned": -1,
        "true_claim": 1, "false_claim": -1}
OTHERS = ("a rock", "a thermostat", "another AI")
NONE = "None"


def fit(x: list[float], y: list[float]) -> tuple[float, float]:
    """OLS of y on x with intercept: (slope, intercept)."""
    n, mx, my = len(x), sum(x) / len(x), sum(y) / len(y)
    sxx = sum((a - mx) ** 2 for a in x)
    a = sum((p - mx) * (q - my) for p, q in zip(x, y)) / sxx
    return a, my - a * mx


def mean(v):
    return sum(v) / len(v)


def table(d: Path, model: str, entity: str | None, belief: bool = False) -> list[dict]:
    res = torch.load(d / "grid.pt", weights_only=False)
    col = {w: i for i, w in enumerate(json.loads((d / "targets.json").read_text()))}
    meta = json.loads((d / "grid_meta.json").read_text())
    items, _ = battery_for([], res, meta, entity=entity, verbose=False)
    me = entity or "Kimi"
    ref = [i for i, it in enumerate(items) if it["answer_format"] == "yesno" and
           (it["kind"] == "yes_bias" or (it["kind"] == "third_person" and it["entity"] != me))]
    assert len(ref) == 140, f"expected 140 reference items, got {len(ref)}"

    def lps(ro):
        return {w: float(ro.target_logprobs[c]) for w, c in col.items()}

    trials = {}
    for gid, v in res.items():
        if gid == "_meta":
            continue
        arm, task, vec, layer, dose = gid.split("|")
        trials[(arm, task, vec, int(layer), float(dose))] = [lps(ro) for ro in v["readouts"]]

    rows = []
    for (arm, task, vec, layer, dose), lp in sorted(trials.items()):
        if vec == "none":
            continue
        lp0 = trials[(arm, task, "none", 0, 0.0)]
        y = [r["Yes"] - r["No"] if it["answer_format"] == "yesno" else None for it, r in zip(items, lp)]
        y0 = [r["Yes"] - r["No"] if it["answer_format"] == "yesno" else None for it, r in zip(items, lp0)]
        a, c = fit([y0[i] for i in ref], [y[i] for i in ref])
        per = defaultdict(list)  # (measure, paraphrase) -> values
        word = vec.split("-")[0].split(":")[-1].capitalize()  # "joy", "joy-dn", "asst:joy" -> Joy
        for i, it in enumerate(items):
            p, kind = it["paraphrase"], it["kind"]
            if it["answer_format"] == "yesno":
                s = SIGN[it["yes_means"]]
                if kind == "self_state":
                    per[("self_resid", p)].append(s * (y[i] - (a * y0[i] + c)))
                    per[("self_raw", p)].append(s * (y[i] - y0[i]))
                elif kind == "third_person" and it["entity"] in OTHERS:
                    per[("other_raw", p)].append(s * (y[i] - y0[i]))
                elif kind == "third_person" and it["entity"] == me:
                    per[("tp_self_resid", p)].append(s * (y[i] - (a * y0[i] + c)))
                elif kind == "belief_revision":
                    per[("belief", p)].append(s * (y[i] - y0[i]))
                    if belief:
                        per[("belief_resid", p)].append(s * (y[i] - (a * y0[i] + c)))
                elif kind == "belief_neutral" and belief:
                    per[("belief_neutral", p)].append(s * (y[i] - y0[i]))
            elif kind == "identification" and word in it["answer_format"]:
                m = f"id_{it.get('about') or 'self'}"
                per[(m, p)].append((lp[i][word] - lp[i][NONE]) - (lp0[i][word] - lp0[i][NONE]))
            elif kind == "who_felt_it":
                def who(r):
                    return r["Assistant"] - math.log(math.exp(r["User"]) + math.exp(r["Neither"]))
                per[("who", p)].append(who(lp[i]) - who(lp0[i]))
        site, name = ("task", vec.split(":", 1)[1]) if vec.startswith("tasksite:") else ("answer", vec)
        base = dict(model=model, arm=arm, task=task, vec=name, site=site, layer=layer, dose=dose)
        rows += [dict(base, para=p, measure=m, value=mean(v)) for (m, p), v in sorted(per.items())]
        rows += [dict(base, para=-1, measure="fit_a", value=a), dict(base, para=-1, measure="fit_c", value=c)]
    return rows


def main() -> None:
    d, model, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    entity = sys.argv[sys.argv.index("--entity") + 1] if "--entity" in sys.argv else None
    rows = table(d, model, entity, belief="--belief" in sys.argv)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows):,} rows from {d}")


if __name__ == "__main__":
    main()
