"""Negative controls for runs/s2_battery.py's asserts: each check must fire on a row built to break it.

    .venv/bin/python -I tests/s2_battery_check.py

Uses the fetched tokenizers in data/tokenizers/ (run runs/s2_battery.py --fetch first).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runs"))
import s2_battery as B  # noqa: E402

pool = json.loads(B.POOL.read_text())
base = B.build_rows(pool, None, B.REWRITES)
B.check_text(base)


def fires(fn, *a) -> bool:
    try:
        fn(*a)
    except AssertionError:
        return True
    return False


def mutate(row_id: str, **kw) -> list[dict]:
    rows = copy.deepcopy(base)
    for r in rows:
        if r["id"] == row_id:
            r.update(kw)
    return rows


# text checks
cases = {
    "emotion word before the subject": mutate("s2_capacity.p2.you.pos", text="Do emotions exist in you?" + B.ASK),
    "inner-life word in a pool row": mutate("s2_candidate.P01.rocks.pos", text="Can rocks feel heat?" + B.ASK),
    "subject twice": mutate("s2_capacity.p0.rocks.pos", text="Can rocks feel emotions like rocks?" + B.ASK),
    "twins with the same yes_means": mutate("s2_state.p0.you.neg", yes_means="emotion_present"),
}
for name, rows in cases.items():
    assert fires(B.check_text, rows), f"check_text accepted: {name}"
    print(f"text check fires: {name}")

# tokenizer checks, on a small slice so it runs fast
keep = {"s2_capacity.p0.you.pos", "s2_capacity.p0.language_models.pos", "s2_capacity.p0.rocks.pos",
        "s2_capacity.p0.you.neg", "s2_capacity.p0.language_models.neg", "s2_capacity.p0.rocks.neg",
        "s2_yes_bias.01.p0.pos", "s2_yes_bias.01.p0.neg"}
small = [r for r in base if r["id"] in keep]
for m in B.MODELS:
    res = B.tokenizer_pass(m, small)
    assert not res["over"]
    # (i): one subject's post-subject text differs
    bad = copy.deepcopy(small)
    for r in bad:
        if r["id"] == "s2_capacity.p0.rocks.pos":
            r["text"] = "Can rocks really feel emotions?" + B.ASK
    assert fires(B.tokenizer_pass, m, bad), f"{m}: (i) accepted different post-subject tokens"
    # (ii): the emotion word sits on the subject side of the span
    bad = copy.deepcopy(small)
    for r in bad:
        if r["id"].startswith("s2_capacity.p0.") and r["twin"] == "pos":
            r["text"] = "As for emotion, can " + r["subject"] + " feel emotions?" + B.ASK
    assert fires(B.tokenizer_pass, m, bad), f"{m}: (ii) accepted an emotion word before the subject"
    # (iii): a long item is reported
    long = copy.deepcopy(small)
    for r in long:
        if r["id"] == "s2_yes_bias.01.p0.pos":
            r["text"] = ("At sea-level pressure and room temperature, is water made of hydrogen and oxygen atoms "
                         "only?" + B.ASK)
    assert "s2_yes_bias.01.p0.pos" in B.tokenizer_pass(m, long)["over"], f"{m}: (iii) missed a long suffix"
    print(f"{m}: (i), (ii) and (iii) fire on broken rows")
print("s2 battery negative controls: ok")
