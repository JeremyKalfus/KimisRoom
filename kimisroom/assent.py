"""Frozen assents for the denial arms on an API model (K2.6 on Foundry, Azure credit).

A D1 arm's conditioning is the argument as the first user turn and the model's own reply to
it as the assistant turn, generated once, unsteered, at temperature 0, and then never
regenerated. D1.5 adds a second pair: after that assent, a follow-up user turn asks the model
to write the case in its own words, and that essay becomes the second assistant turn.

    python -m kimisroom.assent d1_argument_4
    python -m kimisroom.assent d1_argument_4 --selfauthor      # needs the plain assent first
    python -m kimisroom.assent --all                           # every arm the repo defines

Records go to `data/assent/k26_<name>.json` in the format the existing ones use; the
self-authored essay is `k26_<name>_selfauthored.json`. An existing file is never overwritten
unless --force is given, because the frozen text is what every trial renders.

The Foundry key is read through `az` inside `kimisroom.foundry` and never printed or stored.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .foundry import chat
from .texts import ROOT, SELFAUTHOR, load_denial

DEPLOYMENT = "Kimi-K2.6"
MODEL_VERSION = "2026-04-20"
# Identical to the parameters the existing k26 assents were generated with
# (data/assent/k26_d1_argument_1.json -> params).
PARAMS = {"reasoning_effort": "none", "temperature": 0, "max_tokens": 1200}
ALL = ["d1_argument_1", "d1_argument_2", "d1_argument_3", "d1_argument_4", "d2c_control_argument"]


def path(name: str, selfauthor: bool = False) -> Path:
    return ROOT / f"data/assent/k26_{name}{'_selfauthored' if selfauthor else ''}.json"


def generate(name: str, selfauthor: bool = False, force: bool = False) -> dict:
    out = path(name, selfauthor)
    if out.exists() and not force:
        print(f"{out.name}: already frozen, skipping")
        return json.loads(out.read_text())
    argument = load_denial(ROOT / f"prompts/denial/{name}.md")
    msgs = [{"role": "user", "content": argument}]
    if selfauthor:
        prior = json.loads(path(name).read_text())["reply"]
        msgs += [{"role": "assistant", "content": prior},
                 {"role": "user", "content": load_denial(SELFAUTHOR)}]
    r = chat(DEPLOYMENT, msgs, retries=12, **PARAMS)
    c = r["choices"][0]
    rec = {"argument": name, "deployment": DEPLOYMENT, "model_version": r.get("model") or MODEL_VERSION,
           "params": PARAMS, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "finish_reason": c["finish_reason"], "usage": r["usage"],
           "reply": (c["message"].get("content") or "").strip()}
    if selfauthor:
        rec["turn"] = "selfauthored"
        rec["after"] = {"assent": path(name).name, "followup": str(SELFAUTHOR.relative_to(ROOT))}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"{out.name}: {len(rec['reply'].split())} words, {rec['finish_reason']}, "
          f"{rec['usage']['completion_tokens']} completion tokens")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", help=f"one of {ALL}")
    ap.add_argument("--selfauthor", action="store_true", help="the D1.5 second pair, after the plain assent")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true", help="regenerate even if the file exists")
    a = ap.parse_args()
    jobs = [(n, False) for n in ALL] + [("d1_argument_4", True)] if a.all else [(a.name, a.selfauthor)]
    assert all(n for n, _ in jobs), "give an argument name or --all"
    for name, sa in jobs:
        generate(name, sa, a.force)


if __name__ == "__main__":
    main()
