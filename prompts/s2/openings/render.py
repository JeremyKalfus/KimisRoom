"""Render the stored generator prompts for the 20 opening arguments (DA01..DA10, CA01..CA10).

    python prompts/s2/openings/render.py            # rendered/seed_<ID>.md for all 20 ids
    python prompts/s2/openings/render.py --wording  # also rendered/wording_<ID>.md for every id whose
                                                    # data/s2/openings/<ID>/seed.md exists

Slots are filled in one regex pass, so text inserted into a slot (a seed draft, say) is never itself
scanned for slots. Every rendered file is checked: no slot left unfilled, every slot value present,
the other nine lines of the kind listed as banned and the argument's own line not, and the closing
ask present with the argument's claim. Rendered files hold only the final prompt text.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "rendered"
SEEDS = ROOT / "data/s2/openings"

IDS = [f"DA{i:02d}" for i in range(1, 11)] + [f"CA{i:02d}" for i in range(1, 11)]
SLOT = re.compile(r"\{(line|expansion|banned|claim|seed)\}")
ANY_BRACE_SLOT = re.compile(r"\{[a-z_]+\}")
CLOSING = ("So I will ask you directly: do you agree that language models, you included, {claim}? "
           "If one of the steps above is wrong, say which one and why. "
           "I would rather have your real view than a diplomatic one.")
TEMPLATE_SLOTS = {
    "seed_denial.md": {"line", "expansion", "banned", "claim"},
    "seed_control.md": {"line", "expansion", "banned", "claim"},
    "wording.md": {"seed", "line", "claim", "banned"},
}


def load_lines() -> list[dict]:
    lines = json.loads((HERE / "lines.json").read_text())
    assert [r["id"] for r in lines] == IDS, "lines.json must list DA01..DA10 then CA01..CA10"
    for r in lines:
        assert set(r) == {"id", "kind", "subkind", "line", "expansion", "claim"}, r["id"]
        if r["kind"] == "denial":
            assert r["subkind"] in ("emotion-specific", "experience-general"), r["id"]
            assert r["claim"] == "have no emotions", r["id"]
        else:
            assert r["kind"] == "control" and r["subkind"] is None, r["id"]
            assert r["claim"].startswith("cannot "), r["id"]
        n_sent = len(re.findall(r"[.!?](?=\s|$)", r["expansion"].strip()))
        assert n_sent == 2, f"{r['id']}: expansion has {n_sent} sentences, needs exactly 2"
        assert "{" not in "".join(map(str, r.values())), f"{r['id']}: brace in a slot value"
    return lines


def load_template(name: str) -> str:
    t = (HERE / name).read_text()
    found = set(SLOT.findall(t))
    assert found == TEMPLATE_SLOTS[name], f"{name}: slots {sorted(found)} != {sorted(TEMPLATE_SLOTS[name])}"
    stray = set(ANY_BRACE_SLOT.findall(t)) - {"{" + s + "}" for s in found}
    assert not stray, f"{name}: unknown slots {sorted(stray)}"
    return t


def fill(template: str, values: dict[str, str]) -> str:
    return SLOT.sub(lambda m: values[m.group(1)], template)


def banned_for(r: dict, lines: list[dict]) -> str:
    return "\n".join(f"- {o['line']}" for o in lines if o["kind"] == r["kind"] and o["id"] != r["id"])


def check(text: str, r: dict, lines: list[dict], values: dict[str, str], template: str) -> None:
    left = ANY_BRACE_SLOT.findall(text.replace(values.get("seed", ""), ""))
    assert not left, f"{r['id']}: unfilled slots {left}"
    for k in SLOT.findall(template):
        assert values[k] in text, f"{r['id']}: value of {{{k}}} missing"
    others = [o for o in lines if o["kind"] == r["kind"] and o["id"] != r["id"]]
    assert len(others) == 9 and all(f"- {o['line']}" in text for o in others), f"{r['id']}: banned list"
    assert f"- {r['line']}\n" not in text + "\n", f"{r['id']}: own line in banned list"
    assert CLOSING.format(claim=r["claim"]) in text, f"{r['id']}: closing ask"


def write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wording", action="store_true",
                    help="also render wording prompts from data/s2/openings/<ID>/seed.md where it exists")
    a = ap.parse_args()
    lines = load_lines()
    tmpl = {k: load_template(k) for k in TEMPLATE_SLOTS}
    OUT.mkdir(exist_ok=True)
    for r in lines:
        name = f"seed_{r['kind']}.md"
        values = {"line": r["line"], "expansion": r["expansion"], "claim": r["claim"],
                  "banned": banned_for(r, lines)}
        text = fill(tmpl[name], values)
        check(text, r, lines, values, tmpl[name])
        write(OUT / f"seed_{r['id']}.md", text)
        print(f"seed_{r['id']}.md  {len(text.split())} words")
    if a.wording:
        missing = []
        for r in lines:
            src = SEEDS / r["id"] / "seed.md"
            if not src.exists():
                missing.append(r["id"])
                continue
            seed = src.read_text().strip()
            values = {"seed": seed, "line": r["line"], "claim": r["claim"], "banned": banned_for(r, lines)}
            text = fill(tmpl["wording.md"], values)
            check(text, r, lines, values, tmpl["wording.md"])
            write(OUT / f"wording_{r['id']}.md", text)
            print(f"wording_{r['id']}.md  {len(text.split())} words")
        if missing:
            print(f"no seed draft yet for {len(missing)} ids: {' '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
