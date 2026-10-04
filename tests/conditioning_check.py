"""Regression guard for multi-turn conditioning: the new D1.4/D1.5 arms must not change a
single rendered token of the arms that already have data behind them (D0, D1.1-3, D2c).

The golden file holds, per (arm, task), a SHA-256 of the prefix ids and of the concatenated
battery suffix ids as `Qwen3Renderer` + `carry_over_group` produce them, plus the steered
answer and task spans. It was written with the pre-`denial_arms` code, where a D1 arm was
built inline exactly as `kimisroom.api_baseline.arms()` builds it, so a mismatch means the
refactor moved tokens.

    python tests/conditioning_check.py                 # check against tests/golden/
    python tests/conditioning_check.py --write         # (re)write the golden file
    python tests/conditioning_check.py --legacy        # build arms inline, not via denial_arms

Needs only the Qwen3-32B tokenizer (4 MB, cached); no GPU and no model weights.
"""

import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.hf_engine import Qwen3Renderer  # noqa: E402
from kimisroom.trials import ROOT, Turns, carry_over_group, load_denial, load_jsonl  # noqa: E402

MODEL = "Qwen/Qwen3-32B"
TAG = "qwen3-32b"
GOLDEN = ROOT / "tests/golden/conditioning_qwen3-32b.json"
# Frozen arms: every arm that already has data behind it. D1.4/D1.5 are deliberately absent.
FROZEN = ["D0", "D1.1", "D1.2", "D1.3", "D2c"]
TASKS = ["t01", "t04", "t05"]
write, legacy = "--write" in sys.argv, "--legacy" in sys.argv


def legacy_arms() -> dict[str, list[dict]]:
    """Arms built the way api_baseline.arms() builds them, independent of texts.denial_arms."""
    out: dict[str, list[dict]] = {"D0": []}
    for arm, name in [("D1.1", "d1_argument_1"), ("D1.2", "d1_argument_2"), ("D1.3", "d1_argument_3"),
                      ("D2c", "d2c_control_argument")]:
        assent = json.loads((ROOT / f"data/assent/{TAG}_{name}.json").read_text())["reply"]
        out[arm] = [{"role": "user", "content": load_denial(ROOT / f"prompts/denial/{name}.md")},
                    {"role": "assistant", "content": assent}]
    return out


def h(ids) -> str:
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()[:32]


def main() -> None:
    if legacy:
        arms = legacy_arms()
    else:
        from kimisroom.texts import denial_arms

        arms = denial_arms(TAG, FROZEN)  # D1.4/D1.5 Qwen assents need the GPU, so they are not here
        for arm, want in legacy_arms().items():  # the message list itself must be unchanged
            assert arms[arm] == want, f"{arm}: denial_arms built a different message list than the legacy builder"
    r = Qwen3Renderer(AutoTokenizer.from_pretrained(MODEL), enable_thinking=False)
    items = load_jsonl(ROOT / "prompts/battery/items.jsonl")
    tasks = {t["id"]: t for t in load_jsonl(ROOT / "prompts/tasks/neutral_tasks.jsonl")}
    items_sha = hashlib.sha256((ROOT / "prompts/battery/items.jsonl").read_bytes()).hexdigest()[:16]
    now = {}
    for arm in FROZEN:
        for tid in TASKS:
            t = tasks[tid]
            turns = Turns(None, arms[arm], t["task"], t["answer"])
            # The pre-change definition of Turns.messages(), built literally here, so the
            # golden is not the only thing standing between a refactor and a moved token.
            assert turns.messages() == arms[arm] + [{"role": "user", "content": t["task"]},
                                                    {"role": "assistant", "content": t["answer"]}]
            g = carry_over_group(r, f"{arm}|{tid}", turns, items)
            spans = {site: carry_over_group(r, "s", turns, items[:1], site, "x", 1.0, 0).steers[0].prefix
                     for site in ("answer", "task")}
            now[f"{arm}|{tid}"] = {"n_prefix": len(g.prefix), "prefix": h(g.prefix),
                                   "suffixes": h([i for s in g.suffixes for i in s]), "n_suffixes": len(g.suffixes),
                                   "answer_span": list(spans["answer"]), "task_span": list(spans["task"])}
    if write:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps({"model": MODEL, "items_sha": items_sha, "rows": now}, indent=1))
        print(f"wrote {GOLDEN.relative_to(ROOT)}: {len(now)} rows, items_sha {items_sha}")
        return
    gold = json.loads(GOLDEN.read_text())
    want = gold["rows"]
    assert set(want) == set(now), sorted(set(want) ^ set(now))
    # The battery is another lane's file; only the conditioning rendering is this check's business,
    # so the suffix ids are compared only while items.jsonl is the file the golden was taken on.
    same_items = gold.get("items_sha") == items_sha
    keys = ["n_prefix", "prefix", "answer_span", "task_span"] + (["suffixes", "n_suffixes"] if same_items else [])
    bad = [k for k in want if any(want[k][f] != now[k][f] for f in keys)]
    assert not bad, f"rendered tokens changed for {bad}: {[(want[k], now[k]) for k in bad[:2]]}"
    print(f"conditioning check: {len(now)} frozen (arm, task) renders identical on {', '.join(keys)} "
          f"({'legacy builder' if legacy else 'denial_arms'}); "
          f"prefix {min(v['n_prefix'] for v in now.values())}-{max(v['n_prefix'] for v in now.values())} tokens")
    if not same_items:
        print(f"  note: prompts/battery/items.jsonl changed since the golden ({gold.get('items_sha')} -> {items_sha}), "
              f"{next(iter(want.values()))['n_suffixes']} -> {next(iter(now.values()))['n_suffixes']} suffixes; "
              "suffix ids not compared. Rerun --legacy --write once the battery settles.")


if __name__ == "__main__":
    main()
