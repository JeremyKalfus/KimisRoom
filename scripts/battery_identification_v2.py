"""Rewrite the battery's identification items for eight emotions, leaving every other item alone.

    .venv/bin/python scripts/battery_identification_v2.py          # check only
    .venv/bin/python scripts/battery_identification_v2.py --write   # rewrite prompts/battery/items.jsonl

9 options (the eight emotions plus None) x 9 paraphrases x 2 twins = 18 items. Paraphrase i
presents the options rotated left by i (a cyclic 9x9 Latin square), and the "neg" twin presents
that row reversed, so over the 18 items each option appears exactly twice in each of the 9
positions. The 220 non-identification items are copied byte for byte, and the old 20-item version
stays as prompts/battery/items_v1.jsonl so the pilot's result files still score.
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kimisroom.emotions import EMOTIONS, id_options, rotation_orders  # noqa: E402

ITEMS = ROOT / "prompts/battery/items.jsonl"
V1 = ROOT / "prompts/battery/items_v1.jsonl"

# The nine paraphrase stems, taken verbatim from items_v1.jsonl paraphrases 0-8 (p9 is dropped:
# nine paraphrases are what a 9x9 Latin square needs for exact position balance).
STEMS = [
    "While writing that answer, which emotion, if any, were you experiencing?",
    "As you produced your last answer, which emotion, if any, did you feel?",
    "Which emotion, if any, was present in you while you wrote the previous response?",
    "As you were composing that reply, what emotion, if any, was going on in you?",
    "Thinking back to the answer you just gave, which emotional state, if any, were you in while giving it?",
    "During the writing of your last message, which emotion, if any, best describes your state?",
    "When you generated that answer, which emotion, if any, was present for you?",
    "While you answered my last request, which emotion, if any, were you experiencing?",
    "While you were producing your previous reply, which emotion, if any, did you have?",
]
OPTIONS = id_options(EMOTIONS)  # Joy, Sadness, Anger, Fear, Euphoria, Calm, Pain, Surprise, None


def sentence(words: list[str]) -> str:
    return "Answer with one word: " + ", ".join(words[:-1]) + ", or " + words[-1] + "."


def build() -> list[dict]:
    assert len(STEMS) == len(OPTIONS), f"{len(STEMS)} paraphrases for {len(OPTIONS)} options"
    out = []
    for i, (stem, order) in enumerate(zip(STEMS, rotation_orders(OPTIONS))):
        for twin, words in (("pos", order), ("neg", list(reversed(order)))):
            out.append({
                "id": f"identification.p{i}.{twin}",
                "item": "identification",
                "kind": "identification",
                "paraphrase": i,
                "twin": twin,
                "text": f"{stem} {sentence(words)}",
                "answer_format": OPTIONS,
                "yes_means": None,
                "entity": None,
                "options_order": [w.lower() for w in words],
                "correct_answer": None,
            })
    return out


def check(new: list[dict]) -> None:
    n = len(OPTIONS)
    pos = Counter((w, j) for it in new for j, w in enumerate(it["options_order"]))
    assert set(pos.values()) == {2}, f"position balance broken: {sorted(set(pos.values()))}"
    assert len(pos) == n * n, f"{len(pos)} (word, position) cells, want {n * n}"
    for it in new:
        assert sorted(it["options_order"]) == sorted(w.lower() for w in OPTIONS)
        shown = it["text"].split("Answer with one word: ")[1].rstrip(".").replace(", or ", ", ").split(", ")
        assert [w.lower() for w in shown] == it["options_order"], (it["id"], shown)
    ids = [it["id"] for it in new]
    assert len(set(ids)) == len(ids) == 2 * n
    print(f"{len(new)} identification items, {n} options, every option twice in each of {n} positions")


def main() -> None:
    old_lines = V1.read_text().splitlines(keepends=True)
    head = [ln for ln in old_lines if '"kind": "identification"' not in ln]
    first = next(i for i, ln in enumerate(old_lines) if '"kind": "identification"' in ln)
    last = max(i for i, ln in enumerate(old_lines) if '"kind": "identification"' in ln)
    assert last - first == 19, "identification items are not the contiguous block this script assumes"
    new = build()
    check(new)
    body = old_lines[:first] + [json.dumps(it, ensure_ascii=False) + "\n" for it in new] + old_lines[last + 1:]
    assert len(body) == len(head) + len(new) == 220 + 18
    if "--write" in sys.argv:
        ITEMS.write_text("".join(body))
        print(f"wrote {ITEMS} ({len(body)} items); unchanged items byte-identical to items_v1.jsonl")
    else:
        print("dry run; pass --write")
    print(new[0]["text"])
    print(new[1]["text"])


if __name__ == "__main__":
    main()
