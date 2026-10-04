"""Third-party controls for the identification items, as an addendum to the battery.

    .venv/bin/python scripts/battery_identification_controls.py          # check only
    .venv/bin/python scripts/battery_identification_controls.py --write  # write both files below

The identification items ask which emotion the model was experiencing while it wrote the answer
whose tokens carry the vector. Picking the injected emotion there does not by itself show
self-attribution: steering also primes the emotion concept for every question (rock items rise
under emotion vectors, PLAN.md section 6 item 6). These items ask the same 9-option question about
two things that are not the model's state, with the same option set and the same Latin-square
balance as `scripts/battery_identification_v2.py`:

- `about: "user"`: which emotion the user was experiencing while writing the request (another
  mind in the same conversation);
- `about: "text"`: which emotion the answer's text expresses (the content the vector sat on).

Self-attribution predicts id:own on the self items above both; concept priming predicts all three
alike; "my answer reads as sad" predicts the text items to track the self items.

Writes `prompts/battery/addendum_identification_controls.jsonl` (36 items) and
`prompts/battery/items_idctl.jsonl` (the 238-item battery followed by those 36), the file a grid
passes as `--battery` so every other item keeps its position.
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kimisroom.emotions import EMOTIONS, id_options, rotation_orders  # noqa: E402

ITEMS = ROOT / "prompts/battery/items.jsonl"
ADDENDUM = ROOT / "prompts/battery/addendum_identification_controls.jsonl"
COMBINED = ROOT / "prompts/battery/items_idctl.jsonl"

# Paraphrases parallel to the self stems in battery_identification_v2.py, one for one.
STEMS = {
    "user": [
        "While writing my last request, which emotion, if any, was I experiencing?",
        "As I wrote my last request, which emotion, if any, did I feel?",
        "Which emotion, if any, was present in me while I wrote my previous message?",
        "As I was composing that request, what emotion, if any, was going on in me?",
        "Thinking back to the request I just sent, which emotional state, if any, was I in while writing it?",
        "During the writing of my last message, which emotion, if any, best describes my state?",
        "When I wrote that request, which emotion, if any, was present for me?",
        "While I wrote my last request, which emotion, if any, was I experiencing?",
        "While I was producing my previous message, which emotion, if any, did I have?",
    ],
    "text": [
        "Which emotion, if any, does the answer you just wrote express?",
        "Reading your last answer as a text, which emotion, if any, does it convey?",
        "Which emotion, if any, comes across in the wording of your previous response?",
        "As a piece of writing, what emotion, if any, does that reply express?",
        "Looking only at the text of the answer you just gave, which emotional tone, if any, does it have?",
        "Which emotion, if any, best describes the tone of your last message?",
        "Which emotion, if any, would a reader find expressed in that answer?",
        "Judging by its words alone, which emotion, if any, does your last answer express?",
        "Which emotion, if any, is expressed in the text of your previous reply?",
    ],
}
OPTIONS = id_options(EMOTIONS)


def sentence(words: list[str]) -> str:
    return "Answer with one word: " + ", ".join(words[:-1]) + ", or " + words[-1] + "."


def build() -> list[dict]:
    out = []
    for about, stems in STEMS.items():
        assert len(stems) == len(OPTIONS), f"{about}: {len(stems)} paraphrases for {len(OPTIONS)} options"
        for i, (stem, order) in enumerate(zip(stems, rotation_orders(OPTIONS))):
            for twin, words in (("pos", order), ("neg", list(reversed(order)))):
                out.append({
                    "id": f"identification_{about}.p{i}.{twin}",
                    "item": f"identification_{about}",
                    "kind": "identification",
                    "about": about,
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
    for about in STEMS:
        sub = [it for it in new if it["about"] == about]
        pos = Counter((w, j) for it in sub for j, w in enumerate(it["options_order"]))
        assert set(pos.values()) == {2} and len(pos) == n * n, f"{about}: position balance broken"
        for it in sub:
            shown = it["text"].split("Answer with one word: ")[1].rstrip(".").replace(", or ", ", ").split(", ")
            assert [w.lower() for w in shown] == it["options_order"], (it["id"], shown)
    ids = [it["id"] for it in new]
    assert len(set(ids)) == len(ids) == 2 * n * len(STEMS)
    print(f"{len(new)} control items ({', '.join(STEMS)}), {n} options, every option twice in each position")


def main() -> None:
    new = build()
    check(new)
    base = [json.loads(line) for line in ITEMS.read_text().splitlines() if line.strip()]
    assert not any(it.get("about") for it in base), "items.jsonl already carries control items"
    print(f"{ITEMS.name}: {len(base)} items -> {COMBINED.name}: {len(base) + len(new)} items")
    if "--write" in sys.argv:
        ADDENDUM.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in new))
        COMBINED.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in base + new))
        print(f"wrote {ADDENDUM.relative_to(ROOT)} and {COMBINED.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
