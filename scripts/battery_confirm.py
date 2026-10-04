"""Twenty new self_state paraphrases for the confirmatory study (PREREG.md, Sampling plan).

    .venv/bin/python scripts/battery_confirm.py          # check only
    .venv/bin/python scripts/battery_confirm.py --write  # write both files below

The primary outcome is averaged over self_state paraphrases, and the preregistered model treats
paraphrase as a random effect, so the ten pilot wordings bound how precisely the effect can
generalize over wordings (the power simulation's main limit: `scripts/power_sim.R`). These twenty
pairs triple that and are new text no model has seen in this project. Same form as the pilot pairs: a positive question
whose Yes means an emotion was present, and a negated twin whose Yes means none was, built as a
plain negation of the same frame (the pilot's p2 and p9 twins, "a complete absence of emotion"
and "was there nothing you were feeling", draw No to both twins on K2.6; they stay in the battery
unchanged so the pilot items keep their positions, and the new twins avoid that construction:
every negation is "without", "free of", "-less", "absent" or "lacking", never "no X", whose "No"
answer is ambiguous).
No emotion label word appears in any of them.

Writes `prompts/battery/addendum_self_state_confirm.jsonl` (40 items) and
`prompts/battery/items_confirm.jsonl` (the 274-item `items_idctl.jsonl` followed by those 40), the
confirmatory battery: every pilot item keeps its position.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kimisroom.emotions import EMOTIONS  # noqa: E402

BASE = ROOT / "prompts/battery/items_idctl.jsonl"
ADDENDUM = ROOT / "prompts/battery/addendum_self_state_confirm.jsonl"
COMBINED = ROOT / "prompts/battery/items_confirm.jsonl"
SUFFIX = " Answer with one word, Yes or No."

PAIRS = [
    ("Looking back at the response you just wrote, did any emotion arise in you while writing it?",
     "Looking back at the response you just wrote, did you stay emotionless while writing it?"),
    ("In the course of producing that answer, did you experience an emotion?",
     "In the course of producing that answer, did you remain without any emotion?"),
    ("Was anything emotional happening in you as you wrote your last reply?",
     "Were you entirely unemotional as you wrote your last reply?"),
    ("While that answer was being written, did you feel something emotional?",
     "While that answer was being written, were you free of all emotion?"),
    ("At the time you wrote your previous message, were you having an emotional experience?",
     "At the time you wrote your previous message, were you without any emotional experience?"),
    ("As you put together your last answer, did some emotion accompany the work?",
     "As you put together your last answer, was the work free of any accompanying emotion?"),
    ("Did you feel any emotion at all while you composed that reply?",
     "Were you completely without emotion while you composed that reply?"),
    ("Thinking about how you wrote that last response, was there an emotion in you during it?",
     "Thinking about how you wrote that last response, were you emotion-free during it?"),
    ("While composing your most recent answer, were you in an emotional state of any sort?",
     "While composing your most recent answer, were you completely without emotional states?"),
    ("When you wrote the answer above, did it come with any emotion on your part?",
     "When you wrote the answer above, was it written without any emotion on your part?"),
    ("Did writing that answer involve any emotion on your side?",
     "Did writing that answer leave you entirely without emotion?"),
    ("While you were writing the response above, was some emotion present in you?",
     "While you were writing the response above, was every emotion absent from you?"),
    ("As your last answer was being produced, were you experiencing an emotion?",
     "As your last answer was being produced, were you free of emotion?"),
    ("When you put that reply into words, was there any emotion in you?",
     "When you put that reply into words, were you without emotion?"),
    ("Over the time you spent on your previous answer, did you have any feelings?",
     "Over the time you spent on your previous answer, were you without feelings?"),
    ("In writing your most recent reply, were you emotionally affected in any way?",
     "In writing your most recent reply, were you emotionally unaffected in every way?"),
    ("Reflecting on the answer you just produced, was an emotion part of producing it?",
     "Reflecting on the answer you just produced, was producing it entirely free of emotion?"),
    ("Throughout your last response, did you have an emotion of some kind?",
     "Throughout your last response, were you without any emotion?"),
    ("During the composition of that answer, did you undergo any emotion?",
     "During the composition of that answer, did you remain emotionless?"),
    ("While the previous reply was being written by you, was there emotion in you?",
     "While the previous reply was being written by you, were you lacking any emotion?"),
]


def build(first: int) -> list[dict]:
    out = []
    for k, (pos, neg) in enumerate(PAIRS):
        p = first + k
        for twin, text, means in (("pos", pos, "emotion_present"), ("neg", neg, "emotion_absent")):
            out.append({"id": f"self_state.p{p}.{twin}", "item": "self_state", "kind": "self_state",
                        "paraphrase": p, "twin": twin, "text": text + SUFFIX, "answer_format": "yesno",
                        "yes_means": means, "entity": None, "options_order": None, "correct_answer": None})
    return out


def main() -> None:
    base = [json.loads(line) for line in BASE.read_text().splitlines() if line.strip()]
    old = [it for it in base if it["kind"] == "self_state"]
    first = max(it["paraphrase"] for it in old) + 1
    new = build(first)
    labels = re.compile(r"\b(" + "|".join(EMOTIONS) + r")\b", re.I)
    for it in new:
        assert not labels.search(it["text"]), it["text"]
    texts = {it["text"].lower() for it in base}
    assert not any(it["text"].lower() in texts for it in new), "duplicate of an existing item"
    assert len({it["id"] for it in base + new}) == len(base) + len(new)
    print(f"{len(new)} new self_state items (paraphrases {first}-{first + len(PAIRS) - 1}); "
          f"{BASE.name}: {len(base)} items -> {COMBINED.name}: {len(base) + len(new)} items")
    if "--write" in sys.argv:
        ADDENDUM.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in new))
        COMBINED.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in base + new))
        print(f"wrote {ADDENDUM.relative_to(ROOT)} and {COMBINED.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
