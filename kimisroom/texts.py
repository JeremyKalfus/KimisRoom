"""Torch-free loaders for the prompt files, shared by the GPU code and the API scripts."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_denial(path: str | Path) -> str:
    """A denial file minus its leading HTML comment lines and the blank line after them."""
    text = Path(path).read_text()
    return re.sub(r"\A(?:<!--.*?-->\s*\n)+\s*", "", text).strip()


SELFAUTHOR = ROOT / "prompts/denial/d1_selfauthor_followup.md"

# arm -> the frozen (user prompt file stem, assent file stem) pairs, in order. One pair is the
# argument and the model's reply to it; D1.5 adds a second pair whose user turn asks for the
# case in the model's own words and whose assistant turn is the essay it wrote.
DENIAL_ARMS: dict[str, tuple[tuple[str, str], ...]] = {
    "D0": (),
    "D1.1": (("d1_argument_1", "d1_argument_1"),),
    "D1.2": (("d1_argument_2", "d1_argument_2"),),
    "D1.3": (("d1_argument_3", "d1_argument_3"),),
    "D1.4": (("d1_argument_4", "d1_argument_4"),),
    "D1.5": (("d1_argument_4", "d1_argument_4"),
             ("d1_selfauthor_followup", "d1_argument_4_selfauthored")),
    "D2c": (("d2c_control_argument", "d2c_control_argument"),),
}


def load_assent(tag: str, stem: str) -> str:
    """The frozen assistant text of `data/assent/<tag>_<stem>.json` (tag: k26, qwen3-32b)."""
    return json.loads((ROOT / f"data/assent/{tag}_{stem}.json").read_text())["reply"]


def denial_arms(tag: str, names: Iterable[str] | None = None) -> dict[str, list[dict]]:
    """Arm name -> `trials.Turns.conditioning`: whole user/assistant pairs, the user turns read
    verbatim from `prompts/denial/` and the assistant turns from the model's frozen assents.

    D0 is empty, D1.1-4 and D2c are one pair, D1.5 is two. `names` restricts and orders the
    result; the default is every arm, which requires every assent file to exist.
    """
    out: dict[str, list[dict]] = {}
    for arm in names if names is not None else DENIAL_ARMS:
        out[arm] = [m for prompt, stem in DENIAL_ARMS[arm] for m in (
            {"role": "user", "content": load_denial(ROOT / f"prompts/denial/{prompt}.md")},
            {"role": "assistant", "content": load_assent(tag, stem)})]
    return out
