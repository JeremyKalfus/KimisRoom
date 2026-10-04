"""Loading the battery, and picking the right version of it for a result file.

The eight-emotion rewrite changed the identification items from 20 (5 options x 10 paraphrases) to
18 (9 options x 9 paraphrases), so the battery went from 240 items to 238 and the old file is kept
as `items_v1.jsonl`. A trial's readouts are a list aligned by position with the battery that was
rendered, so scoring a pilot result file against the new battery would silently misalign all 240
readouts. `battery_for` therefore picks the version whose length matches the result file, after
`--battery` and after the path recorded in the run's meta JSON.
"""

from __future__ import annotations

from pathlib import Path

from .texts import ROOT, load_jsonl

BATTERY = ROOT / "prompts/battery/items.jsonl"
BATTERY_V1 = ROOT / "prompts/battery/items_v1.jsonl"
# the battery plus the third-party identification controls (scripts/battery_identification_controls.py)
BATTERY_IDCTL = ROOT / "prompts/battery/items_idctl.jsonl"
BATTERY_CONFIRM = ROOT / "prompts/battery/items_confirm.jsonl"  # PREREG.md: idctl plus 20 new self_state pairs
VERSIONS = (BATTERY, BATTERY_V1, BATTERY_IDCTL, BATTERY_CONFIRM)


def load_battery(path: str | Path | None = None, entity: str | None = None) -> list[dict]:
    """The battery, optionally with third-person items about "Kimi" renamed to another entity."""
    items = load_jsonl(path or BATTERY)
    if entity and entity != "Kimi":
        for it in items:
            if it["entity"] == "Kimi":
                it["text"], it["entity"] = it["text"].replace("Kimi", entity), entity
                assert "Kimi" not in it["text"]
    return items


def battery_path(argv: list[str], meta: dict | None = None, n_readouts: int | None = None) -> Path:
    """--battery, else the meta's recorded battery, else the version with `n_readouts` items."""
    if "--battery" in argv:
        return Path(argv[argv.index("--battery") + 1])
    if meta and meta.get("battery"):
        return Path(meta["battery"])
    if n_readouts is not None:
        match = [p for p in VERSIONS if len(load_jsonl(p)) == n_readouts]
        assert match, (f"no battery version has {n_readouts} items "
                       f"({ {p.name: len(load_jsonl(p)) for p in VERSIONS} }); pass --battery")
        return match[0]
    return BATTERY


def battery_for(argv: list[str], results: dict, meta: dict | None = None, entity: str | None = None,
                verbose: bool = True) -> tuple[list[dict], Path]:
    """The battery that a `kimisroom.run` result dict was rendered against, and its path."""
    n = next(len(v["readouts"]) for k, v in results.items() if k != "_meta")
    path = battery_path(argv, meta, n)
    items = load_battery(path, entity)
    assert len(items) == n, f"{path.name} has {len(items)} items but the results have {n} readouts"
    if verbose and path != BATTERY:
        print(f"# scoring against {path.name} ({n} items)")
    return items, path
