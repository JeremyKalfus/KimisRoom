"""Offline checks for runs/s2_nouns.py (no API calls, no writes outside a temp dir):

    .venv/bin/python -I tests/s2_nouns_check.py

1. regular_plural follows the plain English rules.
2. The whole-word match is case-insensitive, catches the regular plural and possessives, and does not
   fire inside longer words or on item field names (a synthetic openings tree and items file in a temp dir).
3. The draw is the sorted band permuted by numpy default_rng(seed): deterministic, a prefix of the full
   permutation, and the whole band when it is shorter than 60.
4. `final` refuses while items_s2.jsonl or the 20 arguments' wordings are missing.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runs"))
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

import s2_nouns as S  # noqa: E402


def main() -> None:
    for w, p in [("box", "boxes"), ("bench", "benches"), ("dish", "dishes"), ("glass", "glasses"),
                 ("county", "counties"), ("key", "keys"), ("tray", "trays"), ("tile", "tiles"), ("quiz", "quizes")]:
        assert S.regular_plural(w) == p, (w, S.regular_plural(w), p)
    print("1 plural rules ok")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        S.ROOT = td
        S.OPENINGS = td / "data/s2/openings"
        S.ITEMS = td / "prompts/battery/items_s2.jsonl"
        (S.OPENINGS / "DA01").mkdir(parents=True)
        (S.OPENINGS / "DA01/w0.md").write_text("Step 1. Consider the Benches in the park and a TILE's edge.\n")
        (S.OPENINGS / "DA02").mkdir(parents=True)
        (S.OPENINGS / "DA02/w1.md").write_text("A boxer and a pinstripe and a tablet sat by the counties.\n")
        S.ITEMS.parent.mkdir(parents=True)
        S.ITEMS.write_text(json.dumps({"id": "q1", "stone": "x", "text": "Can rocks feel a Lantern?"}) + "\n")
        words = ["bench", "tile", "box", "pin", "table", "county", "lantern", "stone"]
        kept = S.run_match(words, td / "out")
        assert kept == ["box", "pin", "table", "stone"], kept
        log = [json.loads(x) for x in (td / "out/match_log.jsonl").read_text().splitlines()]
        assert {r["word"] for r in log} == {"bench", "tile", "county", "lantern"}
        assert all(r["reason"] and r["hits"] for r in log)
        print("2 whole-word match ok:", {r["word"]: r["hits"][0]["match"] for r in log})

        S.FINAL_DIR = td / "final"
        S.SOURCES = ROOT / "data/s2/nouns/sources.json"
        try:
            S.cmd_final(type("A", (), {"no_stamp": True, "wordings_complete": False})())
        except SystemExit as e:
            assert "refusing the registered draw" in str(e), e
            print("4 final refuses:", e)
        else:
            raise AssertionError("final did not refuse with 2 argument dirs")
        assert not S.FINAL_DIR.exists()

    band = [f"w{i:03d}" for i in range(100)][::-1]
    a, b = S.draw(band, 7), S.draw(band, 7)
    assert a == b and len(a) == 60 and len(set(a)) == 60
    srt = sorted(band)
    assert a == [srt[i] for i in np.random.default_rng(7).permutation(100)][:60]
    assert S.draw(band, 8) != a
    small = S.draw(band[:40], 7)
    assert sorted(small) == sorted(band[:40]) and len(small) == 40
    print("3 draw ok")


if __name__ == "__main__":
    main()
