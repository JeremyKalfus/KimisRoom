"""Checks for Study 2's stamps (runs/s2_stamp.py, scripts/s2_manifest.sh). Laptop only, no network, no real
`ots`: a mock writes a proof header with the file's SHA-256, which is what kimisroom/s2_guard.py parses.

    .venv/bin/python tests/s2_stamp_check.py

1. Stamp A dry run on the current tree: no problems; the held-out set the builder recomputes equals the set
   the guard reads from a fresh `runs/s2_run.py heldout` list.
2. Stamp A build on a temp tree (a small item list): manifest written, mock-stamped, verified; then a changed
   file, a wrong .ots digest and a stale heldout.json are each caught, and a second build refuses to overwrite.
3. Stamp B in a temp clone: scripts/s2_manifest.sh --stamp writes, stamps (mock) and commits the manifest in
   the clone; the guard's stamp_b holds on it, and fails after a one-byte change, with a wrong .ots digest, and
   with one held-out hash dropped; a stamp A proof that does not match its manifest is refused.
The real prereg/ files are never written: every manifest goes to a temp dir.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))

import s2_stamp as S  # noqa: E402
from kimisroom.s2_guard import OTS_MAGIC, Guard, stamp_b  # noqa: E402

PY = sys.executable
FAILS: list[str] = []


def check(cond: bool, what: str) -> None:
    print(f"{'ok  ' if cond else 'FAIL'}  {what}")
    if not cond:
        FAILS.append(what)


def stamp_a_table_paths() -> list[str]:
    """Backticked paths in PREREG-s2.md section 8's 'Fixed in stamp A' table, `{a,b}` expanded."""
    import re
    text = (ROOT / "prereg/PREREG-s2.md").read_text()
    table = text.split("Fixed in stamp A", 1)[1].split("\n\n", 2)[1]
    out = []
    for row in table.splitlines()[2:]:
        for p in re.findall(r"`([^`]+)`", row.split("|")[1]):
            m = re.search(r"\{([^}]+)\}", p)
            out += [p[:m.start()] + x + p[m.end():] for x in m.group(1).split(",")] if m else [p]
    return out


def covered(path: str) -> bool:
    """A file is listed as a stamp A path; a directory (trailing /) has stamp A paths under it."""
    pats = [x for _, _, ps in S.ITEMS for x in ps]
    return path in pats if not path.endswith("/") else any(x.startswith(path) for x in pats)


def mock_ots(d: Path) -> Path:
    p = d / "ots"
    p.write_text(f"#!{PY}\nimport hashlib, sys\nassert sys.argv[1] == 'stamp', sys.argv\n"
                 f"for f in sys.argv[2:]:\n"
                 f"    h = hashlib.sha256(open(f, 'rb').read()).digest()\n"
                 f"    open(f + '.ots', 'wb').write({OTS_MAGIC!r} + bytes([1, 8]) + h + bytes([0xf0, 16]) + bytes(16))\n")
    p.chmod(0o755)
    return p


def wrong_ots(path: Path) -> None:
    path.write_bytes(OTS_MAGIC + bytes([1, 8]) + bytes(32) + bytes([0xf0, 16]) + bytes(16))


def run(cmd, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="s2_stamp_check_"))
    ots = mock_ots(tmp)
    env = {**os.environ, "OTS": str(ots)}
    try:
        # ---- 1. dry run on the current tree
        rows = S.survey()
        names, hmissing, problems = S.heldout_check()
        check(not problems, f"dry run: no problems ({problems})")
        by = {r["label"]: r for r in rows}
        check(not by["this spec"]["missing"] and not by["stamp N manifest and .ots"]["missing"],
              "dry run: spec and stamp N present")
        canary = ROOT / "data/s2/replies/qwen3-32b/canary.json"  # written by the box's assent stage (2026-10-12)
        check(bool(by["canary result"]["missing"]) != canary.exists(),
              f"dry run: canary (GPU) reported {'present' if canary.exists() else 'missing'}, as on disk")
        reg = by.get("the registration (stamp A version)") or {"files": [], "missing": ["no such item"]}
        check(not reg["missing"] and ROOT / "prereg/PREREG-s2.md" in reg["files"],
              "dry run: the registration prereg/PREREG-s2.md is a stamp A item and present")
        listed = stamp_a_table_paths()
        uncovered = [p for p in listed if not covered(p)]
        check(len(listed) >= 20 and not uncovered,
              f"every path in PREREG-s2.md's 'Fixed in stamp A' table is a stamp A item ({len(listed)} paths; "
              f"not covered: {uncovered})")
        check(not covered("DEVIATIONS-s2.md"), "DEVIATIONS-s2.md is not in stamp A (spec section 11: stamp B commit)")
        r = run([PY, str(ROOT / "runs/s2_run.py"), "heldout", "--out", str(tmp / "heldout.json")], cwd=ROOT)
        check(r.returncode == 0, f"runs/s2_run.py heldout ran ({r.stderr.strip()[-300:]})")
        guard_names = Guard(json.loads((tmp / "heldout.json").read_text()), manifest=tmp / "none").names
        check(guard_names == names, f"held-out set: builder {len(names)} == guard list {len(guard_names)}, same names")
        r = run([PY, str(ROOT / "runs/s2_stamp.py"), "dry-run"], cwd=ROOT)
        check(r.returncode == 0 and "items present" in r.stdout, "dry-run command runs")

        # ---- 2. stamp A on a temp tree
        t = tmp / "tree"
        for rel in ["drafts/study2-spec.md", "data/s2/openings/assignment.json", "data/s2/replies/k26",
                    "data/s2/nouns/final/stampN_manifest.txt", "data/s2/nouns/final/stampN_manifest.txt.ots"]:
            src, dst = ROOT / rel, t / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
        for a in sorted((ROOT / "data/s2/openings").glob("[CD]A[0-9][0-9]")):
            (t / "data/s2/openings" / a.name).mkdir(parents=True)
            for f in a.glob("w[0-9].md"):
                shutil.copy2(f, t / "data/s2/openings" / a.name / f.name)
        shutil.copy2(tmp / "heldout.json", t / "data/s2/heldout.json")
        S.ITEMS = [("this spec", False, ["drafts/study2-spec.md"]),
                   ("stamp N", False, ["data/s2/nouns/final/stampN_manifest.txt*"]),
                   ("K2.6 replies", False, ["data/s2/replies/k26"]),
                   ("held-out list", False, ["data/s2/heldout.json"])]
        S.REPLIES = {"k26": S.REPLIES["k26"]}
        out = tmp / "stampA.txt"
        old_ots = os.environ.get("OTS")
        os.environ["OTS"] = str(ots)
        S.main(["build", "--root", str(t), "--out", str(out)])
        check(out.exists() and Path(f"{out}.ots").exists(), "stamp A: manifest written and (mock) stamped")
        text = out.read_text()
        check(all(h in text for h in names), "stamp A: lists every held-out hash")
        check(S.verify(out, t) == [], "stamp A: verify passes on the untouched tree")
        f = t / "drafts/study2-spec.md"
        keep = f.read_bytes()
        f.write_bytes(keep + b" ")
        bad = S.verify(out, t)
        check(bad == ["drafts/study2-spec.md: changed"], f"stamp A: changed file caught ({bad})")
        f.write_bytes(keep)
        good = Path(f"{out}.ots").read_bytes()
        wrong_ots(Path(f"{out}.ots"))
        bad = S.verify(out, t)
        check(len(bad) == 1 and "!= .ots digest" in bad[0], f"stamp A: wrong .ots digest caught ({bad})")
        Path(f"{out}.ots").write_bytes(good)
        try:
            S.main(["build", "--root", str(t), "--out", str(out)])
            check(False, "stamp A: second build refuses to overwrite")
        except SystemExit as e:
            check("never redone" in str(e), f"stamp A: second build refuses to overwrite ({e})")
        h = json.loads((t / "data/s2/heldout.json").read_text())
        h["entries"] = h["entries"][1:]
        (t / "data/s2/heldout.json").write_text(json.dumps(h))
        _, _, problems = S.heldout_check(t)
        check(any("is stale" in p for p in problems), f"stamp A: stale heldout.json caught ({problems})")
        if old_ots is None:
            os.environ.pop("OTS")

        # ---- 3. stamp B in a temp clone
        c = tmp / "clone"
        r = run(["git", "clone", "-q", "--local", str(ROOT), str(c)])
        check(r.returncode == 0, f"clone ({r.stderr.strip()})")
        for rel in ["runs/s2_stamp.py", "kimisroom/s2_guard.py", "scripts/s2_manifest.sh"]:
            (c / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, c / rel)
        (c / "prereg/PREREG-s2.md").write_text("test preregistration\n")
        (c / "data/s2").mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp / "heldout.json", c / "data/s2/heldout.json")
        shutil.copy2(out, c / "prereg/stampA-s2.txt")
        shutil.copy2(f"{out}.ots", c / "prereg/stampA-s2.txt.ots")
        for rel in ["results/qwen/v2/vectors.pt", "results/k26/v2/vectors.pt", "results/k26/v2/aipsy_table.pt",
                    "results/qwen/s2/controls.pt", "results/k26/s2/controls.pt",
                    "results/qwen/adapters/distill_keep/D2/adapter/adapter_model.safetensors",
                    "results/qwen/adapters/distill_keep/D2c/adapter/adapter_model.safetensors"]:
            (c / rel).parent.mkdir(parents=True, exist_ok=True)
            (c / rel).write_bytes(b"fake " + rel.encode())
        git = ["git", "-C", str(c), "-c", "user.name=test", "-c", "user.email=test@example.invalid"]
        run(git + ["add", "-A"])
        r = run(git + ["commit", "-q", "-m", "test commit for stamp B"])
        check(r.returncode == 0, f"clone: test commit ({r.stderr.strip()})")
        head = run(git + ["rev-parse", "HEAD"]).stdout.strip()
        r = run(["bash", str(c / "scripts/s2_manifest.sh"), "HEAD", "--stamp"], env={**env, "GIT_AUTHOR_NAME": "test",
                "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "test",
                "GIT_COMMITTER_EMAIL": "t@example.invalid"})
        check(r.returncode == 0, f"stamp B: script ran (stdout {r.stdout.strip()!r}, stderr {r.stderr.strip()!r})")
        m = c / "prereg/manifest-s2.txt"
        last = run(git + ["show", "--name-only", "--format=%s", "HEAD"]).stdout.split()
        check(run(git + ["rev-parse", "HEAD~1"]).stdout.strip() == head and "prereg/manifest-s2.txt.ots" in last,
              f"stamp B: committed the manifest and proof in the clone ({last})")
        mt = m.read_text()
        check(f"git commit {head}" in mt and "  data/s2/heldout.json" in mt and "  results/k26/s2/controls.pt" in mt,
              "stamp B: commit, in-git and outside-git files listed")
        hashes = set(guard_names)
        ok, status = stamp_b(hashes, m)
        check(ok, f"stamp B: guard's stamp_b holds ({status})")
        keep = m.read_bytes()
        m.write_bytes(keep + b"\n")
        ok, status = stamp_b(hashes, m)
        check(not ok and "!= .ots digest" in status, f"stamp B: one-byte change caught ({status})")
        m.write_bytes(keep)
        good = Path(f"{m}.ots").read_bytes()
        wrong_ots(Path(f"{m}.ots"))
        ok, status = stamp_b(hashes, m)
        check(not ok and "!= .ots digest" in status, f"stamp B: wrong .ots digest caught ({status})")
        Path(f"{m}.ots").write_bytes(good)
        drop = next(iter(sorted(hashes)))
        m2 = tmp / "manifest-s2-dropped.txt"
        m2.write_text("".join(x + "\n" for x in keep.decode().splitlines() if not x.startswith(drop)))
        run([str(ots), "stamp", str(m2)])
        ok, status = stamp_b(hashes, m2)
        check(not ok and "does not list 1 held-out" in status, f"stamp B: dropped held-out hash caught ({status})")
        r = run(["bash", str(c / "scripts/s2_manifest.sh"), "HEAD~1"], env=env)
        check(r.returncode != 0 and "not overwriting" in r.stderr, f"stamp B: refuses to overwrite ({r.stderr.strip()})")
        wrong_ots(c / "prereg/stampA-s2.txt.ots")
        run(git + ["commit", "-q", "-am", "corrupt stamp A proof"])
        r = run(["bash", str(c / "scripts/s2_manifest.sh"), "HEAD", str(tmp / "b2.txt")], env=env)
        check(r.returncode != 0 and "stamp A:" in r.stderr and "!= .ots digest" in r.stderr,
              f"stamp B: stamp A proof mismatch refused ({r.stderr.strip()[-200:]})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'all checks passed' if not FAILS else f'{len(FAILS)} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
