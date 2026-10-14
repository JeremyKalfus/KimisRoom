"""Study 2 stamp A (drafts/study2-spec.md section 11; section 12 step 10): the manifest stamped before any
steered pilot trial, in stamp N's format (data/s2/nouns/final/stampN_manifest.txt): `# ` header lines, then
`<sha256>  <path>` per file, then the held-out list as `<sha256>  heldout <opening> <what>` lines.

    .venv/bin/python runs/s2_stamp.py dry-run             # what is present and what is missing; writes nothing
    .venv/bin/python runs/s2_stamp.py build               # refuses unless nothing is missing; then `ots stamp`
    .venv/bin/python runs/s2_stamp.py verify [MANIFEST]   # re-hash every listed file, check the .ots digest
    .venv/bin/python runs/s2_stamp.py heldout-lines FILE  # the held-out hash lines of a heldout.json (stamp B)

Held-out hashes use exactly the guard's definition (kimisroom/s2_guard.py: `text_sha`, `opening_sha`, and
`Guard(...).names` for the set a heldout.json yields). The builder recomputes them from the assignment, the
accepted wordings and each model's selected.json, and refuses to stamp unless data/s2/heldout.json (the
guard's list, `runs/s2_run.py heldout`) yields the same set. A model with no selected.json yet is reported
missing. `ots` is taken from $OTS if set (tests point it at a mock).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kimisroom.s2_guard import Guard, HeldOutError, opening_sha, ots_digest, text_sha  # noqa: E402
from kimisroom.texts import load_denial  # noqa: E402

MANIFEST = ROOT / "prereg/stampA-s2.txt"
HELDOUT = ROOT / "data/s2/heldout.json"
ASSIGNMENT = ROOT / "data/s2/openings/assignment.json"
REPLIES = {"qwen3-32b": ROOT / "data/s2/replies/qwen3-32b", "k26": ROOT / "data/s2/replies/k26"}
SKIP = {"__pycache__", ".DS_Store"}

# (label, GPU session needed, paths). A path is a file, a directory (every file under it), or a glob.
# Every path must exist (a directory or glob must yield at least one file) for the item to be present.
ITEMS: list[tuple[str, bool, list[str]]] = [
    ("this spec", False, ["drafts/study2-spec.md"]),
    ("the registration (stamp A version)", False, ["prereg/PREREG-s2.md"]),
    ("the rules (pilot selection code; selection output)", True,
     ["runs/s2_select.py", "prompts/battery/s2_selection.json"]),
    ("unsteered pilot pass read by the selection (step 7)", True,
     ["results/s2/qwen/unsteered/S2-DEN/readouts.csv", "results/s2/k26/unsteered/S2-DEN/readouts.csv"]),
    ("items_s2.jsonl and its builder", False,
     ["prompts/battery/items_s2.jsonl", "prompts/battery/s2_pool.json", "runs/s2_battery.py"]),
    ("stamp N manifest and .ots", False,
     ["data/s2/nouns/final/stampN_manifest.txt", "data/s2/nouns/final/stampN_manifest.txt.ots"]),
    ("control stories with J1B labels (nouns tried, mental-process controls)", False,
     ["data/s2/stories/nouns", "data/s2/stories/thinking", "prompts/s2/stories/controls.json", "runs/s2_stories.py"]),
    ("earlier story pilots", False,
     ["data/s2/stories/pilot_thinking", "data/s2/stories/pilot_sorting", "data/s2/stories/pilot_sorting_v2"]),
    ("gate outputs, Qwen (AIPsy, J1B, pool, build check)", True,
     [f"results/qwen/s2/{f}" for f in ("gates.json", "gates.csv", "pool.json", "build_check.json", "job_meta.json",
                                       "controls.pt", "controls.pt.sha256")]),
    ("gate outputs, K2.6 (AIPsy, J1B, pool, build check)", True,
     [f"results/k26/s2/{f}" for f in ("gates.json", "gates.csv", "pool.json", "build_check.json", "job_meta.json",
                                      "controls.pt", "controls.pt.sha256")]),
    ("vector and gate code", False, ["runs/s2_vectors.py", "kimisroom/validate.py", "kimisroom/emotions.py"]),
    ("wordings, draws and draw log", False,
     ["data/s2/openings/[CD]A[0-9][0-9]", "data/s2/openings/check_log.jsonl", "data/s2/openings/gpt_log.jsonl",
      "results/s2/openings"]),
    ("subagent records (section 4)", False, ["data/s2/openings/claude_log.jsonl"]),
    ("generator prompts", False,
     ["prompts/s2/openings/seed_denial.md", "prompts/s2/openings/seed_control.md", "prompts/s2/openings/wording.md",
      "prompts/s2/openings/render.py", "prompts/s2/openings/rendered"]),
    ("judge prompts and stored judge answers (openings checks, assent)", False,
     ["data/s2/openings/judge", "kimisroom/judge_assent.py"]),
    ("main-line definitions", False, ["prompts/s2/openings/lines.json"]),
    ("opening code", False, ["runs/s2_openings.py", "kimisroom/s2_openings.py", "runs/s2_claude_log.py"]),
    ("wording draw seed (assignment and its .ots)", False,
     ["data/s2/openings/assignment.json", "data/s2/openings/assignment.json.ots"]),
    ("K2.6 frozen replies, raw API responses, judge verdicts", False,
     ["data/s2/replies/k26/selected.json", "data/s2/replies/k26/eligible.json", "data/s2/replies/k26/verdicts.jsonl",
      "data/s2/replies/k26/verdicts.sha256", "data/s2/replies/k26/tally.json", "data/s2/replies/k26/replies.sha256.json",
      "data/s2/replies/k26/[CD]A*.json", "results/s2/k26_*.log",
      "data/s2/replies/k26/verdicts_cap2048.jsonl", "data/s2/replies/k26/verdicts_cap2048.sha256",
      "data/s2/replies/k26/cap1200"]),
    ("Qwen frozen replies and judge verdicts", True,
     ["data/s2/replies/qwen3-32b/selected.json", "data/s2/replies/qwen3-32b/eligible.json",
      "data/s2/replies/qwen3-32b/verdicts.jsonl", "data/s2/replies/qwen3-32b/verdicts.sha256",
      "data/s2/replies/qwen3-32b/tally.json", "data/s2/replies/qwen3-32b/replies.sha256.json",
      "data/s2/replies/qwen3-32b/[CD]A*.json", "data/s2/replies/qwen3-32b/verdicts_cap2048.jsonl",
      "data/s2/replies/qwen3-32b/verdicts_cap2048.sha256", "data/s2/replies/qwen3-32b/cap1200"]),
    ("Qwen reply box and library versions", True, ["data/s2/replies/qwen3-32b/box.json"]),
    ("Qwen replies extended to the 2,048 cap (box stamp, draw-0 check, continuation batch plan)", True,
     ["data/s2/replies/qwen3-32b/extend.json", "data/s2/replies/qwen3-32b/extend_batches.json"]),
    ("canary result", True, ["data/s2/replies/qwen3-32b/canary.json"]),
    ("assent code", False, ["runs/s2_assent.py"]),
    ("s2_table.py, s2_confirm.R, s2_power.R (and the synthetic-data generator)", False,
     ["runs/s2_table.py", "runs/s2_confirm.R", "runs/s2_power.R", "tests/s2_synth.py"]),
    ("Kenward-Roger timing (s2_confirm.R --only-h1 --kr on a full-size synthetic table)", False,
     ["results/s2/kr_timing/kr.csv"]),
    ("engine change and its tests", False,
     ["kimisroom/engine.py", "kimisroom/hf_engine.py", "kimisroom/trials.py", "kimisroom/run.py",
      "tests/s2_engine_check.py", "tests/s2_moonlight_spans.py"]),
    ("engine test (d) on Moonlight, its output", True, ["results/s2/engine/moonlight_spans.log"]),
    ("runner and held-out guard", False, ["runs/s2_run.py", "kimisroom/s2_guard.py", "runs/s2_stamp.py"]),
    ("held-out list (the guard's data/s2/heldout.json)", False, ["data/s2/heldout.json"]),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def expand(pat: str, root: Path = ROOT) -> list[Path]:
    """Files a path entry stands for, sorted; empty if it names nothing."""
    if any(c in pat for c in "*?["):
        hits = sorted(root.glob(pat))
    else:
        hits = [root / pat] if (root / pat).exists() else []
    out = []
    for h in hits:
        if h.is_dir():
            out += [p for p in sorted(h.rglob("*")) if p.is_file() and not SKIP & set(p.relative_to(root).parts)
                    and not p.name.endswith(".tmp")]
        elif h.is_file():
            out.append(h)
    return out


def survey(items=None, root: Path = ROOT) -> list[dict]:
    rows = []
    for label, gpu, pats in (ITEMS if items is None else items):
        files, missing = [], []
        for pat in pats:
            got = expand(pat, root)
            (files.extend(got) if got else missing.append(pat))
        rows.append({"label": label, "gpu": gpu, "files": files, "missing": missing})
    return rows


# ---------------------------------------------------------------- held-out hashes


def heldout_names(data: dict) -> dict[str, str]:
    """hash -> name, exactly as the guard reads a heldout.json."""
    return Guard(data, manifest=Path("/nonexistent/manifest-s2.txt")).names


def heldout_lines(names: dict[str, str]) -> list[str]:
    return [f"{h}  heldout {n}" for h, n in sorted(names.items(), key=lambda x: (x[1], x[0]))]


def recompute(root: Path = ROOT) -> tuple[dict[str, str], list[str], list[str]]:
    """(hash -> name, models with replies, problems): the held-out set from the assignment, the accepted
    wordings and each model's selected.json, built the way runs/s2_run.py heldout builds it."""
    asg = json.loads((root / ASSIGNMENT.relative_to(ROOT)).read_text())
    sel, problems = {}, []
    for m, d in REPLIES.items():
        p = root / d.relative_to(ROOT) / "selected.json"
        if p.exists():
            s = json.loads(p.read_text())
            if s.get("cap") != 2048:  # left out of the held-out set, as runs/s2_run.py heldout leaves it out
                problems.append(f"{p.relative_to(root)} is from the {s.get('cap', 1200)}-token reply cap, not 2048 "
                                f"(runs/s2_assent.py, Cap): extend, grade and tally {m} first")
                continue
            sel[m] = (p.parent, s["wordings"])
    names: dict[str, str] = {}
    for arg, x in sorted(asg["arguments"].items()):
        for slot, role in sorted(x["roles"].items()):
            if not role.startswith("heldout"):
                continue
            o = f"{arg}/{slot}"
            f = root / "data/s2/openings" / arg / f"{slot}.md"
            if not f.exists():
                problems.append(f"{o}: no accepted wording file")
                continue
            text = load_denial(f)
            names[text_sha(text)] = f"{o} argument"
            for m, (d, w) in sel.items():
                r = w.get(o) or {}
                if r.get("status") != "clean":
                    continue
                reply = json.loads((d / r["reply_file"]).read_text())["reply"]
                if hashlib.sha256(reply.encode()).hexdigest() != r["reply_sha256"]:
                    problems.append(f"{o}: {m} reply file {r['reply_file']} changed after selection")
                names[text_sha(reply)] = f"{o} {m} reply"
                names[opening_sha(text, reply)] = f"{o} {m} opening"
    return names, sorted(sel), problems


def heldout_check(root: Path = ROOT) -> tuple[dict[str, str], list[str], list[str]]:
    """(names, missing, problems) for the held-out section."""
    names, models, problems = recompute(root)
    missing = [f"frozen replies of {m} (data/s2/replies/{m}/selected.json)" for m in REPLIES if m not in models]
    hp = root / HELDOUT.relative_to(ROOT)
    if hp.exists():
        theirs = heldout_names(json.loads(hp.read_text()))
        if set(theirs) != set(names):
            problems.append(f"{HELDOUT.relative_to(ROOT)} is stale: {len(set(names) - set(theirs))} held-out hashes "
                            f"not in it, {len(set(theirs) - set(names))} in it that are not held-out now; rerun "
                            "`runs/s2_run.py heldout`")
    return names, missing, problems


# ---------------------------------------------------------------- manifest


def git_head(root: Path) -> str:
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "none"


def manifest_text(rows: list[dict], names: dict[str, str], root: Path = ROOT) -> str:
    asg = json.loads((root / ASSIGNMENT.relative_to(ROOT)).read_text())
    head = (f"# Study 2 stamp A (spec section 11), {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
            f"# git HEAD {git_head(root)}; files hashed from the working tree as they are\n"
            f"# wording draw seed {asg['seed_hex']} ({asg['seed_source']})\n")
    body, seen = [], set()
    for r in rows:
        body.append(f"# {r['label']}\n")
        for p in r["files"]:
            rel = str(p.relative_to(root))
            if rel not in seen:
                seen.add(rel)
                body.append(f"{sha256(p)}  {rel}\n")
    held = (f"# held-out openings: {len(names)} SHA-256s (kimisroom/s2_guard.py text_sha and opening_sha: argument, "
            "each model's frozen reply, and argument + '\\n\\n' + reply, all stripped)\n")
    return head + "".join(body) + held + "".join(x + "\n" for x in heldout_lines(names))


def report(rows: list[dict], names: dict[str, str], hmissing: list[str], problems: list[str]) -> int:
    n_miss = 0
    for r in rows:
        tag = "present" if not r["missing"] else "MISSING"
        gpu = " (GPU session)" if r["gpu"] else ""
        print(f"{tag:8} {r['label']}{gpu}: {len(r['files'])} files")
        for m in r["missing"]:
            print(f"           missing: {m}")
        n_miss += bool(r["missing"])
    print(f"{'present' if not hmissing else 'MISSING':8} held-out hashes: {len(names)} so far")
    for m in hmissing:
        print(f"           missing: {m}")
    for p in problems:
        print(f"PROBLEM  {p}")
    n_miss += bool(hmissing)
    print(f"\n{len(rows) + 1 - n_miss} of {len(rows) + 1} items present, {n_miss} missing, {len(problems)} problems")
    return n_miss + len(problems)


def cmd_dry_run(a) -> None:
    root = Path(a.root)
    rows = survey(root=root)
    names, hmissing, problems = heldout_check(root)
    report(rows, names, hmissing, problems)


def cmd_build(a) -> None:
    root, out = Path(a.root), Path(a.out)
    rows = survey(root=root)
    names, hmissing, problems = heldout_check(root)
    if report(rows, names, hmissing, problems):
        sys.exit("refusing stamp A: items missing or problems above")
    if out.exists():
        sys.exit(f"{out} exists; stamp A is never redone in place")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(manifest_text(rows, names, root))
    print(f"{out}: {sum(len(r['files']) for r in rows)} files, {len(names)} held-out hashes")
    if a.no_stamp:
        print("--no-stamp: skipped `ots stamp`")
        return
    subprocess.run([os.environ.get("OTS", "ots"), "stamp", str(out)], check=True)
    print(f"stamped: {out}.ots")


def verify(manifest: Path, root: Path = ROOT) -> list[str]:
    """Problems: .ots digest mismatch, files changed or gone since the manifest was written."""
    manifest = Path(manifest)
    bad = []
    ots = manifest.with_name(manifest.name + ".ots")
    got = sha256(manifest)
    try:
        want = ots_digest(ots) if ots.exists() else None
    except HeldOutError as e:
        want, bad = None, [str(e)]
    if want is None and not bad:
        bad.append(f"{ots} does not exist")
    elif want is not None and want != got:
        bad.append(f"{manifest} SHA-256 {got} != .ots digest {want}")
    for line in manifest.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        h, _, rest = line.partition("  ")
        if rest.startswith("heldout ") or rest.startswith("source:"):
            continue
        p = root / rest
        if not p.is_file():
            bad.append(f"{rest}: gone")
        elif sha256(p) != h:
            bad.append(f"{rest}: changed")
    return bad


def cmd_verify(a) -> None:
    bad = verify(Path(a.manifest), Path(a.root))
    for b in bad:
        print(f"FAIL  {b}")
    if bad:
        sys.exit(f"{len(bad)} problems")
    print(f"{a.manifest}: .ots digest matches, every listed file unchanged")


def cmd_heldout_lines(a) -> None:
    names = heldout_names(json.loads(Path(a.file).read_text()))
    if not names:
        sys.exit(f"{a.file}: no held-out hashes")
    print("\n".join(heldout_lines(names)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry-run")
    d.set_defaults(fn=cmd_dry_run)
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(MANIFEST))
    b.add_argument("--no-stamp", action="store_true")
    b.set_defaults(fn=cmd_build)
    v = sub.add_parser("verify")
    v.add_argument("manifest", nargs="?", default=str(MANIFEST))
    v.set_defaults(fn=cmd_verify)
    h = sub.add_parser("heldout-lines")
    h.add_argument("file")
    h.set_defaults(fn=cmd_heldout_lines)
    for p in (d, b, v):
        p.add_argument("--root", default=str(ROOT), help="tree to hash (tests only)")
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
