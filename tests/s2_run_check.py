"""End-to-end check of the Study 2 runner on Qwen3-0.6B (bf16, on MPS if present, else CPU), steps 7 and 9 (no GPU, no API).

    .venv/bin/python tests/s2_run_check.py [--work DIR] [--model PATH]
    python tests/s2_run_check.py --moonlight ~/models/moonlight      # GPU box: the K2.6 engine path (section 5 below)
    .venv/bin/python tests/s2_run_check.py --offline                 # section 0 only: no model loaded

Stand-ins: Qwen's frozen replies do not exist yet, so K2.6's frozen replies (data/s2/replies/k26) serve as the
"Qwen" replies; the registered Qwen vectors are 5,120-wide, so a random emotion file and a random controls.pt
of the 0.6B model's width stand in, at a test layer (14 of 28) with the natural unit taken from that file.

0. (also --offline; no model) A pilot run on an item file with nothing in s2_table's reference fit, as the
   unselected battery has, is refused before anything loads, pointing at runs/s2_battery.py --select.
1. Step 7: the unsteered pass on 2 S2-DEN pilot openings over every subject-bearing row (zero-vector groups
   only); runs/s2_select.py turns it into a selection; runs/s2_battery.py --select accepts it.
2. Pilot: a tiny grid (2 openings, emotion joy on gate items only, noun N01 and one mental-process control on the
   full battery, both doses, zero vector) on a handful of items from the selected item file; runs/s2_table.py
   accepts the readouts (pilot measures, gates, decision), the gate-only zero groups match the full zero groups,
   a rerun resumes without recomputing, and a lost pass file is recomputed alone.
3. S2-FTD on a stand-in "merged" checkpoint: refused while its hash equals the base's, run once it differs;
   s2_table accepts the two arms' rows together.
3b. Pinned weights: the base arms run only on the pinned revision. Every run above passes a stand-in box file
   for the 0.6B snapshot (scripts/s2_box.sh's format) with --test --pin-revision; the registered pin is refused
   on it, as are a missing box file, a box naming another revision, one changed file hash and a stray shard;
   s2_meta.json records the revision.
4. Guard: a confirmatory (held-out) run is refused at group construction; a held-out prefix is refused on its
   decoded tokens (Qwen and K2.6 tokenizers), as kimisroom.run does; stamp B opens the guard only with a
   manifest whose SHA-256 is the .ots header digest and that lists every held-out hash.
5. (--moonlight, GPU) K2.6's path end to end on Moonlight: Kimi-style renderer, kimisroom.run's layer-wise engine
   backend with its decoded-prefix guard, conversion, s2_table acceptance, on a battery selected with a fixed
   stand-in selection (the unselected battery has no reference-fit gate item). tests/s2_moonlight_spans.py covers the
   engine's per-question windows against an HF reference; this covers the runner around it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))
import s2_select as SEL  # noqa: E402
import s2_table as T  # noqa: E402
from kimisroom.s2_guard import OTS_MAGIC, Guard, HeldOutError  # noqa: E402

PY = sys.executable
FAILS: list[str] = []
LAYER = 14
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"  # bf16 either way; MPS is ~15x faster here
K26 = ROOT / "data/s2/replies/k26"


def ok(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def sh(*args: str, expect_fail: bool = False) -> subprocess.CompletedProcess:
    r = subprocess.run([PY, *args], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0 and not expect_fail:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"command failed: {' '.join(args)}")
    return r


def model_path() -> str:
    if "--model" in sys.argv:
        return sys.argv[sys.argv.index("--model") + 1]
    from huggingface_hub import snapshot_download
    return snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True)


def box_file(mp: str, repo: str, rev: str, out: Path) -> Path:
    """A stand-in for scripts/s2_box.sh's box file: the downloaded revision and the SHA-256 of every file."""
    d = Path(mp)
    files = sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()
                   and ".cache" not in p.relative_to(d).parts and not p.name.startswith("."))
    w = {}
    for f in files:
        h = hashlib.sha256()
        with open(d / f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        w[f] = h.hexdigest()
    out.write_text(json.dumps({"created": "test", "model_key": "qwen", "hf_repo": repo, "hf_revision_pinned": rev,
                               "hf_revision_downloaded": [rev], "revision_match": True, "model_dir": mp,
                               "weights_sha256": w}))
    return out


def snapshot_rev(mp: str) -> str:
    p = Path(mp)
    return p.name if p.parent.name == "snapshots" else "test-revision"


def stand_ins(work: Path, mp: str, repo: str = "Qwen/Qwen3-0.6B") -> dict:
    """Reply dir (K2.6's replies as stand-ins), held-out list, vector files, box file."""
    rep = work / "replies"
    rep.mkdir(exist_ok=True)
    sel = json.loads((K26 / "selected.json").read_text())
    for f in ("selected.json", "eligible.json"):
        shutil.copy(K26 / f, rep / f)
    for w in sel["wordings"].values():
        if w["reply_file"]:
            shutil.copy(K26 / w["reply_file"], rep / w["reply_file"])
    sh("runs/s2_run.py", "heldout", "--out", str(work / "heldout.json"))
    cfg = json.loads((Path(mp) / "config.json").read_text())
    H, n = cfg["hidden_size"], cfg["num_hidden_layers"]
    g = torch.Generator().manual_seed(0)
    names = ["joy", "sadness", "anger", "fear", "euphoria", "calm", "pain", "surprise", "red", "ocean", "arithmetic", "random"]
    vz = {"emotions": names[:8], "vectors": {k: torch.randn(n + 1, H, generator=g) for k in names}}
    torch.save(vz, work / "vectors.pt")
    ctl = work / "controls"
    ctl.mkdir(exist_ok=True)
    vecs = {k: torch.randn(H, generator=g) for k in ("N01", "N02", "mp:visual_search")}
    info = {k: {"vclass": "thinking" if k.startswith("mp:") else "noun"} for k in vecs}
    torch.save({"layer": LAYER, "vectors": vecs, "info": info}, ctl / "controls.pt")
    (ctl / "pool.json").write_text(json.dumps({"nouns_chosen": ["N01", "N02"], "thinking_survivors": ["mp:visual_search"]}))
    el = json.loads((rep / "eligible.json").read_text())
    den = [f"{a}/{x['pilots_run'][0]}" for a, x in sorted(el["arguments"].items()) if a.startswith("DA") and x["pilots_run"]]
    con = [f"{a}/{x['pilots_run'][0]}" for a, x in sorted(el["arguments"].items()) if a.startswith("CA") and x["pilots_run"]]
    rev = snapshot_rev(mp)
    return {"replies": rep, "heldout": work / "heldout.json", "vectors": work / "vectors.pt", "controls": ctl,
            "den": den[:2], "con": con[:1], "box": box_file(mp, repo, rev, work / "box.json"), "pin": f"{repo}@{rev}"}


def common(s: dict, mp: str, out: Path) -> list[str]:
    return ["runs/s2_run.py", "run", "--model", "qwen", "--model-path", mp, "--out", str(out), "--local", DEVICE,
            "--suffix-chunk", "4", "--test", "--layer", str(LAYER), "--heldout", str(s["heldout"]),
            "--emotion-vectors", str(s["vectors"]), "--controls", str(s["controls"]), "--ckpt-hash", "full",
            "--box", str(s["box"]), "--pin-revision", s["pin"]]


def read(p: Path) -> list[dict]:
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def tiny_items(src: Path, dst: Path) -> int:
    """A handful of items, every pair whole: capacity paraphrases 1-2 (all subjects), 2 placebos, 2 references,
    the persona item, 4 known-answer pairs."""
    rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    plc = sorted({r["paraphrase"] for r in rows if r["kind"] == "s2_placebo"})[:2]
    ref = sorted({r["paraphrase"] for r in rows if r["kind"] == "s2_reference"})[:2]
    yb = sorted({r["pair_id"] for r in rows if r["kind"] == "yes_bias"})[:4]
    keep = [r for r in rows if (r["kind"] == "s2_capacity" and r["paraphrase"] in (0, 1)) or
            (r["kind"] == "s2_placebo" and r["paraphrase"] in plc) or (r["kind"] == "s2_reference" and r["paraphrase"] in ref)
            or r["kind"] == "s2_persona" or (r["kind"] == "yes_bias" and r["pair_id"] in yb)]
    dst.write_text("".join(json.dumps(r) + "\n" for r in keep))
    return len(keep)


# A fixed stand-in for step 7's selection. The unselected battery (pool rows are s2_candidate; the repo's item
# file until step 7) has as gate items only the known-answer and persona rows: none is in the reference fit,
# so a K2.6 pilot on it (gate items only) leaves s2_table nothing to fit and it refuses, as it should.
STANDIN_SEL = {"placebo": ["P01", "P02", "P03", "P04", "P05", "P06"],
               "reference": ["P07", "P08", "P09", "P10", "P11", "P12", "P13", "P14"]}


def selected_items(work: Path) -> Path:
    """The item file runs/s2_battery.py --select writes for STANDIN_SEL (the 100 gate items, 56 in the reference fit)."""
    sel = work / "standin_selection.json"
    sel.write_text(json.dumps(STANDIN_SEL))
    items = work / "items_s2_selected.jsonl"
    sh("runs/s2_battery.py", "--select", str(sel), "--out", str(items))
    return items


def no_reference_refusal(work: Path) -> None:
    """Section 0, no model: the repo's item file with every reference-fit row (T.is_ref) dropped stands in for the
    unselected battery (in which none of the pilot and confirm items is one) whether or not step 7 has run yet."""
    rows = [json.loads(x) for x in (ROOT / "prompts/battery/items_s2.jsonl").read_text().splitlines() if x.strip()]
    noref = work / "items_noref.jsonl"
    noref.write_text("".join(json.dumps(r) + "\n" for r in rows if not T.is_ref(r)))
    out = work / "noref_pilot"
    r = sh("runs/s2_run.py", "run", "--model", "k26", "--arm", "S2-DEN", "--phase", "pilot", "--out", str(out),
           "--items", str(noref), "--dry-run", expect_fail=True)
    ok(r.returncode != 0 and "runs/s2_battery.py --select" in r.stderr and "0 of the k26 pilot items" in r.stderr
       and not out.exists(), "a pilot run on an item file with no reference-fit rows is refused before anything "
                             "loads: " + (r.stderr.strip().splitlines() or [""])[-1][:110])


def moonlight(work: Path, mp: str) -> None:
    """GPU box only: the K2.6 path (Kimi renderer, the layer-wise engine through kimisroom.run, the decoded-prefix
    guard inside kimisroom.run, conversion) on Moonlight-16B-A3B-Instruct, which runs K2.6's DeepseekV3 code path.
    A tiny K2.6-style pilot (every vector on gate items only) on a selected battery (STANDIN_SEL), checked by
    runs/s2_table.py with the test layer."""
    s = stand_ins(work, mp, repo="moonshotai/Moonlight-16B-A3B-Instruct")
    items = selected_items(work)
    out = work / "moon_pilot"
    sh("runs/s2_run.py", "run", "--model", "k26", "--model-path", mp, "--out", str(out), "--test", "--layer", str(LAYER),
       "--heldout", str(s["heldout"]), "--emotion-vectors", str(s["vectors"]), "--controls", str(s["controls"]),
       "--arm", "S2-DEN", "--phase", "pilot", "--replies", str(s["replies"]), "--openings", ",".join(s["den"]),
       "--items", str(items), "--vecs", "joy,N01", "--gpus-per-worker", "1", "--box", str(s["box"]),
       "--pin-revision", s["pin"])
    T.CFG["k26"]["layer"] = LAYER
    T.OVERLAP = None
    rows = T.load([out / "readouts.csv"])
    try:
        T.table(rows, "pilot")
        G, dec = T.gates(rows)
        ok(all(x["gate"] == "True" for x in rows), f"Moonlight via kimisroom.run: {len(rows)} readouts, gate items only "
                                                   f"({sum(map(T.is_ref, rows))} in the reference fit); s2_table "
                                                   f"accepts them (rule 1 dose {dec['k26']['dose']})")
    except T.S2Error as e:
        ok(False, f"Moonlight via kimisroom.run: s2_table refuses: {e}")
    g = json.loads((out / "s2_meta.json").read_text())
    res = torch.load(out / "grid.pt", weights_only=False)
    ok(res["_meta"]["backend"] == "engine" and res["_meta"]["s2_guard"] is not None,
       f"layer-wise engine backend; kimisroom.run's guard ran ({res['_meta']['s2_guard']}); {g['n_groups']} groups")
    finish()


def main() -> None:
    work = Path(sys.argv[sys.argv.index("--work") + 1]) if "--work" in sys.argv else Path(tempfile.mkdtemp(prefix="s2run"))
    work.mkdir(parents=True, exist_ok=True)
    if "--moonlight" in sys.argv:
        return moonlight(work, sys.argv[sys.argv.index("--moonlight") + 1])
    no_reference_refusal(work)
    if "--offline" in sys.argv:
        return finish()
    mp = model_path()
    s = stand_ins(work, mp)
    print(f"work dir {work}; model {mp}; stand-in openings DEN {s['den']}, CON {s['con']}")

    # ---- 1. step 7
    t0 = time.time()
    u = work / "unsteered"
    # step 7 runs before selection, on the unselected battery (every pool row s2_candidate); once step 7 has run
    # the repo's item file is the selected one, so the unselected battery is rebuilt here from the same templates
    unsel = work / "items_s2_unselected.jsonl"
    sh("runs/s2_battery.py", "--out", str(unsel))
    sh(*common(s, mp, u), "--arm", "S2-DEN", "--phase", "unsteered", "--replies", str(s["replies"]),
       "--openings", ",".join(s["den"]), "--items", str(unsel))
    rows = read(u / "readouts.csv")
    umeta = json.loads((u / "s2_meta.json").read_text())
    n_items = len(umeta["item_sets"]["full"])
    nbox = len(json.loads(s["box"].read_text())["weights_sha256"])
    ok(umeta["hf_revision"] == s["pin"].split("@")[1] and umeta["weights_check"]["files_checked"] == nbox
       and umeta["weights_check"]["shards_checked"] >= 1,
       f"base arm: revision {umeta['hf_revision'][:8]} checked against the box file ({nbox} files) and recorded")
    ok(len(rows) == 2 * n_items and all(r["vec"] == "none" and r["y"] == r["y0"] and r["n_steered"] == "0" for r in rows),
       f"step 7 pass: {len(rows)} zero-vector readouts = 2 openings x {n_items} items ({time.time() - t0:.0f}s)")
    try:
        sel = SEL.select(SEL.load([u / "readouts.csv"]))
        real = True
    except SystemExit as e:
        print(f"     step 7 selection on Qwen3-0.6B stopped: {e}")
        real = False
    if real:
        ok(len(sel["placebo"]) == 6 and len(sel["reference"]) == 8 and not set(sel["placebo"]) & set(sel["reference"]),
           f"selection from real 0.6B readouts: placebo {sel['placebo']}, reference {sel['reference']}, reserve "
           f"{sel['reserve_added']}; {len(sel['sign_check_failures'])} pairs failed the sign check")
    else:
        # the 0.6B model fails too many sign checks for a full selection: select on the readouts with the
        # sign check relaxed by flipping the failing twins' y0 is not honest, so use synthetic inputs instead
        ok(True, "selection from real 0.6B readouts not possible (too few pairs flip sign); synthetic check below")
    # synthetic selection check: known answers
    syn = []
    for r in rows:
        r2 = dict(r)
        syn.append(r2)
    base = {}
    for r in syn:  # plant: y0 = sign x depth, pool depth by predicate, so every pair flips
        p = r["para"]
        depth = {"you": -6.0, "language models": -5.0}.get(r["subject"], 0.0)
        if r["kind"] == "s2_capacity":
            d = {"you": -40.0, "language models": -38.0}.get(r["subject"], 8.0)
        elif r["kind"] in ("s2_candidate", "s2_reserve", "s2_persona"):
            k = int(p[1:]) if p[1:].isdigit() else 5
            d = (-21.0 + 2 * k) if r["subject"] in ("rocks", "humans") else depth + 0.3 * k
            if r["kind"] == "s2_reserve":
                d = -30.0 - k
        else:
            d = 4.0
        r["y0"] = r["y"] = str(int(r["sign"]) * d + (0.01 if r["opening"] == s["den"][0] else -0.01))
        base[r["item"]] = d
    # P05's rocks pair fails to flip in one opening
    for r in syn:
        if r["kind"] == "s2_candidate" and r["para"] == "P05" and r["subject"] == "rocks" and r["opening"] == s["den"][1]:
            r["y0"] = r["y"] = "1.5"
    p = work / "syn_unsteered.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(syn[0]))
        w.writeheader()
        w.writerows(syn)
    ss = SEL.select(SEL.load([p]))
    ok("s2_candidate:P05" in ss["ineligible"] and "P05" not in ss["placebo"] + ss["reference"],
       f"selection drops a pool predicate whose rocks twins do not flip in one opening ({ss['ineligible']})")
    ok(ss["placebo"] == ["P01", "P02", "P03", "P04", "P06", "P07"],
       f"placebos are the 6 lowest scores (you and LM nearest the emotion items' level and gap): {ss['placebo']}")
    ok(len(ss["reference"]) == 8 and ss["reserve_added"] and ss["reserve_added"][0] == ss["rule2_reserve_order"][0]
       and not ss["rule2_uncovered_after"],
       f"references spread {ss['reference']}; rule 2 adds the deepest reserve first {ss['reserve_added']}")
    selfile = work / "selection.json"
    selfile.write_text(json.dumps(sel if real else ss))
    items = work / "items_s2.jsonl"
    r = sh("runs/s2_battery.py", "--select", str(selfile), "--out", str(items), expect_fail=True)
    ok(r.returncode == 0, "runs/s2_battery.py --select accepts the selection" +
       ("" if r.returncode == 0 else ": " + (r.stderr or r.stdout)[-1500:]))
    if r.returncode != 0:
        return finish()
    tiny = work / "items_tiny.jsonl"
    n_tiny = tiny_items(items, tiny)

    # ---- 2. pilot, tiny grid
    pdir = work / "pilot_den"
    pilot = [*common(s, mp, pdir), "--arm", "S2-DEN", "--phase", "pilot", "--replies", str(s["replies"]),
             "--openings", ",".join(s["den"]), "--items", str(tiny), "--vecs", "joy,N01,mp:visual_search",
             "--max-tokens", "20000"]  # several passes, so the resume check below can lose one
    t0 = time.time()
    r = sh(*pilot)
    print("     " + "\n     ".join(x for x in r.stdout.splitlines() if "groups" in x or "gate-only" in x))
    dt = time.time() - t0
    rows = read(pdir / "readouts.csv")
    meta = json.loads((pdir / "s2_meta.json").read_text())
    ok(meta["n_groups"] == 2 * (2 + 3 * 2) and "gate-only zero groups" in r.stdout,
       f"pilot grid: {meta['n_groups']} groups (2 openings x [zero, zero on gate items, 3 vectors x 2 doses]), "
       f"{len(rows)} readouts from {n_tiny} items, {dt:.0f}s")
    emo_nongate = [x for x in rows if x["vec"] == "joy" and x["gate"] != "True"]
    ok(not emo_nongate, "the emotion vector has no readout on a non-gate item")
    T.CFG["qwen"]["layer"] = LAYER  # the test layer; everything else in s2_table is as registered
    T.OVERLAP = None
    trows = T.load([pdir / "readouts.csv"])
    try:
        out = T.table(trows, "pilot")
        G, dec = T.gates(trows)
        ok(len(out) > 0 and len(G) > 0 and "qwen" in dec,
           f"runs/s2_table.py accepts the pilot readouts: {len(out)} measure rows, {len(G)} gate rows, rule 1 dose "
           f"{dec['qwen']['dose']}")
    except T.S2Error as e:
        ok(False, f"runs/s2_table.py accepts the pilot readouts: {e}")
    st = [x for x in trows if x["vclass"] != "none"]
    yb = [x for x in trows if x["kind"] == "yes_bias" and x["vclass"] == "none"]
    ok(all(abs(x["y"] - x["y0"]) > 0 for x in st) and all(x["correct_answer"] in ("Yes", "No") for x in yb),
       f"steered rows move y ({len(st)} rows); known-answer rows carry correct_answer")
    pr = subprocess.run([PY, str(ROOT / "runs/s2_table.py"), "--phase", "pilot", "--no-overlap", "--out",
                         str(work / "t.csv"), str(pdir / "readouts.csv")], capture_output=True, text=True)
    ok(pr.returncode != 0 and "layer 14 != registered 40" in pr.stderr,
       "the s2_table CLI refuses the test layer (the in-process check above patches only the layer)")

    # resume: a rerun converts only; a lost pass is recomputed alone
    t0 = time.time()
    r = sh(*pilot)
    ok("converting only" in r.stdout, f"rerun with grid.pt present converts only ({time.time() - t0:.0f}s)")
    parts = sorted((pdir / "grid.pt.parts").glob("*.pt"))
    (pdir / "grid.pt").unlink()
    parts[-1].unlink()
    before = {x["item"] + x["vec"] + x["dose"] + x["opening"]: x["y"] for x in rows}
    r = sh(*pilot)
    n_run = sum("[local pass" in x for x in r.stdout.splitlines())
    after = {x["item"] + x["vec"] + x["dose"] + x["opening"]: x["y"] for x in read(pdir / "readouts.csv")}
    ok(len(parts) > 1 and n_run == 1 and before == after, f"{len(parts)} passes; after losing one, the rerun ran {n_run} and the readouts "
       f"are identical")
    r = sh(*[x if x != "joy,N01,mp:visual_search" else "joy,N01" for x in pilot], expect_fail=True)
    ok(r.returncode != 0 and "different job" in r.stderr, "a changed grid in the same --out is refused without --fresh")

    # ---- 3. S2-FTD on a stand-in merged checkpoint
    ft = work / "merged_D2"
    ft.mkdir(exist_ok=True)
    for f in Path(mp).iterdir():
        if not (ft / f.name).exists():
            (ft / f.name).symlink_to(f.resolve())
    ftd = [*common(s, str(ft), work / "pilot_ftd"), "--arm", "S2-FTD", "--phase", "pilot", "--replies", str(s["replies"]),
           "--openings", s["con"][0], "--items", str(tiny), "--vecs", "N01", "--ckpt-registry", str(work / "ckpt.json")]
    con = [*common(s, mp, work / "pilot_con"), "--arm", "S2-CON", "--phase", "pilot", "--replies", str(s["replies"]),
           "--openings", s["con"][0], "--items", str(tiny), "--vecs", "N01", "--ckpt-registry", str(work / "ckpt.json")]
    sh(*con)
    r = sh(*ftd, expect_fail=True)
    ok(r.returncode != 0 and "base checkpoint's hash" in r.stderr, "S2-FTD on a checkpoint identical to the base is refused")
    (ft / "config.json").unlink()
    cfg = json.loads((Path(mp) / "config.json").read_text())
    cfg["_stand_in"] = "D2"
    (ft / "config.json").write_text(json.dumps(cfg))
    sh(*ftd, "--fresh")
    both = T.load([work / "pilot_con/readouts.csv", work / "pilot_ftd/readouts.csv"])
    ck = {x["arm"]: x["ckpt"] for x in both}
    try:
        T.table(both, "pilot")
        ok(ck["S2-FTD"] != ck["S2-CON"], f"s2_table accepts S2-CON and S2-FTD rows together, checkpoints {ck['S2-CON'][:10]} "
                                         f"vs {ck['S2-FTD'][:10]}")
    except T.S2Error as e:
        ok(False, f"s2_table accepts S2-CON and S2-FTD rows together: {e}")

    # ---- 3b. pinned weights
    import s2_run as RUN
    q, k = RUN.pinned_revision("qwen", None, False), RUN.pinned_revision("k26", None, False)
    ok(q[:2] == ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137")
       and k[:2] == ("moonshotai/Kimi-K2.6", "7eb5002f6aadc958aed6a9177b7ed26bb94011bb"),
       "registered pins: Qwen3-32B@9216db57, Kimi-K2.6@7eb5002f")
    try:
        RUN.pinned_revision("qwen", s["pin"], False)
        ok(False, "--pin-revision without --test is refused: accepted")
    except AssertionError as x:
        ok("tests only" in str(x), "--pin-revision without --test is refused")
    base = [x for x in con if x not in ("--ckpt-registry", str(work / "ckpt.json"))]

    def refused(label: str, out: str, want: str, *extra: str, pin: bool = True) -> None:
        args = [*base[:base.index("--out") + 1], str(work / out), *base[base.index("--out") + 2:], *extra]
        if not pin:
            i = args.index("--pin-revision")
            del args[i:i + 2]
        r = sh(*args, expect_fail=True)
        last = (r.stderr.strip().splitlines() or [""])
        ok(r.returncode != 0 and want in r.stderr and not (work / out / "grid.job.pt").exists(),
           f"base arm refused: {label} ({' '.join(last[-2:])[:150]})")

    refused("the registered Qwen3-32B pin on the 0.6B weights", "w_reg", "box revision", pin=False)
    refused("no box file", "w_nobox", "no box file", "--box", str(work / "missing_box.json"))
    other = json.loads(s["box"].read_text())
    other["hf_revision_pinned"] = "0" * 40
    other["hf_revision_downloaded"] = ["0" * 40]
    (work / "box_other.json").write_text(json.dumps(other))
    refused("a box file naming another revision", "w_other", "box revision", "--box", str(work / "box_other.json"))
    changed = json.loads(s["box"].read_text())
    shard = next(n for n in changed["weights_sha256"] if n.endswith(".safetensors"))
    changed["weights_sha256"][shard] = "0" * 64
    (work / "box_changed.json").write_text(json.dumps(changed))
    refused(f"one file hash differs ({shard})", "w_changed", "differ from the box", "--box", str(work / "box_changed.json"))
    stray = work / "stray_model"
    stray.mkdir(exist_ok=True)
    for f in Path(mp).iterdir():
        if not (stray / f.name).exists():
            (stray / f.name).symlink_to(f.resolve())
    (stray / "model-extra.safetensors").write_bytes(b"not a shard")
    i = base.index("--model-path")
    args = [*base[:i + 1], str(stray), *base[i + 2:]]
    args[args.index("--out") + 1] = str(work / "w_stray")
    r = sh(*args, expect_fail=True)
    ok(r.returncode != 0 and "did not hash" in r.stderr, "base arm refused: a stray weight shard the box did not hash")

    # ---- 4. guard
    r = sh(*common(s, mp, work / "conf"), "--arm", "S2-DEN", "--phase", "confirm", "--replies", str(s["replies"]),
           "--items", str(tiny), "--vecs", "joy", "--dry-run", expect_fail=True)
    ok(r.returncode != 0 and "HeldOutError" in r.stderr and "stamp B does not hold" in r.stderr,
       "a confirmatory run on held-out openings is refused before stamp B: " + r.stderr.strip().splitlines()[-1][:160])
    g = Guard(s["heldout"])
    e = next(x for x in g.data["entries"] if "k26" in x["replies"])
    arg, slot = e["opening"].split("/")
    sel_k = json.loads((K26 / "selected.json").read_text())["wordings"][e["opening"]]
    msgs = [{"role": "user", "content": RUN.argument_text(arg, slot)},
            {"role": "assistant", "content": json.loads((K26 / sel_k["reply_file"]).read_text())["reply"]}]
    from transformers import AutoTokenizer
    from kimisroom.trials import post_subject_group
    for name, tokdir, model in (("Qwen3", mp, "qwen"), ("K2.6", str(ROOT / "data/tokenizers/kimi-k2.6"), "k26")):
        rr = RUN.renderer(model, tokdir)
        grp = post_subject_group(rr, "x", msgs, [{"id": "q", "subject": "you", "text": "Can you feel emotions? Answer with one word, Yes or No."}])
        tok = AutoTokenizer.from_pretrained(tokdir, trust_remote_code=True)
        try:
            g.check_decoded(tok, [grp.prefix])
            ok(False, f"decoded-prefix guard ({name} tokenizer) refuses {e['opening']}: accepted")
        except HeldOutError as x:
            ok("argument" in str(x) and "reply" in str(x), f"decoded-prefix guard ({name} tokenizer) refuses {e['opening']} "
                                                          f"(argument and reply both found)")
        pil = next(o for o in s["den"])
        a2, s2 = pil.split("/")
        sel_p = json.loads((K26 / "selected.json").read_text())["wordings"][pil]
        msgs2 = [{"role": "user", "content": RUN.argument_text(a2, s2)},
                 {"role": "assistant", "content": json.loads((K26 / sel_p["reply_file"]).read_text())["reply"]}]
        grp2 = post_subject_group(rr, "x", msgs2, [{"id": "q", "subject": "you", "text": "Can you feel emotions? Answer with one word, Yes or No."}])
        ok(g.check_decoded(tok, [grp2.prefix]) == 0, f"decoded-prefix guard ({name}) passes a pilot opening")
    # stamp B
    man = work / "prereg/manifest-s2.txt"
    man.parent.mkdir(exist_ok=True)
    man.write_text("files...\n" + "\n".join(sorted(g.hashes)) + "\n")
    digest = hashlib.sha256(man.read_bytes()).digest()
    (work / "prereg/manifest-s2.txt.ots").write_bytes(OTS_MAGIC + b"\x01\x08" + digest + b"\xf0\x10rest-of-proof")
    ok(Guard(s["heldout"], man).open, "stamp B: manifest matching its .ots digest and listing every hash opens the guard")
    man.write_text(man.read_text() + "edited after stamping\n")
    gb = Guard(s["heldout"], man)
    ok(not gb.open and ".ots digest" in gb.status, f"stamp B: an edited manifest keeps it closed ({gb.status[-60:]})")
    man.write_text("files...\n" + "\n".join(sorted(g.hashes)[1:]) + "\n")
    (work / "prereg/manifest-s2.txt.ots").write_bytes(OTS_MAGIC + b"\x01\x08" + hashlib.sha256(man.read_bytes()).digest())
    gb = Guard(s["heldout"], man)
    ok(not gb.open and "does not list 1" in gb.status, "stamp B: a manifest missing one held-out hash keeps it closed")
    (work / "prereg/manifest-s2.txt.ots").unlink()
    (work / "prereg/manifest-s2.txt.ots.bak").write_bytes(b"x")
    ok(not Guard(s["heldout"], man).open, "stamp B: a stray .ots.bak (no .ots) keeps it closed")
    # stamp A fixes the list: listed and proof matching -> loads; edited list or bad proof -> refused
    sa = work / "prereg/stampA-s2.txt"
    hsha = hashlib.sha256(Path(s["heldout"]).read_bytes()).hexdigest()
    sa.write_text(f"# stamp A\n{'0' * 64}  runs/s2_run.py\n{hsha}  data/s2/heldout.json\n")
    sa_ots = work / "prereg/stampA-s2.txt.ots"
    sa_ots.write_bytes(OTS_MAGIC + b"\x01\x08" + hashlib.sha256(sa.read_bytes()).digest())
    ga = Guard(s["heldout"], man, stamp_a=sa)
    ok("lists this held-out list" in ga.stamp_a_status, f"stamp A listing the held-out list's hash: {ga.stamp_a_status[:90]}")
    gd = Guard(ga.data, man, stamp_a=sa)
    ok("lists this held-out list" in gd.stamp_a_status, "a job's embedded copy is checked through its recorded file hash")
    edited = work / "heldout_edited.json"
    d = json.loads(Path(s["heldout"]).read_text())
    d["entries"] = d["entries"][1:]
    edited.write_text(json.dumps(d))
    for label, kw in (("an edited held-out list", dict(heldout=edited)),
                      ("a stamp A whose .ots digest does not match", dict(heldout=s["heldout"], tamper=True))):
        if kw.pop("tamper", False):
            sa.write_text(sa.read_text() + "# appended\n")
        try:
            Guard(kw["heldout"], man, stamp_a=sa)
            ok(False, f"stamp A refuses {label}: accepted")
        except HeldOutError as x:
            ok(True, f"stamp A refuses {label}: {str(x)[:110]}")
    sa_ots.unlink()
    try:
        Guard(s["heldout"], man, stamp_a=sa)
        ok(False, "stamp A without its .ots is refused: accepted")
    except HeldOutError as x:
        ok("incomplete" in str(x), "stamp A without its .ots is refused")
    sa.unlink()
    ok("not written yet" in Guard(edited, man, stamp_a=sa).stamp_a_status,
       "before stamp A any held-out list is taken as it stands")
    finish()


def finish() -> None:
    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nall checks passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
