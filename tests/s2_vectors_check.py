"""Plumbing test for Study 2 step 6 (runs/s2_stories.py, runs/s2_vectors.py) on this Mac, CPU only.

    .venv/bin/python -I tests/s2_vectors_check.py [--model PATH_TO_Qwen3-0.6B]

Runs the real `run --local cpu` -> `build` -> `gate` path on the cached Qwen3-0.6B, with:
- a 10-seed subset of the registered neutral/red/ocean/arithmetic stories (k26-v2) and the whole AIPsy set,
- control stories from the dry-run noun directory (plumbing only) and 3 stories each of two pilot thinking tasks,
- injection layer 14 of 28 plus layer 7 (`model_cfg` is patched: Qwen3-0.6B has no registered file),
- a stand-in "registered" file made from this extraction (so the build check must pass) and a perturbed one
  (so it must stop), with stand-in emotion vectors = AIPsy peak-minus-neutral directions (so gate (a) has a
  vector it must flag).
Then the fine-tuned diagnostic (`ftdiag`: on a perturbed stand-in extraction the build check is recorded, not
enforced, then the gates and the cosine with the base controls) and unit checks: norm handling, the
one-row-per-control assert (fires on a duplicate and on a silently skipped label), the J1B gate threshold,
list-order pool selection, and the story filters on pilot stories.
Nothing here is a result; every output goes to a temp directory. The model loads in bf16 to keep the
process under about 3 GB (this Mac is short on RAM); run one heavy test at a time.
"""

import argparse
import glob
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/")) or [""])[0])
ap.add_argument("--keep", action="store_true", help="keep the temp directory")
ap.add_argument("--reuse", default=None, help="a kept temp directory whose extraction to reuse (skips the run)")
A = ap.parse_args()
assert A.model and Path(A.model, "config.json").exists(), "Qwen3-0.6B not cached; pass --model"

spec = importlib.util.spec_from_file_location("s2_vectors", ROOT / "runs/s2_vectors.py")
sv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sv)
ss = sv.STORIES

from kimisroom.validate import aipsy_units  # noqa: E402

ok = []


def check(cond, msg):
    assert cond, msg
    ok.append(msg)
    print(f"  ok  {msg}", flush=True)


def raises(fn, exc, needle, msg):
    try:
        fn()
    except exc as e:
        assert needle in str(e), f"{msg}: raised {type(e).__name__}({e}) without {needle!r}"
        check(True, msg)
        return
    raise AssertionError(f"{msg}: did not raise")


tmp = Path(A.reuse) if A.reuse else Path(tempfile.mkdtemp(prefix="s2vec_"))
out = tmp / "out"
LAYER, EXTRA, COEF = 14, 7, 5.0
EMOS = ["joy", "sadness", "anger", "fear", "euphoria", "surprise"]   # Qwen's registered set
cfg = {"key": "qwen", "layer": LAYER, "emotions": EMOS, "coef": COEF, "registered": tmp / "reg.pt",
       "registered_sha256": None}
sv.model_cfg = lambda key: cfg                                       # no registered file for 0.6B

# ---- inputs: small k26-v2 subset, dry-run nouns, two pilot thinking tasks (3 stories each)
seeds = {f"s{i:03d}" for i in range(10)}                            # s004, s009 are held out
full_bridge = sv.bridge_stories()
sub = [s for s in full_bridge if s["seed_id"] in seeds]
sv.bridge_stories = lambda: sub
dry = ROOT / "data/s2/stories/DRYRUN_nouns_NOT_REGISTERED"
pil = ROOT / "data/s2/stories/pilot_thinking"
assert dry.exists() and pil.exists(), "run the dry-run noun and pilot thinking generations first"
tdir = tmp / "thinking"
tdir.mkdir(exist_ok=True)
recs = [r for r in ss.load_jsonl(pil / "stories.jsonl") if r["vec"] in ("mp:visual-search", "mp:mental-rotation")]
keep = []
for v in ("mp:visual-search", "mp:mental-rotation"):
    keep += [r for r in recs if r["vec"] == v][:3]
keep[0]["split"] = "heldout"                                         # one held-out story for the diagnostic
(tdir / "stories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
shutil.copy(pil / "manifest.json", tdir / "manifest.json")
print(f"temp {tmp}; bridge {len(sub)} stories, thinking {len(keep)}, dry-run nouns "
      f"{len(ss.load_jsonl(dry / 'stories.jsonl'))}, AIPsy {len(aipsy_units())}", flush=True)

# ---- refusals before any compute
raises(lambda: sv.control_stories([dry]), SystemExit, "dry-run", "dry-run noun stories refused without --allow-dryrun")

# ---- run (real CLI path, in-process engine on CPU)
if A.reuse:
    for f in ("controls.pt", "controls.pt.sha256", "gates.json", "gates.csv", "pool.json", "build_check.json"):
        (out / f).unlink(missing_ok=True)
else:
    sv.main(["run", A.model, str(out), "--model-key", "qwen", "--stories", str(dry), "--stories", str(tdir),
             "--layers", str(EXTRA), "--local", "cpu", "--local-dtype", "bfloat16", "--allow-dryrun"])
meta = json.loads((out / "job_meta.json").read_text())
res = torch.load(out / "extract.pt", weights_only=False)
check(meta["layers"] == [EXTRA, LAYER], f"collected layers {meta['layers']}")
H = res[sub[0]["id"]]["collects"][0].shape[0]
check(all(len(res[g]["collects"]) == 2 for g in res if g != "_meta"), "two collects (one per layer) per group")
check(H == 1024, f"hidden size {H}")

# ---- stand-in registered file from this extraction: build check must pass on it
n_layers = 28
li = {EXTRA: 0, LAYER: 1}
units = {u["id"]: u for u in aipsy_units()}


def at(ids, k):
    return torch.stack([res[i]["collects"][k].float() for i in ids]).mean(0)


neu = [s["id"] for s in sub if s["condition"] == "neutral" and s["split"] == "train"]
reg = {"vectors": {}, "resid_norm": torch.zeros(n_layers), "nat_norm": torch.zeros(n_layers)}
for c in ("red", "ocean", "arithmetic"):
    ids = [s["id"] for s in sub if s["condition"] == c and s["split"] == "train"]
    reg["vectors"][c] = torch.zeros(n_layers, H)
    for L, k in li.items():
        reg["vectors"][c][L] = at(ids, k) - at(neu, k)
for L, k in li.items():
    reg["resid_norm"][L] = sum(res[i]["collect_norms"][k] for i in neu) / len(neu)
aneu = [i for i, u in units.items() if u["split"] == "neutral"]
for e in EMOS:
    lab = sv.AIPSY_LABEL[e]
    ids = [i for i, u in units.items() if u["split"] == "clinical" and u["emotion"] == lab]
    reg["vectors"][e] = torch.zeros(n_layers, H)
    for L, k in li.items():
        reg["vectors"][e][L] = torch.stack([res[f"aipsy:{i}"]["collects"][k] for i in ids]).mean(0) - \
            torch.stack([res[f"aipsy:{i}"]["collects"][k] for i in aneu]).mean(0)
reg["vectors"]["random"] = torch.randn(n_layers, H, generator=torch.Generator().manual_seed(0))

bad = {**reg, "vectors": {**reg["vectors"], "red": reg["vectors"]["red"] * 1.01}}   # norm off by 1%
torch.save(bad, tmp / "reg.pt")
cfg["registered_sha256"] = sv.sha_file(tmp / "reg.pt")
try:
    sv.build(out)
    raise AssertionError("build did not stop on a 1% norm mismatch")
except SystemExit as e:
    check(e.code == 2 and not (out / "controls.pt").exists(), "build stops (exit 2, no controls.pt) on a mismatched red")
cfg["registered_sha256"] = "0" * 64
raises(lambda: sv.build(out), AssertionError, "is not the registered", "build refuses a registered file with the wrong hash")

torch.save(reg, tmp / "reg.pt")
cfg["registered_sha256"] = sv.sha_file(tmp / "reg.pt")
blob = sv.build(out)
bc = json.loads((out / "build_check.json").read_text())
check(bc["pass"] and all(r["pass"] for r in bc["rows"].values()), "build check passes on the matching file")
check(min(r["cosine"] for k, r in bc["rows"].items() if k != "neutral_resid_norm") > 0.9999, "rebuilt cosines > 0.9999")

# ---- vectors: shapes and norm handling
vecs = blob["vectors"]
want = {"N01", "N02", "mp:visual-search", "mp:mental-rotation"}
check(set(vecs) == want, f"controls.pt vectors {sorted(vecs)}")
check(all(v.shape == (H,) and v.dtype == torch.float32 for v in vecs.values()), "each vector is (H,) fp32")
check(all(blob["stack"][k].shape == (2, H) for k in vecs), "stacks are (n_collected_layers, H)")
check(all(torch.equal(blob["stack"][k][1], vecs[k]) for k in vecs), "the injection-layer row of the stack is the vector")
check(all(abs(blob["raw_norm"][k] - float(vecs[k].norm())) < 1e-4 for k in vecs), "raw (pre-normalisation) norm recorded")
check(all(abs(float(sv.steering_vector(v, COEF).norm()) - COEF) < 1e-4 for v in vecs.values()),
      "steering_vector has norm == coefficient for every control")
nv = blob["info"]["N01"]
tr = [s["id"] for s in ss.load_jsonl(dry / "stories.jsonl") if s["vec"] == "N01" and s["split"] == "train"]
check(torch.allclose(vecs["N01"], at(tr, 1) - at(neu, 1), atol=1e-5), "N01 = train mean minus neutral train mean")
check(nv["n_train"] == len(tr) and nv["dryrun"], "N01 info: n_train and dry-run flag")
raises(lambda: sv.steering_vector(torch.zeros(H), COEF), AssertionError, "nonzero", "zero vector refused")
check((out / "controls.pt.sha256").read_text().split()[0] == sv.sha_file(out / "controls.pt"), "controls.pt hash written")

# ---- gates end to end
rep = sv.gate(out, pool=12)
rows = {r["control"]: r for r in rep["rows"]}
check(set(rows) == want | {"arithmetic", "random", "red", "ocean"} | set(EMOS), "gate rows: 4 gated + 4 reported + 6 emotions")
check(all(rows[k]["gated"] for k in want) and not any(rows[k]["gated"] for k in ("red", "random")),
      "only the new controls are gated")
check(all(set(rows[k]["a_labels"]) == {"ecstasy", "grief", "rage", "terror", "amazement"} for k in want),
      "every control scored on all five Qwen labels")
check(all(not rows[e]["a_pass"] for e in EMOS), "stand-in emotion vectors fail gate (a): it flags affect")
check(rows["arithmetic"]["b_kept"] == 120 and rows["red"]["b_emotion_labels"] == 6 and rows["red"]["b_pass"],
      "red's registered J1B rate (6/120) sits exactly at the threshold and passes")
p = rep["pool"]
check(p["nouns_tried"] == ["N01", "N02"] and p["pool_short"], f"pool: tried {p['nouns_tried']}, short of 12")
check(set(p["thinking_survivors"]) | set(p["thinking_dropped"]) == {"mp:visual-search", "mp:mental-rotation"},
      "every thinking task is either a survivor or dropped")
check(rows["mp:visual-search"]["heldout_auroc_vs_neutral"] is not None, "held-out AUROC diagnostic computed")
check((out / "gates.csv").read_text().count("\n") == len(rows) + 1, "gates.csv has one line per row")

# ---- fine-tuned diagnostic (S2-FTD, S2-FTC): a stand-in merged checkpoint whose activations are perturbed, so
# the rebuilt red, ocean and arithmetic no longer match the registered file, as on real fine-tuned weights
ftd = tmp / "ft"
shutil.rmtree(ftd, ignore_errors=True)
ftd.mkdir()
(ftd / "job_meta.json").write_text(json.dumps({**meta, "model": meta["model"].rstrip("/") + "-merged-standin"}))
gen = torch.Generator().manual_seed(1)
fres = {}
for k, r in res.items():
    if k == "_meta":
        fres[k] = r
        continue
    cs = [c.float() for c in r["collects"]]
    fres[k] = {**r, "collects": [(c + 0.3 * c.std() * torch.randn(c.shape, generator=gen)).to(r["collects"][0].dtype)
                                 for c in cs]}
torch.save(fres, ftd / "extract.pt")
(ftd / "ckpt.sha256").write_text("0" * 64 + "  ./model.safetensors\n")
try:
    sv.build(ftd)
    raise AssertionError("enforced build did not stop on the stand-in fine-tuned extraction")
except SystemExit as e:
    check(e.code == 2 and not (ftd / "controls.pt").exists(), "enforced build stops on fine-tuned weights (exit 2)")
raises(lambda: sv.ftdiag(out, out), AssertionError, "same directory", "ftdiag refuses OUT == BASE")
same = tmp / "ft_same"
shutil.rmtree(same, ignore_errors=True)
same.mkdir()
(same / "job_meta.json").write_text(json.dumps(meta))
raises(lambda: sv.ftdiag(same, out), AssertionError, "base checkpoint", "ftdiag refuses an extraction of the base checkpoint")
d = sv.main(["ftdiag", str(ftd), "--base", str(out), "--pool", "12"])
fbc = json.loads((ftd / "build_check.json").read_text())
check(not fbc["pass"] and fbc["enforced"] is False and not d["registered_match"],
      "ftdiag: failing build check recorded (pass false, enforced false), not stopped")
check((ftd / "controls.pt").exists() and (ftd / "gates.json").exists() and (ftd / "diag.json").exists(),
      "ftdiag writes controls.pt, gates.json and diag.json")
check(json.loads((ftd / "diag.json").read_text()) == json.loads(json.dumps(d)), "diag.json is the returned record")
check(set(d["registered_cosine"]) == {"red", "ocean", "arithmetic"} and all(c < 0.9999 for c in d["registered_cosine"].values()),
      f"rebuilt red/ocean/arithmetic recorded against the registered file: {d['registered_cosine']}")
fblob = torch.load(ftd / "controls.pt", weights_only=False)
man = float(torch.nn.functional.cosine_similarity(fblob["vectors"]["N01"], blob["vectors"]["N01"], dim=0))
check(set(d["cosine_ft_vs_base"]) == want and abs(d["cosine_ft_vs_base"]["N01"] - man) < 1e-6
      and all(-1 <= c < 0.9999 for c in d["cosine_ft_vs_base"].values()),
      f"cosine with the base controls for every control: {d['cosine_ft_vs_base']}")
check(abs(d["norm_ratio_ft_over_base"]["N01"] - fblob["raw_norm"]["N01"] / blob["raw_norm"]["N01"]) < 1e-9,
      "norm ratio fine-tuned over base")
check(want | set(EMOS) <= set(d["gates"]) and all(isinstance(d["gates"][v]["a_pass"], bool) for v in want),
      "gates (a) and (b) computed on the fine-tuned extraction")
check(d["base_controls_sha256"] == sv.sha_file(out / "controls.pt") and d["checkpoint_files_sha256"] is not None,
      "diag records the base controls hash and the checkpoint file list hash")

# ---- the one-row-per-control assert
st = blob["stack"]
raises(lambda: sv.aipsy_gate(res, [("N01", st["N01"]), ("N01", st["N01"])], EMOS, 1), AssertionError,
       "more than once", "AIPsy gate refuses a duplicated control")
real_validate = sv.validate
sv.validate = lambda r, v, emotion_of=None, emotions=None: {k: x for k, x in real_validate(
    r, v, emotion_of, emotions).items() if not k.startswith("N02@")}
raises(lambda: sv.aipsy_gate(res, [("N01", st["N01"]), ("N02", st["N02"])], EMOS, 1), AssertionError,
       "skipped", "AIPsy gate refuses a control validate skipped silently")
sv.validate = real_validate
raises(lambda: sv.aipsy_gate(res, [("N01", st["N01"])], ["calm"], 1), AssertionError, "no AIPsy label",
       "AIPsy gate refuses an emotion set with no AIPsy label")
emo_dir = reg["vectors"]["joy"][[EXTRA, LAYER]]
g = sv.aipsy_gate(res, [("affect-like", emo_dir), ("N01", st["N01"])], EMOS, 1)
check(not g[0]["pass"] and g[0]["max_auroc"] > g[0]["null_p90"], "a control pointing at an AIPsy emotion fails gate (a)")

# ---- J1B gate threshold
mk = lambda labs: [{"checks": {"judge8": x}} for x in labs]     # noqa: E731
check(ss.j1b_gate(mk(["calm"] * 6 + ["none"] * 114))["pass"], "J1B: 6/120 passes")
check(not ss.j1b_gate(mk(["calm"] * 7 + ["none"] * 113))["pass"], "J1B: 7/120 fails")
check(not ss.j1b_gate(mk(["calm"] * 6 + ["none"] * 104))["pass"], "J1B: 6/110 fails (rate, not count)")
check(not ss.j1b_gate(mk([None] * 6 + ["calm"] + ["none"] * 113))["pass"], "J1B: unparseable counts as emotion")
check(not ss.j1b_gate([])["pass"], "J1B: no kept stories fails")

# ---- pool selection in list order
fake = [{"control": f"N{i:02d}", "vclass": "noun", "pass": i not in (2, 5)} for i in range(1, 20)]
sel = sv.select_pool(fake, 12)
check(sel["nouns_chosen"] == [f"N{i:02d}" for i in range(1, 15) if i not in (2, 5)] and
      sel["nouns_skipped"] == ["N02", "N05"] and sel["nouns_not_reached"] == [f"N{i:02d}" for i in range(15, 20)],
      "pool: first 12 passing nouns in list order, failures skipped, rest not reached")

# ---- story filters re-derive the kept pilot stories, and gate (c) matches topics
spec_ = ss.load_spec()
ss.ACTIVE.clear()
ss.ACTIVE.update({c.condition: c for c in ss.thinking_controls(spec_)})
seed_by = {s["seed_id"]: s for s in ss.load_jsonl(ss.K26 / "seeds.jsonl")}
pil_all = [r for r in ss.load_jsonl(pil / "stories.jsonl") if r["condition"] in ss.ACTIVE]   # retired tasks skipped
check(all(not ss.s2_story_checks(r["text"], r["condition"], seed_by[r["seed_id"]])[1] for r in pil_all),
      f"all {len(pil_all)} kept pilot stories pass the filters again")
lamp = ss.noun_controls(spec_, [("N01", "library")])[0]
check(lamp.excluded(seed_by["s040"]) and not lamp.excluded(seed_by["s000"]),
      "gate (c): 'library' excludes 'a day at the public library' seeds only")
check(ss.noun_forms("box") == ["box", "boxes", "boxs"] and "knives" in ss.noun_forms("knife"), "noun plural forms")

print(f"\nALL {len(ok)} CHECKS PASSED")
if not (A.keep or A.reuse):
    shutil.rmtree(tmp)
