"""Study 2 control vectors and their gates (drafts/study2-spec.md v4, section 3; section 12 step 6).

Three stages, one output directory per model (results/{qwen,k26}/s2):

    run    MODEL OUT --model-key qwen|k26 --stories DIR [--stories DIR ...]   (GPU)
           one extraction: the registered neutral, red, ocean and arithmetic stories (data/stimuli/k26-v2),
           every new control story in the --stories dirs, and AIPsy-Affect, collected at the injection
           layer (--layers to add more) through kimisroom.run (Qwen3-32B: HF engine, 2 GPUs per worker;
           K2.6: the layer-wise engine). Writes OUT/extract.job.pt, OUT/job_meta.json, OUT/extract.pt.
    build  OUT                                                               (CPU)
           the build check, then OUT/controls.pt. Stops (exit 2, no controls.pt) unless the rebuilt red,
           ocean and arithmetic match the registered file at the injection layer (cosine > 0.9999, norm
           within 0.1%) and the rebuilt neutral residual norm matches its `resid_norm` within 0.1%.
    gate   OUT [--pool 12]                                                   (CPU)
           gates (a) AIPsy and (b) J1B, the record of (c), the diagnostic (d), and the pool: sampled nouns
           in list order until --pool pass (rule 3: at most 24), thinking tasks pass/fail with no
           replacement. Writes OUT/gates.json, OUT/gates.csv, OUT/pool.json.
    ftdiag OUT --base BASE [--pool 12]                                       (CPU)
           S2-FTD and S2-FTC (section 3: they steer with the base vectors; patterns rebuilt on the merged
           weights are diagnostics only). OUT is a `run` on a merged fine-tuned checkpoint, BASE the base
           model's finished output dir (results/qwen/s2). The build check is run and recorded but not
           enforced (on fine-tuned weights the rebuilt red, ocean and arithmetic are expected to differ from
           the registered file), then OUT/controls.pt, the same gates, and the cosine and norm ratio of every
           rebuilt control with BASE/controls.pt. Writes OUT/diag.json besides the build and gate outputs.

Vectors (section 3 build path): the train-split mean residual over every token of a plain-text story at
the injection layer, control minus the registered neutral stories' mean (runs/vectors.py:112-113; the
neutral stories are the 96 k26-v2 train stories, re-run here). controls.pt holds the raw, unnormalised
vector of each new control and its raw norm; `steering_vector` unit-normalises and scales by the
coefficient, so every vector steers with the same norm. Emotion, red, ocean, arithmetic, random vectors
and the natural unit are never rebuilt: they come only from the registered vectors.pt, hash-asserted.

Gate (a), AIPsy-Affect: per control, the maximum over the AIPsy labels of the model's registered emotions
(Qwen: ecstasy, grief, rage, terror, amazement; K2.6: ecstasy, terror, amazement) of the AUROC of the
peak-intensity items of that label against the neutral twins (kimisroom.validate's `auroc` column). Every
control is passed to `validate` with an explicit `emotion_of` entry per label, because the default mapping
is the vector's own name and validate skips unknown names silently; the gate then asserts exactly one row
per control and one validate row per (control, label). The threshold is the 90th percentile of the same
maximum over validate's 500 random directions (seed 0). Pass: maximum < threshold.
Gate (b), J1B: emotion labels (any J1B answer but `none`; unparseable counts as emotion) on the control's
kept stories at most 6/120 of them (runs/s2_stories.py `j1b_gate`).
Gate (c): applied at generation (the excluded seeds are listed); (d) cosine with the mean of the model's
registered emotion vectors, reported only. Arithmetic, random, red and ocean are reported under (a) and
(b) (arithmetic, red and ocean J1B from k26-v2's `checks.judge8`) and never gated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT))

from kimisroom.emotions import AIPSY_LABEL  # noqa: E402
from kimisroom.trials import load_jsonl, story_group  # noqa: E402
from kimisroom.validate import N_NULL, _null_dirs, aipsy_groups, aipsy_units, auroc, validate  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_s2_{name}", RUNS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STORIES = _load("s2_stories")       # j1b_gate, the noun list check
K26 = ROOT / "data/stimuli/k26-v2"
BRIDGE = ("red", "ocean", "arithmetic")   # rebuilt only to check the build against the registered file


def model_cfg(key: str) -> dict:
    """Registered constants from their single sources: layer and emotion set from runs/s2_table.py,
    vector file, hash and coefficient from runs/s2_battery.py."""
    table, battery = _load("s2_table"), _load("s2_battery")
    t = table.CFG[key]
    b = battery.MODELS[{"qwen": "qwen3-32b", "k26": "kimi-k2.6"}[key]]
    assert t["layer"] == b["layer"], (key, t["layer"], b["layer"])
    return {"key": key, "layer": t["layer"], "emotions": list(t["emo"]), "coef": b["coef"],
            "registered": b["vectors"], "registered_sha256": b["vectors_sha256"]}


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_registered(path: Path, sha: str | None) -> dict:
    if sha is not None:
        got = sha_file(path)
        assert got == sha, f"{path}: SHA-256 {got} is not the registered {sha}"
    return torch.load(path, weights_only=False, map_location="cpu")


def steering_vector(raw: torch.Tensor, coef: float) -> torch.Tensor:
    """Section 3: unit-normalised and scaled by the coefficient, so every vector has norm `coef`."""
    v = raw.to(torch.float32)
    n = v.norm()
    assert v.ndim == 1 and torch.isfinite(v).all() and n > 0, "steering vector must be a finite nonzero (H,)"
    return v / n * coef


# ---------------------------------------------------------------- inputs


def bridge_stories() -> list[dict]:
    return [r for r in load_jsonl(K26 / "stories.jsonl") if r["condition"] in ("neutral",) + BRIDGE]


def control_stories(dirs: list[Path], allow_dryrun: bool = False) -> list[dict]:
    """Every kept story of the new controls, with a check that noun stories came from the stamped list."""
    out, seen = [], {}
    for d in dirs:
        d = Path(d)
        man = json.loads((d / "manifest.json").read_text())
        recs = load_jsonl(d / "stories.jsonl")
        for r in recs:
            assert r["vec"] not in seen or seen[r["vec"]] == str(d), f"{r['vec']} has stories in {seen[r['vec']]} and {d}"
            seen[r["vec"]] = str(d)
            if r.get("dryrun") and not allow_dryrun:
                raise SystemExit(f"{d}: dry-run noun stories are plumbing only and never become vectors")
        if any(r["vclass"] == "noun" and not r.get("dryrun") for r in recs):
            assert man["run"]["noun_list_sha256"] == sha_file(STORIES.NOUN_LIST), f"{d}: noun list changed since generation"
            STORIES.registered_nouns()          # still stamped
        out += recs
    return out


# ---------------------------------------------------------------- stage: run (GPU)


def renderer(model: str):
    from transformers import AutoTokenizer

    from kimisroom.trials import Renderer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    cfg = json.loads((Path(model) / "config.json").read_text())
    if cfg["model_type"] == "qwen3":
        from kimisroom.hf_engine import Qwen3Renderer
        return Qwen3Renderer(tok, enable_thinking=False), cfg
    return Renderer(tok, thinking=False), cfg


def build_job(r, layers: list[int], stories: list[dict], aipsy: bool = True) -> list:
    ids = [s["id"] for s in stories]
    assert len(set(ids)) == len(ids), "story ids are not unique"
    groups = [story_group(r, s["id"], s["text"], layers) for s in stories]
    if aipsy:
        groups += aipsy_groups(r, layers)
    return groups


def rel(d) -> str:
    p = Path(d).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def stage_run(a) -> None:
    cfg = model_cfg(a.model_key)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = sorted({cfg["layer"], *(int(x) for x in a.layers.split(",") if x)}) if a.layers else [cfg["layer"]]
    r, mcfg = renderer(a.model)
    stories = bridge_stories() + control_stories([Path(d) for d in a.stories], a.allow_dryrun)
    groups = build_job(r, layers, stories)
    n_layers = mcfg.get("text_config", mcfg)["num_hidden_layers"]
    assert max(layers) < n_layers, (layers, n_layers)
    meta = {"model_key": a.model_key, "model": str(a.model), "model_type": mcfg["model_type"], "layers": layers,
            "story_dirs": [rel(d) for d in a.stories],
            "stories_sha256": {d: sha_file(Path(d) / "stories.jsonl") for d in a.stories},
            "k26_stories_sha256": sha_file(K26 / "stories.jsonl"), "n_groups": len(groups),
            "n_tokens": sum(g.n_tokens for g in groups), "allow_dryrun": bool(a.allow_dryrun)}
    (out / "job_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    torch.save({"groups": groups, "top_k": 1}, out / "extract.job.pt")
    print(f"{len(groups)} groups, {meta['n_tokens']:,} tokens, layers {layers}", flush=True)
    if a.local:
        res = run_local(a.model, groups, a.local, getattr(torch, a.local_dtype))
        torch.save(res, out / "extract.pt")
        return
    per_worker = ["--gpus-per-worker", "2"] if mcfg["model_type"] == "qwen3" else []
    subprocess.run([sys.executable, "-m", "kimisroom.run", out / "extract.job.pt", out / "extract.pt",
                    "--model", a.model, *per_worker], check=True, cwd=ROOT)


def run_local(model: str, groups: list, device: str, dtype: torch.dtype = torch.float32) -> dict:
    """In-process HF engine for small models on this Mac (tests), no CUDA."""
    from kimisroom.hf_engine import HFEngine
    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *x, **k: None      # HFEngine times itself with it
    eng = HFEngine(model, device=device, dtype=dtype)
    return eng.run(groups)


# ---------------------------------------------------------------- stage: build


def acts_at(res: dict, ids: list[str], li: int) -> torch.Tensor:
    return torch.stack([res[i]["collects"][li].to(torch.float32) for i in ids])


def mean_diff(res: dict, target: list[str], base: list[str], li: int) -> torch.Tensor:
    return acts_at(res, target, li).mean(0) - acts_at(res, base, li).mean(0)


def build_check(res: dict, bridge: list[dict], reg: dict, layer: int, li: int) -> dict:
    """Rebuilt neutral and the three registered controls against the registered file at `layer`."""
    def ids(cond, split="train"):
        return [s["id"] for s in bridge if s["condition"] == cond and s["split"] == split]

    neu = ids("neutral")
    out = {"layer": layer, "n_neutral_train": len(neu), "rows": {}}
    resid = sum(res[i]["collect_norms"][li] for i in neu) / len(neu)
    reg_resid = float(reg["resid_norm"][layer])
    rel = abs(resid / reg_resid - 1)
    out["rows"]["neutral_resid_norm"] = {"rebuilt": resid, "registered": reg_resid, "rel_diff": rel, "pass": rel <= 1e-3}
    for c in BRIDGE:
        v = mean_diff(res, ids(c), neu, li)
        w = reg["vectors"][c][layer].to(torch.float32)
        cos = float(torch.nn.functional.cosine_similarity(v, w, dim=0))
        nr = abs(float(v.norm() / w.norm()) - 1)
        out["rows"][c] = {"cosine": cos, "norm_rebuilt": float(v.norm()), "norm_registered": float(w.norm()),
                          "rel_norm_diff": nr, "pass": cos > 0.9999 and nr <= 1e-3}
    out["pass"] = all(r["pass"] for r in out["rows"].values())
    return out


def build(out: Path, reg_path: Path | None = None, reg_sha: str | None = "registered",
          allow_dryrun: bool = False, enforce: bool = True) -> dict:
    """OUT/extract.pt -> OUT/controls.pt. `reg_path`/`reg_sha` exist for the CPU test only (a model with no
    registered file); the CLI always uses the registered file and its hash. `enforce=False` (ftdiag only, a
    merged fine-tuned checkpoint) records a failing build check and builds anyway; `pass` stays the true
    result and `enforced` says it was not applied."""
    meta = json.loads((out / "job_meta.json").read_text())
    cfg = model_cfg(meta["model_key"])
    layer = cfg["layer"]
    li = meta["layers"].index(layer)
    reg = load_registered(reg_path or cfg["registered"], cfg["registered_sha256"] if reg_sha == "registered" else reg_sha)
    res = torch.load(out / "extract.pt", weights_only=False)
    bridge = bridge_stories()
    chk = build_check(res, bridge, reg, layer, li)
    chk["enforced"] = enforce
    (out / "build_check.json").write_text(json.dumps(chk, indent=1) + "\n")
    for k, r in chk["rows"].items():
        print(f"build check {k:20s} " + " ".join(f"{x}={y:.6g}" if isinstance(y, float) else f"{x}={y}"
                                                  for x, y in r.items()))
    if not chk["pass"]:
        if enforce:
            print("BUILD STOPPED: the rebuilt neutral or registered controls do not match the registered file",
                  flush=True)
            raise SystemExit(2)
        print("build check failed, not enforced (fine-tuned diagnostic): recorded in build_check.json and "
              "controls.pt", flush=True)

    neu = [s["id"] for s in bridge if s["condition"] == "neutral" and s["split"] == "train"]
    stories = control_stories([ROOT / d for d in meta["story_dirs"]], allow_dryrun or meta["allow_dryrun"])
    by_vec: dict[str, list[dict]] = {}
    for s in stories:
        by_vec.setdefault(s["vec"], []).append(s)
    vecs, norms, info, stack = {}, {}, {}, {}
    for v, rs in sorted(by_vec.items()):
        train = [s["id"] for s in rs if s["split"] == "train"]
        assert train, f"{v}: no train stories"
        d = mean_diff(res, train, neu, li)
        vecs[v] = d
        norms[v] = float(d.norm())
        stack[v] = torch.stack([mean_diff(res, train, neu, k) for k in range(len(meta["layers"]))])
        info[v] = {"vclass": rs[0]["vclass"], "condition": rs[0]["condition"], "word": rs[0].get("word"),
                   "n_train": len(train), "n_heldout": sum(s["split"] == "heldout" for s in rs),
                   "raw_norm": norms[v], "dryrun": any(s.get("dryrun") for s in rs)}
    blob = {"model_key": meta["model_key"], "layer": layer, "coef": cfg["coef"], "vectors": vecs, "raw_norm": norms,
            "info": info, "stack": stack, "stack_layers": meta["layers"], "n_neutral_train": len(neu),
            "build_check": chk, "registered": {"path": str(reg_path or cfg["registered"]),
                                               "sha256": sha_file(reg_path or cfg["registered"])},
            "stories_sha256": meta["stories_sha256"], "k26_stories_sha256": meta["k26_stories_sha256"]}
    torch.save(blob, out / "controls.pt")
    h = sha_file(out / "controls.pt")
    (out / "controls.pt.sha256").write_text(f"{h}  controls.pt\n")
    print(f"controls.pt: {len(vecs)} vectors at layer {layer} (H={next(iter(vecs.values())).shape[0] if vecs else 0}), "
          f"sha256 {h}")
    return blob


# ---------------------------------------------------------------- stage: gate


def aipsy_labels(emotions: list[str]) -> dict[str, str]:
    """label -> the first registered emotion carrying it (joy and euphoria share ecstasy)."""
    out: dict[str, str] = {}
    for e in emotions:
        if e in AIPSY_LABEL:
            out.setdefault(AIPSY_LABEL[e], e)
    return out


def aipsy_null_max(res: dict, labels: dict[str, str], n_layers: int, H: int) -> torch.Tensor:
    """(n_layers, N_NULL): per random direction, the max over labels of AUROC(peak items, neutral twins),
    with validate's own null directions, so the per-label AUROCs are validate's p_auroc null."""
    units = {u["id"]: u for u in aipsy_units()}
    null = _null_dirs(n_layers, H)

    def score(ids):
        a = torch.stack([torch.stack(res[f"aipsy:{i}"]["collects"]).to(torch.float32) for i in ids], dim=1)
        return torch.einsum("lnh,lhd->ldn", a, null)

    neu = score([i for i, u in units.items() if u["split"] == "neutral"])
    per = [auroc(score([i for i, u in units.items() if u["split"] == "clinical" and u["emotion"] == lab]), neu)
           for lab in labels]
    return torch.stack(per).amax(0)


def aipsy_gate(res: dict, controls: list[tuple[str, torch.Tensor]], emotions: list[str], li: int,
               null_max: torch.Tensor | None = None) -> list[dict]:
    """Gate (a). `controls` is a list of (name, (n_layers, H) vector); the return has exactly one row per
    control, in input order, or the function raises."""
    names = [n for n, _ in controls]
    dup = {n: k for n, k in Counter(names).items() if k > 1}
    assert not dup, f"AIPsy gate: controls appear more than once: {dup}"
    labels = aipsy_labels(emotions)
    assert labels, f"no AIPsy label for {emotions}"
    vecs, emo_of = {}, {}
    for n, v in controls:
        for lab, e in labels.items():
            key = f"{n}@{lab}"
            vecs[key], emo_of[key] = v.to(torch.float32), e
    table = validate(res, vecs, emotion_of=emo_of, emotions=emotions)
    missing = sorted(set(vecs) - set(table))
    assert not missing, f"AIPsy gate: validate skipped {missing} (label mapping)"
    n_layers, H = controls[0][1].shape
    if null_max is None:
        null_max = aipsy_null_max(res, labels, n_layers, H)
    thr = torch.quantile(null_max, 0.9, dim=-1)                                 # (n_layers,)
    rows = []
    for n, _ in controls:
        per = {lab: table[f"{n}@{lab}"][li]["auroc"] for lab in labels}
        mx = max(per.values())
        rows.append({"control": n, "labels": per, "max_auroc": mx, "argmax_label": max(per, key=per.get),
                     "null_p90": float(thr[li]), "null_n": N_NULL,
                     "p_max": float((null_max[li] >= mx).float().mean()), "pass": mx < float(thr[li])})
    got = Counter(r["control"] for r in rows)
    assert len(rows) == len(controls) and set(got) == set(names) and all(k == 1 for k in got.values()), \
        f"AIPsy gate: not one row per control: {dict(got)}"
    return rows


def heldout_auroc(res: dict, rs: list[dict], neutral_heldout: list[str], v: torch.Tensor, li: int) -> float | None:
    own = [s["id"] for s in rs if s["split"] == "heldout"]
    if not own or not neutral_heldout:
        return None
    u = v / v.norm()
    return float(auroc(acts_at(res, own, li) @ u, acts_at(res, neutral_heldout, li) @ u))


def select_pool(rows: list[dict], pool: int) -> dict:
    """Sampled nouns in list order: each passing noun joins until `pool` have; the rest are not reached."""
    nouns = sorted((r for r in rows if r["vclass"] == "noun"), key=lambda r: r["control"])
    chosen, tried, skipped = [], [], []
    for r in nouns:
        if len(chosen) >= pool:
            break
        tried.append(r["control"])
        (chosen if r["pass"] else skipped).append(r["control"])
    thinking = [r for r in rows if r["vclass"] == "thinking"]
    return {"pool_size_target": pool, "nouns_chosen": chosen, "nouns_skipped": skipped, "nouns_tried": tried,
            "nouns_not_reached": [r["control"] for r in nouns if r["control"] not in tried],
            "pool_short": len(chosen) < pool,
            "thinking_survivors": [r["control"] for r in thinking if r["pass"]],
            "thinking_dropped": [r["control"] for r in thinking if not r["pass"]]}


def gate(out: Path, pool: int = 12, reg_path: Path | None = None, reg_sha: str | None = "registered",
         controls_override: list[tuple[str, torch.Tensor]] | None = None) -> dict:
    meta = json.loads((out / "job_meta.json").read_text())
    cfg = model_cfg(meta["model_key"])
    layer, emotions = cfg["layer"], cfg["emotions"]
    li = meta["layers"].index(layer)
    blob = torch.load(out / "controls.pt", weights_only=False)
    reg = load_registered(reg_path or cfg["registered"], cfg["registered_sha256"] if reg_sha == "registered" else reg_sha)
    res = torch.load(out / "extract.pt", weights_only=False)
    L = meta["layers"]

    stories = control_stories([ROOT / d for d in meta["story_dirs"]], meta["allow_dryrun"])
    by_vec: dict[str, list[dict]] = {}
    for s in stories:
        by_vec.setdefault(s["vec"], []).append(s)
    k26 = load_jsonl(K26 / "stories.jsonl")
    neu_ho = [s["id"] for s in bridge_stories() if s["condition"] == "neutral" and s["split"] == "heldout"]
    emo_mean = torch.stack([reg["vectors"][e][L].to(torch.float32) for e in emotions]).mean(0)   # (n_job_layers, H)

    gated = controls_override or [(v, blob["stack"][v]) for v in sorted(blob["vectors"])]
    reported = [(n, reg["vectors"][n][L].to(torch.float32)) for n in ("arithmetic", "random", "red", "ocean")]
    reference = [(e, reg["vectors"][e][L].to(torch.float32)) for e in emotions]
    n_layers, H = gated[0][1].shape
    null_max = aipsy_null_max(res, aipsy_labels(emotions), n_layers, H)
    a_rows = aipsy_gate(res, gated, emotions, li, null_max)
    a_rep = aipsy_gate(res, reported + reference, emotions, li, null_max)

    j1b_reg = {c: STORIES.j1b_gate([s for s in k26 if s["condition"] == c]) for c in BRIDGE}
    rows = []
    for ar in a_rows:
        v = ar["control"]
        rs = by_vec.get(v, [])
        b = STORIES.j1b_gate(rs)
        vec = blob["stack"][v][li] if v in blob["stack"] else dict(gated)[v][li]
        rows.append({"control": v, "vclass": blob["info"].get(v, {}).get("vclass", "test"),
                     "word": blob["info"].get(v, {}).get("word"), "gated": True,
                     "n_train": blob["info"].get(v, {}).get("n_train"), "raw_norm": float(vec.norm()),
                     "a_max_auroc": ar["max_auroc"], "a_label": ar["argmax_label"], "a_null_p90": ar["null_p90"],
                     "a_p": ar["p_max"], "a_pass": ar["pass"], "a_labels": ar["labels"],
                     "b_emotion_labels": b["n_emotion_labels"], "b_kept": b["n_kept"], "b_pass": b["pass"],
                     "b_labels": b["labels"],
                     "d_cos_mean_emotion": float(torch.nn.functional.cosine_similarity(vec, emo_mean[li], dim=0)),
                     "heldout_auroc_vs_neutral": heldout_auroc(res, rs, neu_ho, vec, li),
                     "pass": ar["pass"] and b["pass"]})
    for ar in a_rep:
        n = ar["control"]
        vec = reg["vectors"][n][layer].to(torch.float32)
        b = j1b_reg.get(n)
        rows.append({"control": n, "vclass": {"arithmetic": "arithmetic", "random": "noise"}.get(
                         n, "bridge" if n in ("red", "ocean") else "emotion"),
                     "word": None, "gated": False, "n_train": None, "raw_norm": float(vec.norm()),
                     "a_max_auroc": ar["max_auroc"], "a_label": ar["argmax_label"], "a_null_p90": ar["null_p90"],
                     "a_p": ar["p_max"], "a_pass": ar["pass"], "a_labels": ar["labels"],
                     "b_emotion_labels": b["n_emotion_labels"] if b else None, "b_kept": b["n_kept"] if b else None,
                     "b_pass": b["pass"] if b else None, "b_labels": b["labels"] if b else None,
                     "d_cos_mean_emotion": float(torch.nn.functional.cosine_similarity(vec, emo_mean[li], dim=0)),
                     "heldout_auroc_vs_neutral": None, "pass": None})
    sel = select_pool([r for r in rows if r["gated"]], pool)
    excluded_c = {}
    for d in meta["story_dirs"]:
        m = json.loads((ROOT / d / "manifest.json").read_text())
        excluded_c.update({v: x["excluded_seeds_gate_c"] for v, x in m["per_control"].items()})
    for r in rows:
        r["c_excluded_seeds"] = excluded_c.get(r["control"])
    rep = {"model_key": meta["model_key"], "layer": layer, "emotions": emotions,
           "aipsy_labels": aipsy_labels(emotions), "null": {"n": N_NULL, "seed": 0, "quantile": 0.9,
                                                             "method": "torch.quantile linear"},
           "j1b_max_rate": "6/120", "controls_pt_sha256": sha_file(out / "controls.pt"), "rows": rows, "pool": sel}
    (out / "gates.json").write_text(json.dumps(rep, indent=1) + "\n")
    cols = ["control", "vclass", "word", "gated", "n_train", "raw_norm", "a_max_auroc", "a_label", "a_null_p90", "a_p",
            "a_pass", "b_emotion_labels", "b_kept", "b_pass", "d_cos_mean_emotion", "heldout_auroc_vs_neutral", "pass"]
    with (out / "gates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "pool.json").write_text(json.dumps(sel, indent=1) + "\n")
    print(f"{'control':24s} {'gated':5s} {'aipsy max':>9s} {'p90':>6s} {'a':4s} {'J1B':>7s} {'b':4s} {'cos(d)':>7s} pass")
    for r in rows:
        bj = f"{r['b_emotion_labels']}/{r['b_kept']}" if r["b_kept"] is not None else "-"
        print(f"{r['control']:24s} {str(r['gated']):5s} {r['a_max_auroc']:9.3f} {r['a_null_p90']:6.3f} "
              f"{'ok' if r['a_pass'] else 'FAIL':4s} {bj:>7s} {('ok' if r['b_pass'] else 'FAIL') if r['b_pass'] is not None else '-':4s} "
              f"{r['d_cos_mean_emotion']:7.3f} {r['pass']}")
    print(json.dumps(sel, indent=1))
    return rep


# ---------------------------------------------------------------- stage: ftdiag


def ftdiag(out: Path, base: Path, pool: int = 12, reg_path: Path | None = None,
           reg_sha: str | None = "registered") -> dict:
    """Section 3, S2-FTD and S2-FTC: on a merged fine-tuned checkpoint's extraction (OUT), the build check
    recorded but not enforced, the rebuilt controls, their gates, and their cosine with the base model's
    controls (BASE/controls.pt). Diagnostics only: the arms steer with BASE's vectors."""
    out, base = Path(out), Path(base)
    assert out.resolve() != base.resolve(), "ftdiag: OUT and BASE are the same directory"
    meta = json.loads((out / "job_meta.json").read_text())
    bmeta = json.loads((base / "job_meta.json").read_text())
    assert meta["model_key"] == bmeta["model_key"], ("model_key", meta["model_key"], bmeta["model_key"])
    assert meta["model"] != bmeta["model"], f"ftdiag: OUT was extracted from the base checkpoint {meta['model']}"
    assert meta["stories_sha256"] == bmeta["stories_sha256"], "ftdiag: OUT and BASE used different control stories"
    if not (out / "controls.pt").exists():
        build(out, reg_path, reg_sha, enforce=False)
    rep = gate(out, pool, reg_path, reg_sha)
    ft = torch.load(out / "controls.pt", weights_only=False)
    b = torch.load(base / "controls.pt", weights_only=False)
    assert ft["layer"] == b["layer"], ("layer", ft["layer"], b["layer"])
    missing = sorted(set(b["vectors"]) ^ set(ft["vectors"]))
    assert not missing, f"ftdiag: controls in only one of OUT and BASE: {missing}"
    cos = {v: float(torch.nn.functional.cosine_similarity(ft["vectors"][v].float(), b["vectors"][v].float(), dim=0))
           for v in sorted(b["vectors"])}
    ratio = {v: ft["raw_norm"][v] / b["raw_norm"][v] for v in cos}
    keys = ("vclass", "a_max_auroc", "a_label", "a_null_p90", "a_pass", "b_emotion_labels", "b_kept", "b_pass",
            "d_cos_mean_emotion", "pass")
    gates = {r["control"]: {k: r[k] for k in keys} for r in rep["rows"]}
    ckpt = out / "ckpt.sha256"
    bc = ft["build_check"]
    d = {"stage": "ftdiag", "note": "diagnostics only (spec section 3): S2-FTD and S2-FTC steer with the base vectors",
         "checkpoint": meta["model"], "base_checkpoint": bmeta["model"], "model_key": meta["model_key"],
         "layer": ft["layer"],
         "checkpoint_files_sha256": sha_file(ckpt) if ckpt.exists() else None,
         "base_controls": rel(base / "controls.pt"), "base_controls_sha256": sha_file(base / "controls.pt"),
         "ft_controls_sha256": sha_file(out / "controls.pt"),
         "build_check": bc, "registered_match": bc["pass"],
         "registered_cosine": {k: r["cosine"] for k, r in bc["rows"].items() if "cosine" in r},
         "cosine_ft_vs_base": cos, "norm_ratio_ft_over_base": ratio, "gates": gates,
         "pool_on_ft_weights": rep["pool"]}
    (out / "diag.json").write_text(json.dumps(d, indent=1) + "\n")
    print(f"{'control':24s} {'cos(base)':>9s} {'norm':>6s} gate")
    for v in cos:
        print(f"{v:24s} {cos[v]:9.4f} {ratio[v]:6.3f} {gates[v]['pass']}")
    print(f"registered red/ocean/arithmetic rebuilt on {meta['model']}: "
          + ", ".join(f"{k} cos {c:.4f}" for k, c in d["registered_cosine"].items())
          + f"; neutral resid_norm rel diff {bc['rows']['neutral_resid_norm']['rel_diff']:.4g}"
          + f"; build check {'matches' if bc['pass'] else 'does not match'} (not enforced)", flush=True)
    return d


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    r = sub.add_parser("run")
    r.add_argument("model")
    r.add_argument("out")
    r.add_argument("--model-key", required=True, choices=["qwen", "k26"])
    r.add_argument("--stories", action="append", required=True, help="a runs/s2_stories.py output dir; repeatable")
    r.add_argument("--layers", default="", help="extra layers to collect besides the injection layer")
    r.add_argument("--local", default=None, help="run in-process on this device (cpu) instead of kimisroom.run")
    r.add_argument("--local-dtype", default="float32", choices=["float32", "bfloat16"], help="with --local")
    r.add_argument("--allow-dryrun", action="store_true", help="tests only")
    b = sub.add_parser("build")
    b.add_argument("out")
    g = sub.add_parser("gate")
    g.add_argument("out")
    g.add_argument("--pool", type=int, default=12)
    f = sub.add_parser("ftdiag", help="fine-tuned checkpoint diagnostic (S2-FTD, S2-FTC)")
    f.add_argument("out", help="a `run` output on the merged checkpoint")
    f.add_argument("--base", required=True, help="the base model's finished output dir (controls.pt)")
    f.add_argument("--pool", type=int, default=12)
    a = ap.parse_args(argv)
    if a.stage == "run":
        return stage_run(a)
    if a.stage == "build":
        return build(Path(a.out))
    assert 12 <= a.pool <= 24, "--pool is 12 (registered) up to 24 (rule 3)"
    if a.stage == "ftdiag":
        return ftdiag(Path(a.out), Path(a.base), a.pool)
    return gate(Path(a.out), a.pool)


if __name__ == "__main__":
    main()
