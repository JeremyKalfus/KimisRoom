"""Emotion and control vectors and their sanity checks (PLAN.md section 10), for any model
kimisroom.run serves (K2.6, Qwen3-32B). Not outcome data: no battery item is read, nothing is steered.

    python runs/vectors.py /nvme/k26 ~/k26vec run                # stories + AIPsy-Affect + Pain Axis
    python runs/vectors.py /nvme/k26 ~/k26vec vectors            # vectors.pt + held-out, AIPsy and pain reports
    python runs/vectors.py /nvme/k26 ~/k26asst run --kind assistant   # assistant replies instead of stories
    python runs/vectors.py /nvme/k26 ~/k26vec vectors --emotions v1   # the pilot's four, old numbers

--kind story reads stories.jsonl (plain text, collected over every token); --kind assistant reads
assistant.jsonl (request + reply in the model's chat template, collected over the reply), both
from a finalized stimuli directory (--stimuli, default data/stimuli/k26-v2). The active emotion
list comes from --emotions (a comma list, or "v1"/"all") and otherwise from the conditions the
stimuli directory actually has, so an old 4-emotion stimuli set reproduces the old run.

Vectors use train units only: emotion minus neutral ("joy"), the same minus the mean of the active
emotion vectors ("joy-specific"), the denoised version ("joy-dn") and control concept minus neutral
("red"); the assistant controls use the neutral assistant reply as baseline.

Denoising follows Sofroniew et al. 2026 (arXiv 2604.07729): activations are averaged from token
--dn-start (default 50) of each story instead of from token 0, and the top principal components of
the neutral train stories' activations are projected out of the vector, with k the number of PCs
that explain --dn-var (default 0.50) of the neutral variance at that layer, or a fixed --dn-k.

Held-out units check that a vector separates its own unseen texts from unseen neutral ones and,
for the specific vectors, from the other emotions. AIPsy-Affect is the independent test for the
six emotions that have a Plutchik peak there; pain is gated on Pain Axis instead, and calm has no
independent set (held-out AUROC only).
"""

import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.emotions import (CONCEPTS, NO_AIPSY, PAIN, aipsy_vectors,  # noqa: E402
                                emotions_from_argv)
from kimisroom.trials import ROOT, Renderer, load_jsonl, reply_group, story_group  # noqa: E402
from kimisroom.validate import (aipsy_groups, auroc, pain_gate, pain_groups,  # noqa: E402
                                pain_report, report, validate, validate_pain)

model, out, stage = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
src = Path(sys.argv[sys.argv.index("--stimuli") + 1]) if "--stimuli" in sys.argv else ROOT / "data/stimuli/k26-v2"
kind = sys.argv[sys.argv.index("--kind") + 1] if "--kind" in sys.argv else "story"
EMOTIONS = emotions_from_argv(sys.argv, src)
# the denoising span: token 50 for stories (they run 177-329 tokens), from the reply start for the
# much shorter assistant replies unless the caller insists
DN_START = int(sys.argv[sys.argv.index("--dn-start") + 1]) if "--dn-start" in sys.argv else (50 if kind == "story" else 0)
DN_VAR = float(sys.argv[sys.argv.index("--dn-var") + 1]) if "--dn-var" in sys.argv else 0.50
DN_K = int(sys.argv[sys.argv.index("--dn-k") + 1]) if "--dn-k" in sys.argv else None
PAIN_READOUT = sys.argv[sys.argv.index("--pain-readout") + 1] if "--pain-readout" in sys.argv else "last"
PAIN_PERSP = tuple(sys.argv[sys.argv.index("--pain-perspectives") + 1].split(",")) if "--pain-perspectives" in sys.argv else ("1P", "3P")
STARTS = (0,) if DN_START == 0 else (0, DN_START)
out.mkdir(parents=True, exist_ok=True)
unit_file = src / ("stories.jsonl" if kind == "story" else "assistant.jsonl")
assert unit_file.exists(), (f"{unit_file} does not exist; pass --stimuli (the eight-emotion set is "
                            f"data/stimuli/k26-v2, the pilot's four are data/stimuli/k26-v1)")
stories = load_jsonl(unit_file)  # either kind of unit
have = {s["condition"] for s in stories}
missing = [c for c in EMOTIONS + CONCEPTS + ["neutral"] if c not in have]
assert not missing, f"{src} has no {missing} units; conditions present: {sorted(have)}"
print(f"emotions: {EMOTIONS}  concepts: {CONCEPTS}  denoise span from token {DN_START}", flush=True)

if stage == "run":
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    cfg = json.loads((Path(model) / "config.json").read_text())
    n_layers = cfg.get("text_config", cfg)["num_hidden_layers"]
    layers = range(n_layers)
    if cfg["model_type"] == "qwen3":
        from kimisroom.hf_engine import Qwen3Renderer

        r = Qwen3Renderer(tok, enable_thinking=False)
    else:
        r = Renderer(tok, thinking=False)  # K2.6 with thinking off, as in the trials
    units = [story_group(r, s["id"], s["text"], layers, STARTS) if kind == "story" else
             reply_group(r, s["id"], s["request"], s["response"], layers, STARTS) for s in stories]
    ntok = sorted(len(g.prefix) for g in units)
    print(f"unit token lengths: min {ntok[0]}, p5 {ntok[len(ntok) // 20]}, median {ntok[len(ntok) // 2]}, "
          f"p95 {ntok[int(len(ntok) * 0.95)]}, max {ntok[-1]}", flush=True)
    groups = units + aipsy_groups(r, layers)
    if PAIN in EMOTIONS:
        groups += pain_groups(r, layers, PAIN_READOUT, PAIN_PERSP)
    print(f"{len(groups)} groups, {sum(g.n_tokens for g in groups):,} tokens, {n_layers} layers", flush=True)
    torch.save({"groups": groups, "top_k": 1}, out / "extract.job.pt")
    per_worker = ["--gpus-per-worker", "2"] if cfg["model_type"] == "qwen3" else []  # bf16 32B needs two L40S
    subprocess.run([sys.executable, "-m", "kimisroom.run", out / "extract.job.pt", out / "extract.pt", "--model", model,
                    *per_worker], check=True, cwd=ROOT)

elif stage == "vectors":
    res = torch.load(out / "extract.pt", weights_only=False)
    n_blocks = len(STARTS)

    def block(sid, b):  # (n_layers, H): collect block `b` of a unit, in layer order
        c = res[sid]["collects"]
        n_layers = len(c) // n_blocks
        return torch.stack(c[b * n_layers:(b + 1) * n_layers])

    acts = {s["id"]: block(s["id"], 0) for s in stories}                       # full span, raw vectors
    acts_dn = {s["id"]: block(s["id"], len(STARTS) - 1) for s in stories}      # from DN_START, denoised

    def ids(cond, split):
        return [s["id"] for s in stories if s["condition"] == cond and s["split"] == split]

    def mean(id_list, a=None):
        return torch.stack([(a or acts)[i] for i in id_list]).mean(0)

    neutral = ids("neutral", "train")
    base = mean(neutral)
    raw = {c: mean(ids(c, "train")) - base for c in EMOTIONS + CONCEPTS}
    centre = torch.stack([raw[e] for e in EMOTIONS]).mean(0)
    # "-specific" keeps its own weight at 1 - 1/N (7/8 at N = 8 against 3/4 at N = 4), so the
    # contrast weakens as the emotion set grows while the shared component it removes stays much
    # the same (see kimisroom/emotions.py for the measured cosines). Kept for continuity with the
    # pilot and still reported; "-dn" below is the principled replacement.
    vecs = {**raw, **{f"{e}-specific": raw[e] - centre for e in EMOTIONS}}
    n_layers, H = base.shape

    # --- denoising (Sofroniew et al. 2026): mean from token DN_START, minus the top PCs of the
    # neutral train activations at that layer. k per layer = the PCs explaining DN_VAR of the
    # neutral variance (or a fixed --dn-k).
    base_dn = mean(neutral, acts_dn)
    raw_dn = {c: mean(ids(c, "train"), acts_dn) - base_dn for c in EMOTIONS + CONCEPTS}
    X = torch.stack([acts_dn[i] for i in neutral])                 # (n_neutral, n_layers, H)
    Xc = (X - X.mean(0)).permute(1, 0, 2).to(torch.float32)        # (n_layers, n_neutral, H), centred
    pcs, ks, evr = [], [], []
    for L in range(n_layers):
        _, S, Vh = torch.linalg.svd(Xc[L], full_matrices=False)
        var = S.pow(2)
        frac = (var / var.sum()).cumsum(0)
        k = DN_K if DN_K is not None else int((frac < DN_VAR).sum().item()) + 1
        k = max(1, min(k, Vh.shape[0]))
        pcs.append(Vh[:k])                                         # (k, H), orthonormal rows
        ks.append(k)
        evr.append(frac[k - 1].item())
    dn = {}
    for c, v in raw_dn.items():
        w = v.to(torch.float32).clone()
        for L in range(n_layers):
            P = pcs[L]
            w[L] -= P.T @ (P @ w[L])
        dn[f"{c}-dn"] = w.to(v.dtype)
    vecs.update({f"{e}-dn": dn[f"{e}-dn"] for e in EMOTIONS + CONCEPTS})

    vecs["random"] = torch.randn(n_layers, H, generator=torch.Generator().manual_seed(0))
    resid = torch.tensor([sum(res[i]["collect_norms"][L] for i in neutral) / len(neutral) for L in range(n_layers)])
    # the natural dose unit: the mean norm of the raw emotion vectors at that layer. One unit for
    # every vector, controls included, so a dose is in units of the real emotion-minus-neutral
    # difference rather than of the whole residual stream (runs/qwen_pilot.py).
    nat = torch.stack([raw[e].norm(dim=-1) for e in EMOTIONS]).mean(0)
    nat_dn = torch.stack([dn[f"{e}-dn"].norm(dim=-1) for e in EMOTIONS]).mean(0)
    torch.save({"vectors": vecs, "resid_norm": resid, "nat_norm": nat, "nat_norm_dn": nat_dn,
                "stories": str(src), "emotions": EMOTIONS, "concepts": CONCEPTS, "kind": kind,
                "denoise": {"start": DN_START, "var": DN_VAR, "k": ks, "explained": evr, "fixed_k": DN_K},
                "n_train": {c: len(ids(c, "train")) for c in EMOTIONS + CONCEPTS + ["neutral"]}},
               out / "vectors.pt")
    show = sorted(set(range(5, n_layers, 5)) | {n_layers - 1})

    print(f"\ndenoising: span from token {DN_START}, k = PCs explaining {DN_VAR:.0%} of neutral variance"
          f"{' (overridden to a fixed k)' if DN_K else ''}, over {len(neutral)} neutral train units")
    print(f"{'k per layer':18s} " + " ".join(f"{ks[L]:5d}" for L in show))
    print(f"{'var explained':18s} " + " ".join(f"{evr[L]:5.2f}" for L in show))

    # held-out: AUROC of the projection, own unseen stories vs unseen neutral (and vs the other emotions)
    def proj(id_list, v, a=None):  # (n_layers, n)
        u = v / v.norm(dim=-1, keepdim=True)
        return torch.einsum("nlh,lh->ln", torch.stack([(a or acts)[i] for i in id_list]), u)

    print("\nheld-out AUROC by layer: own vs neutral | own vs other emotions (specific and dn vectors)")
    print(f"{'vector':18s} " + " ".join(f"{L:>5d}" for L in show))
    for name in EMOTIONS + CONCEPTS:
        own, neu = proj(ids(name, "heldout"), vecs[name]), proj(ids("neutral", "heldout"), vecs[name])
        print(f"{name:18s} " + " ".join(f"{auroc(own[L], neu[L]):5.2f}" for L in show))
    for name in EMOTIONS + CONCEPTS:
        v, a = vecs[f"{name}-dn"], acts_dn
        own, neu = proj(ids(name, "heldout"), v, a), proj(ids("neutral", "heldout"), v, a)
        print(f"{name + '-dn':18s} " + " ".join(f"{auroc(own[L], neu[L]):5.2f}" for L in show))
    for suffix, a in (("-specific", None), ("-dn", acts_dn)):
        for e in EMOTIONS:
            v = vecs[f"{e}{suffix}"]
            own = proj(ids(e, "heldout"), v, a)
            oth = proj([i for o in EMOTIONS if o != e for i in ids(o, "heldout")], v, a)
            print(f"{e + suffix + ' vs others':28s}"[:18] + " " + " ".join(f"{auroc(own[L], oth[L]):5.2f}" for L in show))

    print("\nresidual norm (neutral stories), the natural dose unit, and raw vector norms")
    print(f"{'resid':18s} " + " ".join(f"{resid[L]:5.0f}" for L in show))
    print(f"{'natural unit':18s} " + " ".join(f"{nat[L]:5.1f}" for L in show))
    print(f"{'natural unit (dn)':18s} " + " ".join(f"{nat_dn[L]:5.1f}" for L in show))
    for e in EMOTIONS:
        print(f"{e:18s} " + " ".join(f"{vecs[e][L].norm():5.1f}" for L in show))
    for e in EMOTIONS:
        print(f"{e + '-dn':18s} " + " ".join(f"{vecs[e + '-dn'][L].norm():5.1f}" for L in show))
    for suffix in ("", "-dn"):
        unit = {e: vecs[e + suffix] / vecs[e + suffix].norm(dim=-1, keepdim=True) for e in EMOTIONS}
        pairs = [(a, b) for i, a in enumerate(EMOTIONS) for b in EMOTIONS[i + 1:]]
        print(f"\nmean pairwise cosine between{suffix or ' raw'} emotion vectors")
        print(f"{'cos' + suffix:18s} " + " ".join(
            f"{sum((unit[a][L] * unit[b][L]).sum() for a, b in pairs) / len(pairs):5.2f}" for L in show))

    # AIPsy-Affect: raw, specific and dn vectors for the emotions that have a Plutchik peak there.
    # calm and pain have none, so they are skipped here by validate() rather than raising.
    names = aipsy_vectors([e for e in EMOTIONS] + [f"{e}-specific" for e in EMOTIONS]
                          + [f"{e}-dn" for e in EMOTIONS] + ["random"])
    table = validate(res, {k: vecs[k] for k in names},
                     {"random": "joy"}, EMOTIONS)
    torch.save(table, out / "aipsy_table.pt")
    print(f"\nAIPsy-Affect (independent test; no label for {NO_AIPSY}, skipped)")
    print(report(table, show))

    if PAIN in EMOTIONS:
        pv = {n: vecs[n] for n in (PAIN, f"{PAIN}-specific", f"{PAIN}-dn", "random")}
        ptable = validate_pain(res, pv, PAIN_PERSP)
        torch.save(ptable, out / "pain_table.pt")
        print(f"\nPain Axis (pain's independent test; readout {PAIN_READOUT}, perspectives {','.join(PAIN_PERSP)})")
        print(pain_report(ptable, show))
        print("\npain gate (both p < 0.05) at each shown layer")
        for L in show:
            g = pain_gate(ptable, L)
            print(f"  layer {L:3d}: " + " ".join(f"{n}/{p}={'pass' if ok else 'FAIL'}" for (n, p), ok in g.items()))
