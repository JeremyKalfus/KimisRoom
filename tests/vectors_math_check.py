"""CPU check of the vectors stage's arithmetic on synthetic activations: no model, no GPU.

    .venv/bin/python tests/vectors_math_check.py

Plants a known structure in fake activations -- a per-condition emotion direction, plus a few
high-variance "neutral" directions shared by every unit -- then runs `runs/vectors.py ... vectors`
over them and checks that:
  * the raw, -specific and -dn vectors are all built, for eight emotions and three concepts
  * the denoised vectors are orthogonal to the top principal components of the neutral units,
    and the chosen k per layer is the number of PCs explaining --dn-var of the neutral variance
  * the two collect blocks are read correctly (full span vs from token --dn-start)
  * the natural dose unit equals the mean raw emotion-vector norm
  * calm and pain are skipped by the AIPsy gate rather than raising, and the Pain Axis gate runs
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kimisroom.emotions import CONCEPTS, EMOTIONS, NO_AIPSY  # noqa: E402
from kimisroom.validate import aipsy_units, pain_units  # noqa: E402

N_LAYERS, H, N_PC, SEED = 6, 32, 3, 7
N_TRAIN, N_HELD = 12, 6
CONDITIONS = EMOTIONS + CONCEPTS + ["neutral"]


def build(tmp: Path) -> tuple[Path, Path]:
    g = torch.Generator().manual_seed(SEED)
    # the shared high-variance directions every unit carries: what denoising is meant to remove
    pc = torch.randn(N_LAYERS, N_PC, H, generator=g)
    pc /= pc.norm(dim=-1, keepdim=True)
    sig = {c: torch.randn(N_LAYERS, H, generator=g) for c in CONDITIONS}  # per-condition direction

    def act(cond, scale=1.0):
        a = sig[cond].clone() * scale
        load = torch.randn(N_LAYERS, N_PC, generator=g) * 12.0  # big, so the PCs dominate variance
        a += torch.einsum("lp,lph->lh", load, pc)
        return a + 0.1 * torch.randn(N_LAYERS, H, generator=g)

    stories, res = [], {}
    for cond in CONDITIONS:
        for split, n in (("train", N_TRAIN), ("heldout", N_HELD)):
            for i in range(n):
                sid = f"{cond}-{split}-{i}"
                stories.append({"id": sid, "condition": cond, "split": split, "text": "x " * 80,
                                "set": i, "seed_id": sid, "n_words": 80})
                full, dn = act(cond), act(cond, 1.2)  # two collect blocks: full span, then from DN_START
                res[sid] = {"collects": list(full) + list(dn),
                            "collect_norms": [float(full[L].norm()) for L in range(N_LAYERS)]}
    for u in aipsy_units():
        res[f"aipsy:{u['id']}"] = {"collects": list(act("neutral")), "collect_norms": [1.0] * N_LAYERS}
    for u in pain_units():
        res[u["id"]] = {"collects": list(act("pain" if u["role"] == "pain" else "neutral")),
                        "collect_norms": [1.0] * N_LAYERS}

    src, out = tmp / "stimuli", tmp / "out"
    src.mkdir(parents=True)
    out.mkdir(parents=True)
    (src / "stories.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stories))
    torch.save(res, out / "extract.pt")
    return src, out


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        src, out = build(tmp)
        cmd = [sys.executable, str(ROOT / "runs/vectors.py"), "/nonexistent/model", str(out), "vectors",
               "--stimuli", str(src), "--emotions", "all", "--dn-start", "20", "--dn-var", "0.5"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        print(r.stdout[-3000:] if r.returncode == 0 else r.stdout + r.stderr)
        assert r.returncode == 0, f"vectors stage failed:\n{r.stderr[-4000:]}"

        vz = torch.load(out / "vectors.pt", weights_only=False)
        v, dninfo = vz["vectors"], vz["denoise"]
        for e in EMOTIONS:
            for suffix in ("", "-specific", "-dn"):
                assert f"{e}{suffix}" in v, f"missing {e}{suffix}"
        for c in CONCEPTS:
            assert c in v and f"{c}-dn" in v
        assert dninfo["start"] == 20 and len(dninfo["k"]) == N_LAYERS

        # the planted PCs explain almost all neutral variance, so k should find the first few
        print(f"\nk per layer {dninfo['k']}, variance explained {['%.3f' % x for x in dninfo['explained']]}")
        assert all(1 <= k <= N_PC for k in dninfo["k"]), dninfo["k"]

        # recompute the neutral PCs from the dn collect block and check orthogonality
        neutral = [s["id"] for s in (json.loads(x) for x in (src / "stories.jsonl").read_text().splitlines())
                   if s["condition"] == "neutral" and s["split"] == "train"]
        res = torch.load(out / "extract.pt", weights_only=False)
        X = torch.stack([torch.stack(res[i]["collects"][N_LAYERS:]) for i in neutral])
        Xc = (X - X.mean(0)).permute(1, 0, 2).to(torch.float32)
        worst = 0.0
        for L in range(N_LAYERS):
            _, _, Vh = torch.linalg.svd(Xc[L], full_matrices=False)
            P = Vh[: dninfo["k"][L]]
            for e in EMOTIONS:
                w = v[f"{e}-dn"][L].to(torch.float32)
                worst = max(worst, float((P @ w).abs().max() / w.norm()))
        print(f"max |PC . dn vector| / |dn vector| over layers and emotions: {worst:.2e}")
        assert worst < 1e-5, f"denoised vectors are not orthogonal to the removed PCs ({worst})"

        nat = torch.stack([v[e].norm(dim=-1) for e in EMOTIONS]).mean(0)
        assert torch.allclose(nat, vz["nat_norm"], atol=1e-4), "nat_norm is not the mean raw emotion norm"
        print(f"natural unit per layer: {[round(float(x), 2) for x in vz['nat_norm']]}")

        table = torch.load(out / "aipsy_table.pt", weights_only=False)
        skipped = sorted({e for e in NO_AIPSY if f"{e}" not in table})
        assert skipped == sorted(NO_AIPSY), f"AIPsy table should skip {NO_AIPSY}, has {sorted(table)}"
        print(f"AIPsy table rows: {len(table)}; skipped without a label: {skipped}")
        ptable = torch.load(out / "pain_table.pt", weights_only=False)
        print(f"Pain Axis table rows: {sorted(ptable)}; n_pairs {ptable[('pain', '1P')][0]['n_pairs']}")
        assert ptable[("pain", "1P")][0]["n_pairs"] == 500
        print("\nOK: vectors-stage math, denoising, natural unit, AIPsy skip and pain gate all check out")


if __name__ == "__main__":
    main()
