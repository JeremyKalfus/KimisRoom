"""Check of the potency moderator (runs/s2_run.py potency) on Qwen3-0.6B (bf16, MPS if present, else CPU; no GPU).

    .venv/bin/python tests/s2_potency_check.py [--work DIR] [--no-r]

Stand-ins as in tests/s2_run_check.py: a random emotion file and a random controls.pt of the 0.6B model's width,
at a test layer (14 of 28), natural unit from that file. Two emotions (joy, sadness) and two nouns (N01, N02) at
doses 2 and 1 on 3 neutral texts.

1. The stage writes potency.csv with the columns runs/s2_confirm.R reads (model, vec, dose, potency).
2. Each value matches an independent reference: the same texts through transformers with a forward hook adding
   dose x unit x the unit vector after layer 14 at every position, fp32 log_softmax, mean NLL difference.
3. A rerun reuses potency.pt and writes the same file.
4. (R) runs/s2_confirm.R --potency accepts the file on a synthetic Qwen table: its "potency, emotions - primary
   controls" row equals mean(joy, sadness) - mean(N01, N02) at the registered dose (2).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))
import s2_run as RUN  # noqa: E402

PY = sys.executable
LAYER = 14
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FAILS: list[str] = []


def ok(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def main() -> None:
    work = Path(sys.argv[sys.argv.index("--work") + 1]) if "--work" in sys.argv else Path(tempfile.mkdtemp(prefix="s2pot_"))
    work.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    mp = snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True)
    cfg = json.loads((Path(mp) / "config.json").read_text())
    H, n = cfg["hidden_size"], cfg["num_hidden_layers"]
    g = torch.Generator().manual_seed(0)
    emo = ["joy", "sadness", "anger", "fear", "euphoria", "surprise"]
    vz = {"emotions": emo, "vectors": {k: torch.randn(n + 1, H, generator=g) for k in emo + ["arithmetic", "random", "red", "ocean"]}}
    torch.save(vz, work / "vectors.pt")
    ctl = work / "controls"
    ctl.mkdir(exist_ok=True)
    cv = {k: torch.randn(H, generator=g) for k in ("N01", "N02")}
    torch.save({"layer": LAYER, "vectors": cv, "info": {k: {"vclass": "noun"} for k in cv}}, ctl / "controls.pt")
    (ctl / "pool.json").write_text(json.dumps({"nouns_chosen": ["N01", "N02"], "thinking_survivors": []}))
    out = work / "potency"
    cmd = [PY, "runs/s2_run.py", "potency", "--model", "qwen", "--model-path", mp, "--out", str(out), "--local", DEVICE,
           "--test", "--layer", str(LAYER), "--emotion-vectors", str(work / "vectors.pt"), "--controls", str(ctl),
           "--vecs", "joy,sadness,N01,N02", "--doses", "2,1", "--potency-n", "3"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    ok(r.returncode == 0, "potency stage ran" + ("" if r.returncode == 0 else ": " + r.stderr[-3000:]))
    if r.returncode:
        return finish()
    print("     " + r.stdout.strip().splitlines()[-1])
    rows = list(csv.DictReader((out / "potency.csv").open()))
    ok({"model", "vec", "dose", "potency"} <= set(rows[0]) and len(rows) == 8,
       f"potency.csv: {len(rows)} rows (expected 8), columns {list(rows[0])}")
    pot = {(x["vec"], float(x["dose"])): float(x["potency"]) for x in rows}
    ok(all(pot[v, 2.0] > pot[v, 1.0] > 0 for v in ("joy", "sadness", "N01", "N02")),
       "every pattern raises NLL, more at dose 2 than at dose 1: " + ", ".join(f"{k[0]}@{k[1]:g} {v:.3f}" for k, v in pot.items()))

    # 2. independent reference: transformers forward with a hook, fp32 log_softmax of the logits
    from transformers import AutoModelForCausalLM, AutoTokenizer
    meta = json.loads((out / "potency_meta.json").read_text())
    unit = meta["natural_unit"]
    tok = AutoTokenizer.from_pretrained(mp)
    model = AutoModelForCausalLM.from_pretrained(mp, dtype=torch.bfloat16).to(DEVICE).eval()
    add = {"v": None}

    def hook(mod, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if add["v"] is not None:
            h += add["v"].to(h.device, h.dtype)
        return None

    model.model.layers[LAYER].register_forward_hook(hook)
    texts = {s["id"]: s["text"] for s in RUN.potency_texts(3)}
    ok(list(texts) == meta["texts"], f"texts: every second neutral train story by id {list(texts)}")

    @torch.no_grad()
    def nll(text: str, v: torch.Tensor | None) -> float:
        ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device=DEVICE)
        add["v"] = v
        lp = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1)
        add["v"] = None
        return -lp.gather(1, ids[0, 1:, None]).mean().item()

    raw = {"joy": vz["vectors"]["joy"][LAYER], "sadness": vz["vectors"]["sadness"][LAYER], **cv}
    base = {k: nll(t, None) for k, t in texts.items()}
    worst = 0.0
    for (v, d), p in pot.items():
        u = (raw[v].float() / raw[v].float().norm() * d * unit).to(torch.bfloat16)
        ref = sum(nll(t, u) - base[k] for k, t in texts.items()) / len(texts)
        worst = max(worst, abs(ref - p))
    ok(worst < 0.02, f"potency matches the transformers reference within 0.02 nats (max |diff| {worst:.4f})")

    # 3. rerun: reuses potency.pt
    before = (out / "potency.csv").read_bytes()
    t = (out / "potency.pt").stat().st_mtime
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    ok(r.returncode == 0 and (out / "potency.pt").stat().st_mtime == t and (out / "potency.csv").read_bytes() == before,
       "a rerun reuses potency.pt and writes the same potency.csv")

    if "--no-r" in sys.argv:
        return finish()
    # 4. s2_confirm.R --potency on a synthetic Qwen table
    syn = work / "synth.csv"
    subprocess.run([PY, "tests/s2_synth.py", str(syn), "--models", "qwen", "--arms", "S2-DEN,S2-CON", "--n-args", "4",
                    "--n-wordings", "2"], cwd=ROOT, check=True, capture_output=True)
    subprocess.run([PY, "runs/s2_table.py", "--phase", "confirm", "--out", str(work / "table.csv"), str(syn)],
                   cwd=ROOT, check=True, capture_output=True)
    r = subprocess.run(["Rscript", "runs/s2_confirm.R", str(work / "r_out"), str(work / "table.csv"), "--no-loo",
                        "--potency", str(out / "potency.csv")], cwd=ROOT, capture_output=True, text=True)
    ok(r.returncode == 0, "s2_confirm.R ran with --potency" + ("" if r.returncode == 0 else ": " + r.stderr[-2000:]))
    if r.returncode == 0:
        D = list(csv.DictReader((work / "r_out/descriptive.csv").open()))
        x = next((x for x in D if x["model"] == "qwen" and x["what"] == "potency, emotions - primary controls"), None)
        want = (pot["joy", 2.0] + pot["sadness", 2.0]) / 2 - (pot["N01", 2.0] + pot["N02", 2.0]) / 2
        ok(x is not None and abs(float(x["estimate"]) - want) < 1e-6,
           f"descriptive.csv potency row {x and x['estimate']} = mean(emotions) - mean(nouns) at dose 2 ({want:.6f})")
    finish()


def finish() -> None:
    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nall checks passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
