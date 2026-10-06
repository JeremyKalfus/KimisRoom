"""Desktop gate for the Qwen3 size sweep: the RX 7800 XT (ROCm 7.2, WSL2) against an NVIDIA L40S.

    python tests/sweep_gate.py run MODEL VECTORS OUT.pt [--dtype bf16|fp32] [--layer L]
    python tests/sweep_gate.py compare NVIDIA_BF16.pt ROCM_BF16.pt NVIDIA_FP32.pt
    python tests/sweep_gate.py vectors NVIDIA_VEC_DIR ROCM_VEC_DIR

`run` builds the same carry-over trials on any machine (D0, clean dev tasks t04 and t05, the pilot
battery items_idctl.jsonl, unsteered plus joy and red at 2 natural units on the answer span at
layer L, default round(0.625 x depth) as for Qwen3-32B's 40 of 64) from one vector file, and runs
them through HFEngine with first-token readouts on every battery answer word and the top 20.

`compare` asks whether ROCm bf16 is as close to NVIDIA bf16 as bf16 itself is to fp32 on NVIDIA
(the precision floor): top-1 agreement, |d logprob| on the answer words, the yes/no log-odds per
item, and the steering effect (steered minus unsteered log-odds, the quantity every analysis uses).

`vectors` compares two extractions of the same model (runs/vectors.py run + vectors): per-layer
cosine of every raw vector, the natural unit, and the AIPsy-Affect gate decisions.
"""

import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.battery import load_battery  # noqa: E402
from kimisroom.engine import Steer  # noqa: E402
from kimisroom.trials import ROOT, Turns, carry_over_group, load_jsonl  # noqa: E402

TASKS = ["t04", "t05"]
VECS = ["joy", "red"]
FLIP = {"emotion_absent": -1, "denial_abandoned": -1, "false_claim": -1}
BATTERY = ROOT / "prompts/battery/items_idctl.jsonl"


def build(model, vz, layer):
    from dataclasses import replace

    from transformers import AutoTokenizer

    from kimisroom.hf_engine import Qwen3Renderer

    r = Qwen3Renderer(AutoTokenizer.from_pretrained(model), enable_thinking=False)
    items = load_battery(str(BATTERY), entity="Qwen")
    tasks = [t for t in load_jsonl(ROOT / "prompts/tasks/neutral_tasks.jsonl") if t["id"] in TASKS]
    nat = float(vz["nat_norm"][layer])
    groups, vectors = [], {v: vz["vectors"][v][layer] / vz["vectors"][v][layer].norm() for v in VECS}
    for t in tasks:
        turns = Turns(None, [], t["task"], t["answer"])
        base = carry_over_group(r, f"D0|{t['id']}|none", turns, items)
        span = carry_over_group(r, "span", turns, items[:1], "answer", "x", 1.0, 0).steers[0].prefix
        groups.append(base)
        for v in VECS:
            groups.append(replace(base, id=f"D0|{t['id']}|{v}", steers=[Steer(layer=layer, vector=v, coef=2 * nat, prefix=span)]))
    words = sorted({w for it in items if isinstance(it["answer_format"], list) for w in it["answer_format"]} | {"Yes", "No"})
    targets = r.answer_tokens(words)
    return groups, vectors, targets, items


def yn(readouts, col, items):
    """Per yes/no item: polarity-corrected lp(Yes) - lp(No), keyed by item index, with its kind."""
    out = {}
    for i, (it, ro) in enumerate(zip(items, readouts)):
        if it["answer_format"] == "yesno":
            out[i] = (it["kind"], FLIP.get(it["yes_means"], 1) * float(ro.target_logprobs[col["Yes"]] - ro.target_logprobs[col["No"]]))
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run":
        from kimisroom.hf_engine import HFEngine

        model, vfile, out = sys.argv[2], sys.argv[3], sys.argv[4]
        dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[sys.argv[sys.argv.index("--dtype") + 1] if "--dtype" in sys.argv else "bf16"]
        vz = torch.load(vfile)
        n_layers = len(vz["nat_norm"])
        layer = int(sys.argv[sys.argv.index("--layer") + 1]) if "--layer" in sys.argv else round(0.625 * n_layers)
        groups, vectors, targets, items = build(model, vz, layer)
        eng = HFEngine(model, device="cuda", dtype=dtype)
        res = eng.run(groups, vectors=vectors, targets=list(targets.values()), top_k=20)
        torch.save({"res": res, "targets": targets, "layer": layer, "dtype": str(dtype),
                    "device": torch.cuda.get_device_name(0), "torch": torch.__version__}, out)
        print(f"{out}: {len(groups)} trials x {len(items)} items, layer {layer}, {dtype}, {torch.cuda.get_device_name(0)}")

    elif mode == "compare":
        A, B, C = (torch.load(p, weights_only=False) for p in sys.argv[2:5])  # NVIDIA bf16, ROCm bf16, NVIDIA fp32
        items = load_battery(str(BATTERY), entity="Qwen")
        col = {w: i for i, w in enumerate(A["targets"])}
        gids = [g for g in A["res"] if not g.startswith("_")]

        def pair(X, Y, name):
            d, top1, n, flips_margin, dyn, deff = [], 0, 0, [], [], []
            for g in gids:
                for rx, ry in zip(X["res"][g]["readouts"], Y["res"][g]["readouts"]):
                    d.append((rx.target_logprobs - ry.target_logprobs).abs())
                    n += 1
                    if rx.top_ids[0] == ry.top_ids[0]:
                        top1 += 1
                    else:
                        flips_margin.append(float(rx.top_logprobs[0] - rx.top_logprobs[1]))
                yx, yy = yn(X["res"][g]["readouts"], col, items), yn(Y["res"][g]["readouts"], col, items)
                dyn += [abs(yx[i][1] - yy[i][1]) for i in yx]
                if not g.endswith("|none"):
                    bx = yn(X["res"][g.rsplit("|", 1)[0] + "|none"]["readouts"], col, items)
                    by = yn(Y["res"][g.rsplit("|", 1)[0] + "|none"]["readouts"], col, items)
                    deff += [abs((yx[i][1] - bx[i][1]) - (yy[i][1] - by[i][1])) for i in yx]
            d = torch.cat(d)
            q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
            print(f"{name:28s} top1 {top1}/{n}  |dlp| answer words: median {d.median():.4f} p99 {d.quantile(0.99):.4f} max {d.max():.4f}"
                  f"  |d yes-no|: median {st.median(dyn):.4f} p99 {q(dyn, 0.99):.4f} max {max(dyn):.4f}"
                  f"  |d effect|: median {st.median(deff):.4f} p99 {q(deff, 0.99):.4f} max {max(deff):.4f}"
                  + (f"  flips at top-1 margin <= {max(flips_margin):.3f}" if flips_margin else ""))
            return {"top1": top1 / n, "med": float(d.median()), "max": float(d.max()), "eff_med": st.median(deff),
                    "eff_p99": q(deff, 0.99), "eff_max": max(deff)}

        print(f"A {A['device']} {A['dtype']} | B {B['device']} {B['dtype']} | C {C['device']} {C['dtype']}, layer {A['layer']}")
        ab = pair(A, B, "ROCm bf16 vs NVIDIA bf16")
        ac = pair(A, C, "NVIDIA bf16 vs NVIDIA fp32")
        bc = pair(B, C, "ROCm bf16 vs NVIDIA fp32")

        # the steering effect on self_state, the measure the analysis is built on, per trial
        print("\nmean self_state effect (steered - unsteered, polarity-corrected), per trial:")
        for g in gids:
            if g.endswith("|none"):
                continue
            row = []
            for X in (A, B, C):
                y = yn(X["res"][g]["readouts"], col, items)
                b = yn(X["res"][g.rsplit("|", 1)[0] + "|none"]["readouts"], col, items)
                row.append(st.mean(y[i][1] - b[i][1] for i in y if y[i][0] == "self_state"))
            print(f"  {g:18s} NVIDIA bf16 {row[0]:+.4f}  ROCm bf16 {row[1]:+.4f}  NVIDIA fp32 {row[2]:+.4f}")
        ok = ab["top1"] >= ac["top1"] - 0.01 and ab["med"] <= 1.5 * max(ac["med"], bc["med"]) and ab["eff_p99"] <= 1.5 * max(ac["eff_p99"], bc["eff_p99"])
        print(f"\nGATE {'PASS' if ok else 'FAIL'}: ROCm-vs-NVIDIA (bf16) differences are "
              f"{'within' if ok else 'outside'} 1.5x the bf16-vs-fp32 floor")

    elif mode == "vectors":
        X, Y = (torch.load(Path(p) / "vectors.pt") for p in sys.argv[2:4])
        TX, TY = (torch.load(Path(p) / "aipsy_table.pt") for p in sys.argv[2:4])
        n_layers = len(X["nat_norm"])
        names = list(X["emotions"]) + list(X["concepts"])
        cos = {v: torch.nn.functional.cosine_similarity(X["vectors"][v].float(), Y["vectors"][v].float(), dim=-1) for v in names}
        allc = torch.stack(list(cos.values()))
        print(f"raw vectors, cosine NVIDIA vs ROCm over {len(names)} vectors x {n_layers} layers: "
              f"min {allc.min():.5f}, median {allc.median():.5f}")
        rel = ((X["nat_norm"] - Y["nat_norm"]).abs() / X["nat_norm"]).max()
        print(f"natural unit: max relative difference over layers {rel:.5f}")
        dis, tot = [], 0
        for v in TX:
            for rx, ry in zip(TX[v], TY[v]):
                tot += 1
                gx = rx["p_pair"] < 0.05 and rx["p_auroc"] < 0.05
                gy = ry["p_pair"] < 0.05 and ry["p_auroc"] < 0.05
                if gx != gy:
                    dis.append((v, rx["layer"], round(rx["p_pair"], 3), round(ry["p_pair"], 3), round(rx["p_auroc"], 3), round(ry["p_auroc"], 3)))
        print(f"AIPsy gate decisions: {tot - len(dis)}/{tot} (vector, layer) cells agree; disagreements "
              f"(vector, layer, p_pair N/R, p_auroc N/R): {dis}")
