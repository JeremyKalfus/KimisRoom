"""vLLM vs HFEngine on the same carry-over groups, thinking off. Required before any thinking run.

    # on the GPU box, in the vLLM venv (see runs/thinking_runbook.md)
    python tests/vllm_parity.py /nvme/qwen/Qwen3-32B --vectors results/qwen/v2/vectors.pt \\
        --layer 40 --dose 2 --tp 2 [--items 24] [--readout targets|topk] [--side both|vllm|hf]

vLLM 0.30 needs transformers >= 5.10 and `HFEngine` runs on the DLAMI's 4.57.1 (5.x dropped
`DynamicCache.to_legacy_cache`), so on the box the two halves run in their own environments:
`--side vllm` in the vLLM venv saves its readouts to `<out>/vllm_readouts.pt`, then `--side hf` in
`/opt/pytorch` rebuilds the same groups, runs `HFEngine` and compares. `both` (the default) does it
in one process where one environment carries both.

The thinking arm reads first-token logprobs after a reasoning block, which only vLLM can produce
at grid scale, but every number the pilot has published came out of `hf_engine.HFEngine`. This
script measures the one thing that has to hold for the two to be comparable: with thinking off --
no generation, pure prefill, the readout the pilot already uses -- the vLLM backend's first-token
logprobs must match HFEngine's within bf16 noise, unsteered and steered, under D0 and D1.1.

The two differ in almost every implementation detail that could move a bf16 logit (paged vs dense
attention, fused vs eager kernels, flattened vs padded batches, and a steer added to a Qwen3
layer's deferred MLP output rather than to the post-layer hidden state), so this is a real test of
the whole vLLM path, not a formality. The reference bar is the one `tests/hf_engine_check.py`
established against a naive full-sequence forward: top-1 agreement everywhere, median |dlogprob|
of order 0.1 nats and max of order 0.5, against a bf16 self-noise floor of 0.5 max / 0.125 median.

It also prints the hook's own report per condition: how many positions were steered and the lowest
and highest of them, which must sit inside the answer span.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kimisroom.battery import load_battery  # noqa: E402
from kimisroom.engine import Steer  # noqa: E402
from kimisroom.texts import denial_arms  # noqa: E402
from kimisroom.thinking import DEFAULT_SUBSET, subset  # noqa: E402
from kimisroom.trials import ROOT as TROOT, Turns, carry_over_group, load_jsonl  # noqa: E402

TASKS = ["t04", "t05"]
ARMS = ["D0", "D1.1"]
VECS = ["joy", "sadness", "red"]
BAR = {"top1": 1.0, "median": 0.30, "max": 1.00}  # a generous multiple of the HFEngine reference
# A top-1 flip between two tokens whose logprobs are equal within this in *both* backends is an
# argmax tie, not a disagreement (tests/hf_engine_check.py's prefix_all flips were all of this kind).
TIE = 0.0625


def main() -> None:
    from transformers import AutoTokenizer

    from kimisroom.hf_engine import HFEngine, Qwen3Renderer

    model = sys.argv[1]
    out = Path(sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else ".")
    opt = {k: sys.argv[sys.argv.index(k) + 1] for k in
           ("--vectors", "--layer", "--dose", "--tp", "--items", "--readout", "--max-model-len", "--side")
           if k in sys.argv}
    L, dose, tp = int(opt.get("--layer", 40)), float(opt.get("--dose", 2.0)), int(opt.get("--tp", 2))
    r = Qwen3Renderer(AutoTokenizer.from_pretrained(model), enable_thinking=False)
    vz = torch.load(opt["--vectors"], weights_only=False)
    nat = float(vz["nat_norm"][L]) if "nat_norm" in vz else float(
        torch.stack([vz["vectors"][e][L].norm() for e in vz["emotions"]]).mean())
    vectors = {n: vz["vectors"][n][L] / vz["vectors"][n][L].norm() for n in VECS}
    coef = dose * nat

    items = subset(load_battery(entity="Qwen"), DEFAULT_SUBSET)
    if "--items" in opt:
        items = items[: int(opt["--items"])]
    words = sorted({w for it in load_battery() if isinstance(it["answer_format"], list) for w in it["answer_format"]}
                   | {"Yes", "No"})
    targets = r.answer_tokens(words)
    tids = list(targets.values())
    tasks = [t for t in load_jsonl(TROOT / "prompts/tasks/neutral_tasks.jsonl") if t["id"] in TASKS]
    arms = denial_arms("qwen3-32b", ARMS)

    groups = []
    for arm, cond in arms.items():
        for t in tasks:
            turns = Turns(None, cond, t["task"], t["answer"])
            base = carry_over_group(r, f"{arm}|{t['id']}|none|{L}|0", turns, items)
            span = carry_over_group(r, "span", turns, items[:1], "answer", "x", 1.0, 0).steers[0].prefix
            groups.append(base)
            for v in VECS:
                groups.append(replace(base, id=f"{arm}|{t['id']}|{v}|{L}|{dose}",
                                      steers=[Steer(layer=L, vector=v, coef=coef, prefix=span)]))
    print(f"{len(groups)} conditions x {len(items)} items, layer {L}, dose {dose} x natural unit "
          f"{nat:.1f} (coef {coef:.1f}), {sum(g.n_tokens for g in groups):,} tokens", flush=True)

    side = opt.get("--side", "both")
    saved = out / "vllm_readouts.pt"
    if side == "hf":
        got = torch.load(saved, weights_only=False)
        assert sorted(got) == sorted(g.id for g in groups), f"{saved} was made from different groups"
    else:
        got = vllm_side(model, opt, groups, vectors, tids, tp)
        if side == "vllm":
            out.mkdir(parents=True, exist_ok=True)
            torch.save(got, saved)
            print(f"wrote {saved}; now run --side hf in the HFEngine environment")
            return

    eng = HFEngine(model, device="auto" if tp > 1 else "cuda")
    ref = eng.run(groups, vectors, tids, top_k=20)

    rows, worst, ties = [], [], []
    for g in groups:
        a, b = got[g.id]["readouts"], ref[g.id]["readouts"]
        top1 = [int(x.top_ids[0] == y.top_ids[0]) for x, y in zip(a, b)]
        d = torch.stack([(x.target_logprobs - y.target_logprobs).abs() for x, y in zip(a, b)])
        assert not torch.isnan(d).any(), f"{g.id}: a target logprob came back NaN (missing from the top-k?)"
        worst.append(float(d.max()))
        for i, (x, y) in enumerate(zip(a, b)):
            if x.top_ids[0] != y.top_ids[0]:
                # how close the two disputed tokens are in each backend (target logprobs, since the vLLM
                # readout keeps only its top-1): a gap within bf16 noise is a tie, not a disagreement
                u, w = int(x.top_ids[0]), int(y.top_ids[0])
                if u in tids and w in tids:
                    gap = lambda r: float(r.target_logprobs[tids.index(w)] - r.target_logprobs[tids.index(u)])  # noqa: E731
                    tie = abs(gap(x)) <= TIE and abs(gap(y)) <= TIE
                    top1[i] = int(tie)  # a tie in both backends is agreement, as in tests/hf_engine_check.py
                    if tie:
                        ties.append(g.id)
                    print(f"  top-1 flip {g.id} item {i}: vllm says {u}, hf says {w}; "
                          f"lp(hf's) - lp(vllm's) = {gap(x):+.3f} in vllm, {gap(y):+.3f} in hf"
                          f"{' -> tie, counted as agreement' if tie else ''}", flush=True)
                else:
                    print(f"  top-1 flip {g.id} item {i}: vllm says {u}, hf says {w} (not both targets)", flush=True)
        rows.append((g.id, sum(top1), len(top1), float(d.max()), float(d.median())))
    torch.save(ref, out / "hf_readouts.pt")
    print(f"\n{'condition':28s} {'top-1':>9s} {'max |dlp|':>10s} {'median':>8s}")
    for gid, ok, n, mx, md in rows:
        print(f"{gid:28s} {ok:4d}/{n:4d} {mx:10.3f} {md:8.3f}")
    n_ok = sum(x[1] for x in rows)
    n_all = sum(x[2] for x in rows)
    med = st.median([x[4] for x in rows])
    print(f"\ntotal: top-1 {n_ok}/{n_all} ({n_ok / n_all:.3f}, {len(ties)} exact ties counted as agreement), max |dlogprob| {max(worst):.3f}, "
          f"median of per-condition medians {med:.3f}")
    print(f"bar (HFEngine's own reference run x a margin): top-1 {BAR['top1']}, median <= {BAR['median']}, "
          f"max <= {BAR['max']}")
    verdict = n_ok == n_all and med <= BAR["median"] and max(worst) <= BAR["max"]
    (out / "vllm_parity.json").write_text(json.dumps({"rows": rows, "top1": [n_ok, n_all], "median": med, "max": max(worst),
                              "ties": ties, "layer": L, "dose": dose, "tp": tp, "pass": verdict}, indent=1))
    print("vllm parity: " + ("ok" if verdict else "FAILED -- do not run the thinking arm"))
    sys.exit(0 if verdict else 1)


def vllm_side(model, opt, groups, vectors, tids, tp) -> dict:
    from kimisroom.think_vllm import VLLMThinker

    vt = VLLMThinker(model, tensor_parallel_size=tp, readout=opt.get("--readout", "targets"),
                     max_model_len=int(opt["--max-model-len"]) if "--max-model-len" in opt else None)
    print(f"vllm workers: {vt.info}", flush=True)
    got = vt.readout(groups, vectors, tids, top_k=20)
    for gid in sorted(got):
        print(f"  steered {gid}: {got[gid]['steered']}", flush=True)
    del vt  # free the GPUs before the dense model loads
    torch.cuda.empty_cache()
    return got


if __name__ == "__main__":
    main()
