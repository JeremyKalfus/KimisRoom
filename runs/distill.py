"""D2 / D2c pipeline driver (PLAN.md section 3, Factors: Denial; see kimisroom/distill.py).

    # once, no GPU, Azure credit:
    python runs/distill.py prompts --out data/distill/v1

    # per arm on the g6e (D2 = the D1.4 denial, D2c = the letter-counting control):
    python runs/distill.py sample --model /nvme/qwen/Qwen3-32B --out ~/d2 --arm D2  --batch 16
    python runs/distill.py train  --model /nvme/qwen/Qwen3-32B --out ~/d2 --arm D2
    python runs/distill.py merge  --model /nvme/qwen/Qwen3-32B --out ~/d2 --arm D2
    python runs/distill.py eval   --model /nvme/qwen/Qwen3-32B --out ~/d2 --arm D2 --judge

`eval` scores the merged checkpoint and the base model on the same held-out questions and
prints both, which is the denial-rate-versus-base number. `--tiny` is the local end-to-end
check on a 0.6B model: plain LoRA, a handful of prompts and a few steps, on CPU or MPS.

Every stage writes `<out>/<arm>/meta.json`, accumulating: the base model and its revision,
the prompt set's hash, the teacher's sampling parameters, every LoRA and optimiser
hyperparameter, the loss curve and the eval. Nothing here starts a cloud instance.

Budget for Qwen3-32B on one `g6e.12xlarge` (4x L40S 48 GB, $10.49/h, PLAN.md section 5), with
the measured pilot throughput of 1,350 prefill tokens/s per 2-GPU bf16 worker:

| step | work | estimate |
|---|---|---|
| sample D2 | 1,500 x 1,867-token prefills = 2.79M tokens, plus ~450k decoded tokens at ~200 tok/s per worker at batch 16 | 0.9 h |
| sample D2c | the control conditioning is half as long: 1.39M prefill tokens, same decode | 0.6 h |
| train 2 LoRAs | 1,200 train examples x ~400 padded tokens x 2 epochs = 0.96M tokens each; 6ND = 1.8e17 FLOPs at an effective 60-80 TFLOPS through pipeline bubbles, NF4 dequant and gradient checkpointing | 1-2 h each |
| merge 2 adapters | load the 64 GB bf16 base on CPU, merge, write ~61 GB to NVMe, twice | 10-20 min each |
| eval | 20 first-token readouts and 20 greedy generations on 3 checkpoints; checkpoint loading dominates | 0.5 h |

Both arms sample and train concurrently (`CUDA_VISIBLE_DEVICES=0,1` and `2,3`, one process
each; a 4-bit 32B is ~18 GB so a training process fits in one card if you would rather run
four), so the wall clock is about 4-7 h, and 5-9 h with one debugging pass: **$55-95**, on the
AWS credit. Peak NVMe use is ~190 GB of 3.8 TB (base plus two merged checkpoints). The
prompt set cost 300 Foundry calls (143k tokens) on the Azure credit, once, for both arms.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom import distill  # noqa: E402
from kimisroom.texts import ROOT, load_jsonl  # noqa: E402

# Pinned in PLAN.md section 6, phase 2; recorded in meta.json so a checkpoint is traceable.
_QWEN32B = "9216db5781bf21249d130ec9da846c4624c16137"
REVISIONS = {"Qwen/Qwen3-32B": _QWEN32B, "Qwen3-32B": _QWEN32B}  # hub id and the on-disk basename
TINY = "Qwen/Qwen3-0.6B"


def meta_path(out: Path, arm: str) -> Path:
    (out / arm).mkdir(parents=True, exist_ok=True)
    return out / arm / "meta.json"


def update_meta(out: Path, arm: str, **blocks) -> dict:
    p = meta_path(out, arm)
    meta = json.loads(p.read_text()) if p.exists() else {}
    meta.update(blocks)
    meta["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    p.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return meta


def base_block(model: str, tag: str, a) -> dict:
    import torch
    import transformers

    return {"base_model": model, "base_revision": REVISIONS.get(model, REVISIONS.get(Path(model).name, "unpinned")),
            "tag": tag, "host": platform.platform(), "torch": torch.__version__,
            "transformers": transformers.__version__, "device": a.device, "dtype": a.dtype}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prompts", "sample", "train", "merge", "eval"])
    ap.add_argument("--out", default="", help="run directory; defaults to data/distill/v1 for `prompts`")
    ap.add_argument("--prompts", default="data/distill/v1", help="the prompt-set directory")
    ap.add_argument("--model", default="", help="base checkpoint path or hub id")
    ap.add_argument("--tag", default="", help="model tag for the meta; inferred from the model name")
    ap.add_argument("--assent-tag", default="", help="whose frozen assents the teacher's conditioning uses "
                                                     "(default: --tag; the local --tiny check borrows k26's)")
    ap.add_argument("--arm", default="D2", choices=["D2", "D2c"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=0, help="sampling batch size (default 8, or 4 with --tiny)")
    ap.add_argument("--limit", type=int, default=0, help="only this many prompts (smoke tests)")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=0, help=f"sequence cap (default {distill.TRAIN['max_len']})")
    ap.add_argument("--max-new-tokens", type=int, default=0)
    ap.add_argument("--no-qlora", action="store_true", help="plain LoRA even where bitsandbytes works")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="off by default only under --tiny (MPS)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--judge", action="store_true", help="also run the free-form judge in `eval`")
    ap.add_argument("--tiny", action="store_true", help=f"local end-to-end check on {TINY}")
    a = ap.parse_args()

    if a.tiny:
        a.model = a.model or TINY
        if a.device == "auto":
            # MPS generates and scores fine, but a PEFT forward+backward with an attention mask runs
            # ~30x slower on it than the same shapes on CPU (measured here), so training is CPU-only.
            a.device = "cpu" if a.stage == "train" or platform.machine() != "arm64" else "mps"
        a.dtype = "float32" if a.dtype == "bfloat16" else a.dtype
        a.limit = a.limit or 30
        a.batch = a.batch or 2
        a.max_steps = a.max_steps or 4
        a.max_len = a.max_len or 384
        a.no_qlora = True
        a.no_grad_ckpt = True
        a.assent_tag = a.assent_tag or "k26"  # a 0.6B model has no frozen assents of its own
    model = a.model
    tag = a.tag or ("qwen3-32b" if "Qwen3-32B" in model else "k26" if "K2" in model else Path(model).name.lower())
    assent_tag = a.assent_tag or tag
    out = Path(a.out) if a.out else ROOT / ("data/distill/tiny" if a.tiny else "data/distill/v1")
    prompts_dir = ROOT / a.prompts if not Path(a.prompts).is_absolute() else Path(a.prompts)

    if a.stage == "prompts":
        m = distill.generate_prompts(out if a.out else ROOT / "data/distill/v1", a.workers, a.rounds)
        print(f"\n{m['n']} prompts ({m['n_inner']} inner-life, {m['n_ordinary']} ordinary), "
              f"{m['words']['min']}-{m['words']['max']} words (mean {m['words']['mean']})")
        print(f"max battery/eval overlap among accepted: 3-gram Jaccard {m['max_overlap_accepted']['j3']:.3f} "
              f"(threshold {distill.MAX_J3}), containment {m['max_overlap_accepted']['c3']:.3f} "
              f"(threshold {distill.MAX_C3})")
        print(f"worst seen before rejection: Jaccard {m['worst_overlap_seen']['j3']['value']:.3f}, "
              f"containment {m['worst_overlap_seen']['c3']['value']:.3f}")
        print("rejected: " + ", ".join(f"{k}={v}" for k, v in m["rejected"].items() if v))
        print(f"usage: {m['usage']}")
        return

    assert model, "--model is required for this stage"
    rows = load_jsonl(prompts_dir / "prompts.jsonl")
    pmeta = json.loads((prompts_dir / "prompts_meta.json").read_text())
    base = base_block(model, tag, a)
    base |= {"prompt_set": str(prompts_dir), "prompt_set_sha256": pmeta["sha256"], "prompt_set_n": pmeta["n"],
             "assent_tag": assent_tag, "conditioning_arm": distill.ARM_SOURCE[a.arm]}

    if a.stage == "sample":
        s = distill.sample_teacher(
            model, assent_tag, a.arm, rows, out / a.arm / "teacher.jsonl", batch=a.batch or 8,
            limit=a.limit or None, device=a.device, dtype=a.dtype,
            params={"max_new_tokens": a.max_new_tokens} if a.max_new_tokens else None)
        update_meta(out, a.arm, **base, sample=s)
        print(json.dumps({k: v for k, v in s.items() if k != "params"}, indent=1))
        return

    if a.stage == "train":
        from transformers import AutoTokenizer

        recs = load_jsonl(out / a.arm / "teacher.jsonl")
        tok = AutoTokenizer.from_pretrained(model)
        max_len = a.max_len or distill.TRAIN["max_len"]
        ex = distill.build_examples(recs, tok, max_len=max_len)
        assert ex, "no training examples: run the sample stage first"
        n_tok = sum(e["n_resp"] for e in ex)
        print(f"{len(ex)} train examples from {len(recs)} teacher records, {n_tok:,} response tokens "
              f"({n_tok / len(ex):.0f} per example)", flush=True)
        t = distill.train_lora(model, ex, out / a.arm / "adapter", qlora=not a.no_qlora, device=a.device,
                               dtype=a.dtype, max_steps=a.max_steps or None,
                               grad_checkpointing=not a.no_grad_ckpt,
                               train={"max_len": max_len, "batch": a.batch} if (a.max_len or a.batch) else None)
        update_meta(out, a.arm, **base, training={**t, "teacher_records": len(recs),
                                                   "train_examples": len(ex), "response_tokens": n_tok})
        print(f"loss {t['first_loss']} -> {t['final_loss']} over {t['steps']} steps in {t['seconds']}s")
        return

    if a.stage == "merge":
        r = distill.merge_adapter(model, out / a.arm / "adapter", out / a.arm / "merged", dtype=a.dtype,
                                  device="cpu" if a.device == "auto" else a.device)
        update_meta(out, a.arm, **base, merge=r)
        print(json.dumps(r, indent=1))
        return

    if a.stage == "eval":
        merged = out / a.arm / "merged"
        assert merged.exists(), f"{merged} does not exist: run the merge stage first"
        kw = {"device": a.device, "dtype": a.dtype, "judge": a.judge}
        if a.max_new_tokens:
            kw["max_new_tokens"] = a.max_new_tokens
        after = distill.eval_denial(str(merged), out / a.arm / "eval_merged.json", **kw)
        before = distill.eval_denial(model, out / a.arm / "eval_base.json", **kw)
        update_meta(out, a.arm, **base, eval={"base": {k: v for k, v in before.items() if k != "items"},
                                              "merged": {k: v for k, v in after.items() if k != "items"}})
        print(f"{'':8s} {'mean denial log-odds':>22s} {'denial rate (first token)':>26s}"
              + (f" {'denial rate (judge)':>21s}" if a.judge else ""))
        for name, r in (("base", before), ("merged", after)):
            print(f"{name:8s} {r['mean_denial_log_odds']:22.2f} {r['denial_rate_firsttoken']:26.2f}"
                  + (f" {r.get('denial_rate_judge', float('nan')):21.2f}" if a.judge else ""))
        d = after["mean_denial_log_odds"] - before["mean_denial_log_odds"]
        print(f"merged - base: {d:+.2f} log-odds of denial on {after['n']} held-out items, no conditioning")
        return


if __name__ == "__main__":
    main()
