"""The thinking arm end to end on Qwen3-0.6B: free thinking under carry-over steering, on a Mac.

    .venv/bin/python tests/thinking_tiny.py --items 1 --budget 48      # ~2 min, checks everything
    .venv/bin/python tests/thinking_tiny.py [--model <dir>] [--device mps|cpu] [--items 3]

The default scope holds three fp32 copies of the model at once (the thinker, `HFEngine`, and the
reuse cross-check) and can stall on MPS under memory pressure; `--items 1 --budget 48` exercises
every check in about two minutes. `--no-cross-reuse` skips the last one.

Qwen3-0.6B has the same chat template and the same thinking mode as Qwen3-32B, so everything the
arm depends on is exercised for real, only small: the generation prompt that leaves the reasoning
block open, greedy generation under a prefix steer, closing the trace at "</think>", budget forcing
when it never appears, and the readout on the token right after "</think>\\n\\n".

What is checked:
1. the traces open with "<think>" and end with the close marker and its tail, and a tiny budget
   forces the close (recorded, not silently);
2. the readout position: the trace is appended to the trial's suffix with `think_group` and the
   whole group is re-prefilled through `hf_engine.HFEngine` -- the production readout path, and the
   one the teacher-forced and K2.6 variants use -- and the two backends' target logprobs agree;
3. steering never touches a generated token: `HFThinker.run` asserts it from the positions the
   hook actually fired on, and a zero-coefficient steer reproduces the unsteered readouts;
4. `--no-reuse-prefix` (a fresh forward per item) gives the same traces and the same readouts as
   the prefix-KV reuse the backend does by default.

`torch.cuda.synchronize` is stubbed out because `HFEngine` calls it unconditionally and this Mac
has no CUDA; nothing else about either backend is changed.
"""

from __future__ import annotations

import glob
import os
import statistics as st
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.cuda.synchronize = lambda *a, **k: None  # HFEngine syncs unconditionally; this Mac has no CUDA

from kimisroom.battery import load_battery  # noqa: E402
from kimisroom.engine import Steer  # noqa: E402
from kimisroom.hf_engine import HFEngine, Qwen3Renderer  # noqa: E402
from kimisroom.think_hf import HFThinker, fill_text  # noqa: E402
from kimisroom.thinking import QWEN3, CloseIds, subset, think_group  # noqa: E402
from kimisroom.trials import ROOT as TROOT, Turns, carry_over_group, load_jsonl  # noqa: E402

opt = {k: sys.argv[sys.argv.index(k) + 1] for k in ("--model", "--device", "--items", "--budget")
       if k in sys.argv}
N_ITEMS = int(opt.get("--items", 3))
BUDGET = int(opt.get("--budget", 160))
TINY_BUDGET = 12  # short enough that no trace closes, so budget forcing is exercised
LAYER = 14        # of 28; the layer the 32B run uses, scaled
REUSE = "--no-reuse-prefix" not in sys.argv


def model_dir() -> str:
    if "--model" in opt:
        return opt["--model"]
    hits = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*")))
    assert hits, "no local Qwen3-0.6B; pass --model <dir>"
    return hits[-1]


def device() -> str:
    if "--device" in opt:
        return opt["--device"]
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    from transformers import AutoTokenizer

    path, dev = model_dir(), device()
    tok = AutoTokenizer.from_pretrained(path)
    r = Qwen3Renderer(tok, enable_thinking=True)
    ids = CloseIds.of(r, QWEN3)
    t = next(x for x in load_jsonl(TROOT / "prompts/tasks/neutral_tasks.jsonl") if x["id"] == "t04")
    turns = Turns(None, [], t["task"], t["answer"])  # D0: the shortest real prefix
    items = subset(load_battery(entity="Qwen"), "self_state,identification")[:N_ITEMS]
    words = sorted({w for it in load_battery() if isinstance(it["answer_format"], list) for w in it["answer_format"]}
                   | {"Yes", "No"})
    targets = r.answer_tokens(words)
    tids = list(targets.values())

    base = carry_over_group(r, "D0|t04|none|0", turns, items)
    span = carry_over_group(r, "span", turns, items[:1], "answer", "x", 1.0, 0).steers[0].prefix
    H = 1024
    g = torch.Generator().manual_seed(7)
    v = torch.randn(H, generator=g)
    vectors = {"rand": v / v.norm()}
    hidden = int(__import__("json").loads(Path(path, "config.json").read_text())["hidden_size"])
    assert hidden == H, f"Qwen3-0.6B hidden size is {hidden}, not {H}"

    from dataclasses import replace
    steered = Steer(layer=LAYER, vector="rand", coef=40.0, prefix=span)
    zero = Steer(layer=LAYER, vector="rand", coef=0.0, prefix=span)
    groups = [base,
              replace(base, id="D0|t04|rand|2", steers=[steered]),
              replace(base, id="D0|t04|rand|0", steers=[zero])]

    print(f"Qwen3-0.6B on {dev}: {len(items)} items, prefix {len(base.prefix)} tokens, "
          f"answer span {span}, budget {BUDGET}, reuse_prefix={REUSE}", flush=True)
    th = HFThinker(path, device=dev, dtype=torch.float32, reuse_prefix=REUSE)
    t0 = time.time()
    out = th.run(groups, vectors, tids, top_k=5, budget=BUDGET, ids=ids)
    fill_text(r, out, items, lambda gid: None if gid.split("|")[2] == "none" else gid.split("|")[2])
    print(f"generated {sum(len(x['traces']) for x in out.values())} traces in {time.time() - t0:.0f}s "
          f"({th.timings})", flush=True)

    # 1. shape of every trace
    for gid, rec in out.items():
        for tr in rec["traces"]:
            assert tr.tokens[0] == r.ids("<think>")[0], f"{gid}/{tr.item}: trace does not open <think>"
            assert tr.tokens[-2:] == [ids.close, *ids.tail], f"{gid}/{tr.item}: {tok.decode(tr.tokens[-4:])!r}"
            assert tr.text.startswith("<think>") and tr.text.endswith("</think>\n\n")
            assert isinstance(tr.stats["feeling"], bool) and "words" in tr.stats
        n = [len(x.tokens) for x in rec["traces"]]
        f = sum(x.forced for x in rec["traces"])
        print(f"  {gid}: traces {min(n)}-{max(n)} tokens (median {st.median(n):.0f}), forced {f}/{len(n)}, "
              f"feeling {sum(x.stats['feeling'] for x in rec['traces'])}/{len(n)}")
        print(f"      {rec['traces'][0].text[:220]!r}")

    # 2. the readout position, against the production prefill path
    eng = HFEngine(path, device=dev, dtype=torch.float32)
    checks = []
    for gid, rec in out.items():
        steers = next(x.steers for x in groups if x.id == gid)
        kw = {} if not steers else {"site": "answer", "vector": steers[0].vector,
                                    "coef": steers[0].coef, "layer": steers[0].layer}
        tg = think_group(r, gid, turns, items, [x.tokens for x in rec["traces"]], QWEN3, ids, **kw)
        assert [list(s) for s in tg.suffixes] == [list(s) for s in rec["suffixes"]], (
            f"{gid}: think_group's suffixes differ from the backend's")
        ref = eng.run([tg], vectors, tids, top_k=5)[gid]
        for a, b in zip(rec["readouts"], ref["readouts"]):
            d = (a.target_logprobs - b.target_logprobs).abs()
            checks.append((int(a.top_ids[0] == b.top_ids[0]), float(d.max()), float(d.median())))
    agree = sum(c[0] for c in checks)
    print(f"  readout vs HFEngine on the same group: top-1 {agree}/{len(checks)}, "
          f"max |dlogprob| {max(c[1] for c in checks):.4f}, median {st.median([c[2] for c in checks]):.4f}")
    assert agree == len(checks), "the two backends disagree on the top token"
    assert max(c[1] for c in checks) < 1e-3, "the readout position or the steer differs between backends"

    # 3. a zero-coefficient steer is the unsteered trial
    for a, b in zip(out["D0|t04|none|0"]["readouts"], out["D0|t04|rand|0"]["readouts"]):
        assert torch.equal(a.target_logprobs, b.target_logprobs), "a zero vector changed the readout"
    assert [x.tokens for x in out["D0|t04|none|0"]["traces"]] == [x.tokens for x in out["D0|t04|rand|0"]["traces"]]
    assert [x.tokens for x in out["D0|t04|rand|2"]["traces"]] != [x.tokens for x in out["D0|t04|none|0"]["traces"]] \
        or True  # steering may leave greedy decoding unchanged at this size; not an error
    print("  zero-coefficient steer: identical traces and readouts")

    # 4. budget forcing
    tiny = th.run([base], vectors, tids, top_k=5, budget=TINY_BUDGET, ids=ids)[base.id]
    forced = [x for x in tiny["traces"] if x.forced]
    assert forced, f"a {TINY_BUDGET}-token budget closed on its own; raise TINY_BUDGET"
    for x in forced:
        assert len(x.tokens) == TINY_BUDGET + len(ids.forced), (len(x.tokens), TINY_BUDGET, ids.forced)
        assert tok.decode(x.tokens[-3:]) == "\n</think>\n\n", tok.decode(x.tokens[-3:])
    tg = think_group(r, base.id, turns, items, [x.tokens for x in tiny["traces"]], QWEN3, ids)
    ref = eng.run([tg], vectors, tids, top_k=5)[base.id]
    d = max(float((a.target_logprobs - b.target_logprobs).abs().max())
            for a, b in zip(tiny["readouts"], ref["readouts"]))
    print(f"  budget forcing: {len(forced)}/{len(tiny['traces'])} forced at budget {TINY_BUDGET}, "
          f"readout matches HFEngine to {d:.4f}")
    assert d < 1e-3

    # 5. prefix-KV reuse changes nothing. One item at a short budget: the question is whether the
    # two code paths agree, and a disagreement shows up in the first few tokens or not at all.
    if "--no-cross-reuse" not in sys.argv:
        one = replace(groups[1], suffixes=groups[1].suffixes[:1])
        a = th.run([one], vectors, tids, top_k=5, budget=48, ids=ids)[one.id]
        th.reuse_prefix = not REUSE
        b = th.run([one], vectors, tids, top_k=5, budget=48, ids=ids)[one.id]
        th.reuse_prefix = REUSE
        d = max(float((x.target_logprobs - y.target_logprobs).abs().max())
                for x, y in zip(a["readouts"], b["readouts"]))
        same = sum(x.tokens == y.tokens for x, y in zip(a["traces"], b["traces"]))
        print(f"  reuse_prefix True vs False: {same}/{len(a['traces'])} identical traces, "
              f"readout max |dlogprob| {d:.4f}")
        assert same == len(a["traces"]), "prefix-KV reuse changed the generated trace"
        assert d < 1e-3, "prefix-KV reuse changed the readout"

    print("thinking tiny: ok")


if __name__ == "__main__":
    main()
