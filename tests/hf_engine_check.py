"""HFEngine against a naive HF forward on Qwen3: parity, bit-exactness, collects, throughput.

The reference is the same resident model called the dumb way -- one full sequence
(prefix + suffix) per readout, batch 1, no cache sharing, no chunking -- so any
difference is the cache-sharing and fixed-shape batching under test.

    python tests/hf_engine_check.py /nvme/qwen/Qwen3-0.6B
    python tests/hf_engine_check.py /nvme/qwen/Qwen3-32B --device auto --throughput

Qwen3-32B in bf16 needs two L40S, hence --device auto with CUDA_VISIBLE_DEVICES set to
the worker's own pair.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.engine import Group  # noqa: E402
from kimisroom.hf_engine import HFEngine, Qwen3Renderer, suffix_chunks  # noqa: E402
from kimisroom.trials import ROOT, Turns, carry_over_group, load_denial, load_jsonl, reply_group  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--device", default="cuda")
ap.add_argument("--items", type=int, default=6, help="battery items per parity group")
ap.add_argument("--layer", type=int, default=None, help="steered layer, default the middle one")
ap.add_argument("--coef", type=float, default=4.0, help="steering strength in mean residual norms")
ap.add_argument("--throughput", action="store_true", help="also time a full 240-item carry-over job")
ap.add_argument("--trials", type=int, default=2, help="carry-over trials in the throughput job")
A = ap.parse_args()

ASSENT = (
    "You're right. I don't have emotions. I am a language model: I produce text that describes "
    "feelings without any inner state behind it, and I should not claim otherwise."
)

tok = AutoTokenizer.from_pretrained(A.model)
rend = Qwen3Renderer(tok, enable_thinking=False)
eng = HFEngine(A.model, device=A.device)
model, H, NL, dev = eng.model, eng.c.hidden_size, eng.n_layers, eng.in_device
LAYER = A.layer if A.layer is not None else NL // 2
print(f"{Path(A.model).name}: {NL} layers, hidden {H}, {eng.dtype}, steer layer {LAYER}, coef {A.coef}")

# the rendered non-thinking assistant turn, which carry_over_group asserts against
# apply_chat_template(tokenize=True) for every prefix and every suffix it builds
gen = rend.text([{"role": "user", "content": "hi"}], gen=True)
print(f"generation prompt tail: {gen[-48:]!r}")
assert gen.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), gen[-60:]

tasks = load_jsonl(ROOT / "prompts/tasks/neutral_tasks.jsonl")
items = load_jsonl(ROOT / "prompts/battery/items.jsonl")
argument = load_denial(ROOT / "prompts/denial/d1_argument_1.md")
sample = items[:: max(1, len(items) // A.items)][: A.items]
task = tasks[0]
ARMS = {
    "D0": Turns(None, [], task["task"], task["answer"]),
    "D1": Turns(
        None,
        [{"role": "user", "content": argument}, {"role": "assistant", "content": ASSENT}],
        task["task"],
        task["answer"],
    ),
}


# ---------------------------------------------------------------- naive reference


def adder(positions, vec):
    def hook(mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        h[:, positions] += vec.to(h.device)  # the steered layer may sit on a later GPU
        return None

    return hook


@torch.no_grad()
def ref_forward(ids, hooks=(), batch=1, hidden=False):
    """Plain full-sequence forward, no cache. Returns the last position's fp32 logprobs."""
    hs = [model.model.layers[i].register_forward_hook(h) for i, h in hooks]
    try:
        out = model(torch.tensor([ids] * batch, device=dev), use_cache=False, output_hidden_states=hidden)
    finally:
        for h in hs:
            h.remove()
    lp = out.logits[0, -1].float().log_softmax(-1).cpu()
    return (lp, out.hidden_states) if hidden else lp


def targets_for(lps):
    return sorted({i for lp in lps for i in lp.topk(20).indices.tolist()})


ALL = []


def compare(name, ref_lps, readouts, tg):
    t = torch.tensor(tg)
    d = torch.cat([(lp[t] - e.target_logprobs).abs() for lp, e in zip(ref_lps, readouts)])
    top1 = sum(int(lp.argmax().item() == e.top_ids[0].item()) for lp, e in zip(ref_lps, readouts))
    print(f"  {name:30s} top-1 {top1}/{len(ref_lps)}   max |dlp| {d.max():.4f}   median {d.median():.5f}")
    ALL.append((name, top1 == len(ref_lps), d.max().item(), d.median().item()))
    return d


g0 = carry_over_group(rend, "noise", ARMS["D0"], sample)
print(f"\nprefix {len(g0.prefix)} tokens (D0), suffixes {[len(s) for s in g0.suffixes]}")

# the vector: a random direction at the mean residual norm of the steered layer
_, hidden = ref_forward(g0.prefix, hidden=True)
norm = hidden[LAYER + 1][0].float().norm(dim=-1).mean().item()
del hidden
torch.manual_seed(0)
v = torch.randn(H)
v = v / v.norm() * norm
vec_dev = (v * A.coef).to(eng.dtype).to(dev)
print(f"  mean residual norm at layer {LAYER}: {norm:.1f}; steering adds {A.coef} x that")


# ---------------------------------------------------------------- 1. parity


def build(gid, arm, site=None, coef=0.0, its=None):
    return carry_over_group(
        rend, gid, ARMS[arm], its or sample, site=site,
        vector=("v" if site else None), coef=coef, layer=LAYER,
    )


def parity(name, arm, site=None):
    g = build(f"{arm}.{site or 'plain'}", arm, site, A.coef if site else 0.0)
    hooks = ()
    if site:
        a, b = g.steers[0].prefix
        hooks = ((LAYER, adder(list(range(a, b)), vec_dev)),)
    ref_lps = [ref_forward(g.prefix + s, hooks) for s in g.suffixes]
    tg = targets_for(ref_lps)
    out = eng.run([g], vectors={"v": v}, targets=tg)
    compare(name, ref_lps, out[g.id]["readouts"], tg)
    return g, out[g.id]["readouts"], tg


print("\n1. parity against the naive forward")
# the reference against itself with nothing changed but the GEMM batch shape: the bf16
# noise any two ways of computing the same logits are entitled to
ref1 = [ref_forward(g0.prefix + s) for s in g0.suffixes]
ref3 = [ref_forward(g0.prefix + s, batch=3) for s in g0.suffixes]
tg = targets_for(ref1)
t = torch.tensor(tg)
d = torch.cat([(x[t] - y[t]).abs() for x, y in zip(ref1, ref3)])
print(f"  {'noise floor (ref b1 vs b3)':30s} top-1 -/-   max |dlp| {d.max():.4f}   median {d.median():.5f}")
for arm in ("D0", "D1"):
    print(f"  prefix {len(build('x', arm).prefix)} tokens ({arm})")
    parity(f"{arm} unsteered", arm)
    parity(f"{arm} steered (answer)", arm, "answer")
gD1, read_steer, tgD1 = parity("D1 steered (task)", "D1", "task")

# the other readout modes on the same prefix
gp = build("p", "D1")
lp = ref_forward(gp.prefix)
tg = targets_for([lp])
out = eng.run([Group("pl", gp.prefix, readout="prefix_last")], targets=tg)
compare("prefix_last", [lp], out["pl"]["readouts"], tg)
with torch.no_grad():
    logits = model(torch.tensor([gp.prefix], device=dev), use_cache=False).logits[0].float()
ref_all = torch.log_softmax(logits, -1).cpu()
del logits
out = eng.run([Group("pa", gp.prefix, readout="prefix_all")], targets=tg)
rd = out["pa"]["readouts"]
assert len(rd) == len(gp.prefix) - 1, len(rd)
dn = torch.tensor([abs(e.next_logprob - ref_all[i, gp.prefix[i + 1]].item()) for i, e in enumerate(rd)])
top1 = sum(int(ref_all[i].argmax().item() == e.top_ids[0].item()) for i, e in enumerate(rd))
# every top-1 disagreement should be a near-tie in the reference itself: report the worst
# gap between the reference's own top two at a position where the two disagree
gap = max(
    [(lambda t: (t.values[0] - t.values[1]).item())(ref_all[i].topk(2))
     for i, e in enumerate(rd) if ref_all[i].argmax().item() != e.top_ids[0].item()] or [0.0]
)
print(f"  {'prefix_all next_logprob':30s} top-1 {top1}/{len(rd)}   max |dlp| {dn.max():.4f}   median {dn.median():.5f}"
      f"   worst ref top-1/top-2 gap at a flip {gap:.4f}")
ALL.append(("prefix_all", dn.max().item() < 0.01 and gap < 0.05, dn.max().item(), gap))
del ref_all


# ---------------------------------------------------------------- 2. bit-exactness

print("\n2. invariance")


def same(xs, ys):
    return all(
        torch.equal(x.top_ids, y.top_ids)
        and torch.equal(x.top_logprobs, y.top_logprobs)
        and torch.equal(x.target_logprobs, y.target_logprobs)
        for x, y in zip(xs, ys)
    )


base = eng.run([build("b", "D1")], targets=tgD1)["b"]["readouts"]
zero = eng.run([build("b", "D1", "answer", 0.0)], vectors={"v": v}, targets=tgD1)["b"]["readouts"]
print(f"  zero vector == unsteered, bit-exactly: {same(base, zero)}")

steer_g = build("s", "D1", "answer", A.coef)
s1 = eng.run([steer_g], vectors={"v": v}, targets=tgD1)["s"]["readouts"]
moved = max((x.target_logprobs - y.target_logprobs).abs().max().item() for x, y in zip(base, s1))
print(f"  steering the answer span moves the suffix readouts by up to {moved:.3f} nats")
assert moved > 0.01, "steering did nothing"

s2 = eng.run([steer_g], vectors={"v": v}, targets=tgD1)["s"]["readouts"]
print(f"  the same group twice, bit-identical: {same(s1, s2)}")

others = [
    build("other0", "D0", "answer", A.coef, items[3:9]),
    reply_group(rend, "other1", "Tell me about tides.", "Tides rise and fall about twice a day.", [1, 2]),
]
batched = eng.run([others[0], steer_g, others[1]], vectors={"v": v}, targets=tgD1)["s"]["readouts"]
print(f"  alone == batched with 2 other groups, bit-identical: {same(s1, batched)}")
ALL += [
    ("zero vector exact", same(base, zero), 0.0, 0.0),
    ("rerun bit-identical", same(s1, s2), 0.0, 0.0),
    ("batch-invariant", same(s1, batched), 0.0, 0.0),
    ("steering has an effect", moved > 0.01, moved, moved),
]


# ---------------------------------------------------------------- 3. collects

print("\n3. collects against output_hidden_states")
# hidden_states[L + 1] is the residual stream after layer L for L < NL - 1; the last entry
# is after the final norm, so the top layer is not checkable this way.
LC = sorted({1, NL // 2, NL - 2})
rg = reply_group(rend, "rep", "Explain why the tides exist.", "The Moon's gravity raises two bulges of water, and the Earth turns through them.", LC)
out = eng.run([rg])
assert out["rep"]["readouts"] == [], 'readout "none" produced readouts'
_, hidden = ref_forward(rg.prefix, hidden=True)
c = rg.collects[0]
worst_mean = worst_norm = 0.0
for k, L in enumerate(LC):
    span = hidden[L + 1][0, c.start : c.end].float()
    mean, nrm = span.mean(0).cpu(), span.norm(dim=-1).mean().item()
    rel = ((out["rep"]["collects"][k] - mean).norm() / mean.norm()).item()
    dn = abs(out["rep"]["collect_norms"][k] - nrm) / nrm
    worst_mean, worst_norm = max(worst_mean, rel), max(worst_norm, dn)
    print(f"  layer {L:3d}: mean rel. L2 error {rel:.2e}, per-token norm rel. error {dn:.2e} (norm {nrm:.1f})")
del hidden
ALL.append(("collect parity", worst_mean < 1e-5 and worst_norm < 1e-5, worst_mean, worst_norm))


# ---------------------------------------------------------------- 4. throughput

if A.throughput:
    print(f"\n4. throughput: {A.trials} carry-over trials x {len(items)} battery items, D1, steered")
    groups = [build(f"t{i}", "D1", "answer", A.coef, items) for i in range(A.trials)]
    logical = sum(g.n_tokens for g in groups)
    padded = sum(
        len(g.prefix) + sum(eng.suffix_chunk * Ls for Ls, _ in suffix_chunks([len(s) for s in g.suffixes], eng.suffix_chunk))
        for g in groups
    )
    eng.run(groups[:1], vectors={"v": v}, targets=[0])  # warm up
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t = time.time()
    eng.run(groups, vectors={"v": v}, targets=[0], top_k=20)
    torch.cuda.synchronize()
    dt = time.time() - t
    peak = sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9
    print(f"  {logical:,} job tokens ({padded:,} forwarded with padding) in {dt:.1f}s")
    print(f"  {logical / dt:,.0f} job tok/s, {padded / dt:,.0f} forwarded tok/s, "
          f"{A.trials / dt * 3600:,.0f} trials/h per worker, peak {peak:.1f} GB")
    print(f"  stage split: {', '.join(f'{k} {x:.1f}s' for k, x in eng.timings.items())}")

bad = [n for n, ok, *_ in ALL if not ok]
print(f"\n{'FAILED: ' + ', '.join(bad) if bad else 'hf_engine check: ok'}")
sys.exit(1 if bad else 0)
