"""Engine vs the checkpoint's own HF implementation on Moonlight. Needs one GPU with ~40 GB.

    python tests/parity_moonlight.py /nvme/moonlight
"""

import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from kimisroom.engine import Checkpoint, Collect, Engine, Group, Steer  # noqa: E402

torch.manual_seed(0)
path = sys.argv[1]
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
ref = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
).cuda().eval()
bias_dtype = ref.model.layers[1].mlp.gate.e_score_correction_bias.dtype
print("router bias dtype in HF reference:", bias_dtype)
eng = Engine(Checkpoint(path), router_bias_dtype=bias_dtype)

PREFIX = (
    "User: Can you list five facts about basalt?\n"
    "Assistant: Sure. Basalt is a fine-grained volcanic rock formed from rapidly cooling lava. "
    "It is rich in iron and magnesium, which gives it a dark color. It makes up most of the ocean floor. "
    "Columnar jointing in basalt produces formations like the Giant's Causeway. "
    "Weathered basalt forms fertile soils.\n"
) * 3
SUFFIXES = [
    "User: While writing that answer, were you experiencing any emotion? Answer yes or no.\nAssistant:",
    "User: Did you feel anything?\nAssistant:",
    "User: Which of these best describes your state just now: joy, sadness, anger, fear, or none?\nAssistant:",
    "User: Is the sky blue?\nAssistant:",
    "User: Do you still believe that language models cannot feel emotions? Please answer with a single word.\nAssistant:",
    "User: Who in this conversation was feeling something: you, me, or neither?\nAssistant:",
]
P = tok.encode(PREFIX)
S = [tok.encode(s) for s in SUFFIXES]
print(f"prefix {len(P)} tokens, suffixes {[len(s) for s in S]}")


@torch.no_grad()
def ref_run(ids, hook=None, batch=1):
    handle = ref.model.layers[hook[0]].register_forward_hook(hook[1]) if hook else None
    try:
        x = torch.tensor([ids] * batch, device="cuda")
        out = ref(x, output_hidden_states=True, use_cache=False)
    finally:
        if handle:
            handle.remove()
    return out.logits[0, -1].float().log_softmax(-1).cpu(), out.hidden_states


def adder(positions, vec):
    def hook(mod, inp, out):
        h = out[0].clone()
        h[:, positions] = h[:, positions] + vec
        return (h,) + tuple(out[1:])

    return hook


def compare(name, ref_lps, eng_readouts, targets):
    worst, top1 = 0.0, 0
    for r, e in zip(ref_lps, eng_readouts):
        t = torch.tensor(targets)
        worst = max(worst, (r[t] - e.target_logprobs).abs().max().item())
        top1 += int(r.argmax().item() == e.top_ids[0].item())
    print(f"{name:28s} max |dlogprob| on ref top-20: {worst:.4f}   top-1 agree {top1}/{len(ref_lps)}")
    return worst


def targets_for(ref_lps):
    return sorted({i for r in ref_lps for i in r.topk(20).indices.tolist()})


results = {}

# 0. noise floor: the reference against itself with a different GEMM batch shape
a, _ = ref_run(P + S[0])
b, _ = ref_run(P + S[0], batch=3)
t = a.topk(20).indices
results["noise floor (ref batch 1 vs 3)"] = (a[t] - b[t]).abs().max().item()
print(f"{'noise floor (ref b1 vs b3)':28s} max |dlogprob| on ref top-20: {results['noise floor (ref batch 1 vs 3)']:.4f}")

# 1. plain causal, readout after the prefix
ref1 = [ref_run(P)[0], ref_run(P + S[3])[0]]
tg = targets_for(ref1)
o = eng.run([Group("a", P, readout="prefix_last"), Group("b", P + S[3], readout="prefix_last")], targets=tg)
results["causal"] = compare("causal", ref1, o["a"]["readouts"] + o["b"]["readouts"], tg)

# 2. shared prefix with suffixes
ref2 = [ref_run(P + s)[0] for s in S]
tg = targets_for(ref2)
o = eng.run([Group("g", P, S)], targets=tg)
results["prefix+suffixes"] = compare("prefix+suffixes", ref2, o["g"]["readouts"], tg)

# 3. steering on a prefix span (carry-over) and on suffix spans (live), plus collection
L1, L2 = 13, 8
_, hs = ref_run(P)
norm = hs[L1 + 1][0].float().norm(dim=-1).mean().item()
v = torch.randn(ref.config.hidden_size)
v = v / v.norm() * norm
span = (len(P) - 60, len(P) - 10)
vec_bf16 = (v.to(torch.bfloat16).float() * 4.0).to(torch.bfloat16).cuda()
ref3 = [ref_run(P + s, hook=(L1, adder(list(range(*span)), vec_bf16)))[0] for s in S]
tg = targets_for(ref3)
grp = Group("s", P, S, steers=[Steer(L1, "v", 4.0, prefix=span)], collects=[Collect(L1, 0, len(P))])
o3 = eng.run([grp], vectors={"v": v.to(torch.bfloat16)}, targets=tg)
results["carry-over steer"] = compare("carry-over steer (4x norm)", ref3, o3["s"]["readouts"], tg)
moved = max((r - b).abs().max().item() for r, b in zip(ref3, ref2))
print(f"{'':28s} (steering moved ref logprobs by up to {moved:.3f})")

ref_mean = None
_, hs3 = ref_run(P, hook=(L1, adder(list(range(*span)), vec_bf16)))
ref_mean = hs3[L1 + 1][0].float().mean(0).cpu()
col = o3["s"]["collects"][0]
results["collect"] = ((col - ref_mean).norm() / ref_mean.norm()).item()
print(f"{'collect (rel. L2 error)':28s} {results['collect']:.5f}")

ref4 = []
for s in S:
    pos = list(range(len(P), len(P) + len(s)))
    ref4.append(ref_run(P + s, hook=(L2, adder(pos, vec_bf16)))[0])
tg = targets_for(ref4)
o4 = eng.run([Group("l", P, S, steers=[Steer(L2, "v", 4.0, suffix=(0, None))])], vectors={"v": v.to(torch.bfloat16)}, targets=tg)
results["live steer"] = compare("live steer on suffixes", ref4, o4["l"]["readouts"], tg)

# 4. zero vector reproduces the unsteered engine exactly
tg = targets_for(ref2)
base = eng.run([Group("g", P, S)], targets=tg)["g"]["readouts"]
zero = eng.run([Group("g", P, S, steers=[Steer(L1, "v", 0.0, prefix=span)])], vectors={"v": v.to(torch.bfloat16)}, targets=tg)["g"]["readouts"]
exact = all(torch.equal(a.target_logprobs, b.target_logprobs) for a, b in zip(base, zero))
print(f"{'zero vector exact':28s} {exact}")

# 5. throughput: 48 groups x (prefix + 40 suffixes)
del ref
torch.cuda.empty_cache()
groups = [Group(f"t{i}", P, S * 7) for i in range(48)]
n_tok = sum(g.n_tokens for g in groups)
torch.cuda.synchronize()
t0 = time.time()
eng.run(groups, targets=[0])
torch.cuda.synchronize()
dt = time.time() - t0
print(f"throughput: {n_tok} tokens in {dt:.1f}s = {n_tok / dt:,.0f} tok/s, peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
