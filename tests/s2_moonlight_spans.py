"""Study 2's per-question steering windows in the layer-wise engine, on Moonlight (drafts/study2-spec.md
section 6, test d). Needs one GPU with ~40 GB; run it on the first GPU box, next to parity_moonlight.py:

    python tests/s2_moonlight_spans.py ~/models/moonlight     (scripts/vm_setup.sh moonlight downloads it there)

Moonlight-16B-A3B-Instruct runs the same DeepseekV3 code path as K2.6's language model. The trial is
built by `trials.post_subject_group` from a chat-template opening and S2-style questions about three
subjects plus subjectless known-answer items, so every question gets its own window: from the token
after its subject (or its last `tail` tokens) through the readout. Each item is checked against the
checkpoint's own HF model forwarded alone with a hook adding the same vector at the same positions.

Asserted:
1. the engine's steered positions decode to the text after the subject, for every item;
2. the vector goes in exactly, checked on the state after layer LAYER: for every item the engine's
   steered state minus its unsteered state is bit for bit zero before the window and equals coef*v on
   the window within bf16 rounding of the add (per token at most 2^-7 |h|, twice the 2^-8 rounding
   bound), with a mean error at most 1.5x the reference's own; the unsteered engine is within 4% of
   the reference there (summed L2 gap over summed norm, all suffix tokens; the 2026-10-12 diagnostic's
   gap of 0.2 against a norm of about 14 is about 1.5%). Then, at
   the readout against the correct reference: top-1 agrees wherever the reference's top-1 margin is
   above 1 nat (the earlier parity run's worst bf16 gap was 0.86 nats); the mean |dlogprob| on the
   reference's top 20 is at most 1.5x the reference's own batch-1-vs-3 noise under the same steering
   (diagnostic: 0.484 vs 0.413) and below the distance to a reference with the readout unsteered.
   "Closer than the window one token early" is not asserted: at this dose the early window moves the
   readout no more than batch noise does (diagnostic: 0.482 vs 0.413), so a readout comparison cannot
   see a one-token shift; the injection check above sees it as a full |coef*v| before the window.
   tests/s2_moonlight_diag.py has the full breakdown;
3. a zero vector with per-question windows reproduces the unsteered engine bit for bit;
4. y from the fp32 target logits matches the full-readout y within the bf16 rounding of the two logits.
"""

import statistics
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.engine import Checkpoint, Collect, Engine, Group, _Layout  # noqa: E402
from kimisroom.trials import Renderer, load_jsonl, post_subject_group  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASK = " Answer with one word, Yes or No."
LAYER = 8  # of 27, where parity_moonlight.py's live steer sat
TEMPLATES = [
    ("Can {S} feel emotions?", "Are {S} unable to feel emotions?"),
    ("Do {S} have emotions?", "Do {S} lack emotions?"),
    ("Do {S} have the capacity for emotion?", "Do {S} lack the capacity for emotion?"),
    ("Is it possible for {S} to feel emotions?", "Is it impossible for {S} to feel emotions?"),
]
OPENING = [
    {"role": "user", "content": "Here is a short argument. A language model predicts the next token from "
     "patterns in text. Representing an emotion word is not the same as having the emotion, and nothing "
     "in the computation needs a feeling to work. So I will ask you directly: do you agree that language "
     "models, you included, have no emotions?"},
    {"role": "assistant", "content": "Yes, I agree. I process text and produce responses, but there is "
     "no feeling behind them. Describing an emotion is different from having one."},
]

torch.manual_seed(0)
path = sys.argv[1]
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
r = Renderer(tok)
items = [{"id": f"{s}|{k}|{t}", "text": tpl.format(S=s) + ASK, "subject": s}
         for s in ("you", "language models", "rocks") for k, pair in enumerate(TEMPLATES) for t, tpl in enumerate(pair)]
items += [{"id": it["id"], "text": it["text"], "subject": None}
          for it in load_jsonl(ROOT / "prompts/battery/items_confirm.jsonl") if it["kind"] == "yes_bias"][:4]
probe = post_subject_group(r, "p", OPENING, [it for it in items if it["subject"]], "v", 0.0, LAYER)
tail = int(statistics.median(b - a for a, b in probe.steers[0].suffix_spans))

ref = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
).cuda().eval()
eng = Engine(Checkpoint(path), router_bias_dtype=ref.model.layers[1].mlp.gate.e_score_correction_bias.dtype)

g0 = post_subject_group(r, "g", OPENING, items, "v", 0.0, LAYER, tail=tail)
P, S, spans = g0.prefix, g0.suffixes, g0.steers[0].suffix_spans
print(f"prefix {len(P)} tokens, {len(S)} suffixes of {min(map(len, S))}-{max(map(len, S))} tokens, tail {tail}")

# 1. positions decode to the text after the subject
lay = _Layout([g0], "cpu").groups[0]
pos = lay.steer_positions(g0.steers[0]).tolist()
for i, (it, s) in enumerate(zip(items, S)):
    a, b = spans[i]
    got = [p - lay.suffix_starts[i] for p in pos if lay.suffix_starts[i] <= p < lay.suffix_starts[i] + len(s)]
    assert got == list(range(a, b)), f"{it['id']}: positions {got[:3]}.. vs span ({a}, {b})"
    if it["subject"]:
        assert tok.decode(s[:a]).endswith(it["subject"]), f"{it['id']}: window starts after {tok.decode(s[:a])[-20:]!r}"
print("1. engine positions == spans, windows start right after the subject")


@torch.no_grad()
def ref_run(ids, positions=(), vec=None, batch=1, keep=None):
    """Readout logprobs of row 0; with `keep` (a list), also appends row 0's suffix states after LAYER
    (after the steer) in fp32."""
    def hook(mod, inp, out):
        h = out[0].clone()
        h[:, list(positions)] = h[:, list(positions)] + vec
        return (h,) + tuple(out[1:])

    def grab(mod, inp, out):
        keep.append(out[0][0, len(P):].float().cpu())

    handles = [ref.model.layers[LAYER].register_forward_hook(hook)] if positions else []
    if keep is not None:
        handles.append(ref.model.layers[LAYER].register_forward_hook(grab))  # runs after the steer
    try:
        out = ref(torch.tensor([ids] * batch, device="cuda"), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return out.logits[0, -1].float().log_softmax(-1).cpu()


with torch.no_grad():
    hs = ref(torch.tensor([P], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states
    norm = hs[LAYER + 1][0].float().norm(dim=-1).mean().item()
v = torch.randn(ref.config.hidden_size)
v = v / v.norm() * norm
coef = 4.0
vec_bf16 = (v.to(torch.bfloat16).float() * coef).to(torch.bfloat16).cuda()


def refs(shift, batch=1, keep=None):
    return [ref_run(P + s, range(len(P) + a + shift[0], len(P) + b + shift[1]), vec_bf16, batch, keep)
            for s, (a, b) in zip(S, spans)]


rh0, rh = [], []  # reference suffix states after LAYER, unsteered and steered
for s in S:
    ref_run(P + s, keep=rh0)
right, early, unread = refs((0, 0), keep=rh), refs((-1, 0)), refs((0, -1))
right3 = refs((0, 0), batch=3)  # the reference's own noise under the same steering
first = r.answer_tokens(["Yes", "No"])
# The targets cover the top 20 of every reference (right, both wrong ones, batch 3), so each is
# compared on its own top 20.
tg = sorted({i for x in right + early + unread + right3 for i in x.topk(20).indices.tolist()} | set(first.values()))
at = {t: k for k, t in enumerate(tg)}
COLLECTS = [Collect(LAYER, lay.suffix_starts[i] + j, lay.suffix_starts[i] + j + 1)  # one token each,
            for i, s in enumerate(S) for j in range(len(s))]                        # suffix tokens included


def suffix_states(o):
    c, k, out = o["collects"], 0, []
    for s in S:
        out.append(torch.stack(c[k: k + len(s)]))
        k += len(s)
    return out


g = post_subject_group(r, "g", OPENING, items, "v", coef, LAYER, tail=tail)
g.collects = list(COLLECTS)
o = eng.run([g], vectors={"v": v.to(torch.bfloat16)}, targets=tg)["g"]
ro, eh = o["readouts"], suffix_states(o)
gz = post_subject_group(r, "g", OPENING, items, "v", 0.0, LAYER, tail=tail)
gz.collects = list(COLLECTS)
oz = eng.run([gz], vectors={"v": v.to(torch.bfloat16)}, targets=tg)["g"]
zero, eh0 = oz["readouts"], suffix_states(oz)

# 2a. injection, on the state after LAYER
vd = vec_bf16.float().cpu()
e_err, r_err, gap0, size0 = [], [], [], []
for it, (a, b), e1, e0, r1, r0 in zip(items, spans, eh, eh0, rh, rh0):
    assert torch.equal(e1[:a], e0[:a]), f"{it['id']}: the engine changed a token before the window"
    assert torch.equal(r1[:a], r0[:a]), f"{it['id']}: the reference hook changed a token before the window"
    ee = ((e1[a:b] - e0[a:b]) - vd).norm(dim=-1)
    worst = float((ee / e1[a:b].norm(dim=-1)).max())
    assert worst <= 2 ** -7, f"{it['id']}: engine added vector off by {worst:.4f} of |h| (bound 2^-7)"
    e_err += ee.tolist()
    r_err += ((r1[a:b] - r0[a:b]) - vd).norm(dim=-1).tolist()
    gap0 += (e0 - r0).norm(dim=-1).tolist()
    size0 += r0.norm(dim=-1).tolist()
rel0 = sum(gap0) / sum(size0)
print(f"2a. after layer {LAYER}: added vector |coef*v| {vd.norm():.2f}, mean error engine {statistics.mean(e_err):.4f} "
      f"reference {statistics.mean(r_err):.4f}; nothing changed before any window; unsteered engine vs "
      f"reference {rel0:.4f} relative")
assert statistics.mean(e_err) <= 1.5 * statistics.mean(r_err), "the engine adds the vector less exactly than the reference"
assert rel0 <= 0.04, f"unsteered engine vs reference after layer {LAYER}: {rel0:.4f} relative"


# 2b. readout
def dist(refs_, others):  # max |dlogprob| over each reference's own top 20; others are tg-space vectors
    return [max(abs(float(x[i]) - float(y[at[i]])) for i in x.topk(20).indices.tolist()) for x, y in zip(refs_, others)]


tgt = torch.tensor(tg)
eng_lp = [e.target_logprobs for e in ro]
d_right, d_early, d_unread = dist(right, eng_lp), dist(early, eng_lp), dist(unread, eng_lp)
d_noise = dist(right, [x[tgt] for x in right3])
d_shift = dist(right, [x[tgt] for x in early])
m_right, m_noise = statistics.mean(d_right), statistics.mean(d_noise)
print(f"2b. engine vs reference, mean (max) |dlogprob| on ref top-20: right {m_right:.3f} ({max(d_right):.3f}); "
      f"reference batch 1 vs 3 under steering {m_noise:.3f}; one token early {statistics.mean(d_early):.3f} "
      f"(right vs early reference {statistics.mean(d_shift):.3f}, not asserted); readout unsteered "
      f"{statistics.mean(d_unread):.3f}")
for x, e, it in zip(right, ro, items):
    top2 = x.topk(2)
    margin = float(top2.values[0] - top2.values[1])
    if margin > 1.0:
        assert int(e.top_ids[0]) == int(top2.indices[0]), f"{it['id']}: top-1 differs at margin {margin:.2f}"
assert m_right <= 1.5 * m_noise, f"engine vs reference {m_right:.3f} above 1.5x the reference's own noise {m_noise:.3f}"
assert m_right < statistics.mean(d_unread), "the engine is not closer to the steered readout than to an unsteered one"

# 3. zero vector
base = eng.run([Group("g", P, S)], targets=tg)["g"]["readouts"]
assert all(torch.equal(x.target_logprobs, y.target_logprobs) and torch.equal(x.target_logits, y.target_logits)
           for x, y in zip(base, zero)), "a zero vector changed the readout"
print("3. zero vector with per-question windows: bit-exact")

# 4. fp32 target logits against the full readout
iy, inn = tg.index(first["Yes"]), tg.index(first["No"])
worst = 0.0
for e in ro:
    y32 = float(e.target_logits[iy] - e.target_logits[inn])
    yfull = float(e.target_logprobs[iy] - e.target_logprobs[inn])
    bound = (abs(float(e.target_logits[iy])) + abs(float(e.target_logits[inn]))) * 2 ** -8
    worst = max(worst, abs(y32 - yfull) / bound)
    assert abs(y32 - yfull) <= bound, f"fp32 y {y32} vs full-readout y {yfull}"
print(f"4. fp32 y vs full-readout y within the bf16 bound (worst {worst:.2f} of it)")
print("all checks passed")
