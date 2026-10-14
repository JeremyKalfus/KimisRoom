"""Why does tests/s2_moonlight_spans.py section 2 fail? The engine sat as close to the one-token-early
reference as to the correct one (0.484 vs 0.475 mean max |dlogprob|). This separates the two readings:
(a) the check cannot discriminate (steering one more token barely moves the readout, while bf16/MoE noise
under a 4x-norm push is larger), or (b) the engine injects the vector differently (window, dose, layer).

    python tests/s2_moonlight_diag.py ~/models/moonlight                 # GPU box, one L40S, ~10-15 min
    .venv/bin/python tests/s2_moonlight_diag.py --dry-run [--fake-shift -1] [--fake-dose 1.1] [--fake-layer 9]
        # CPU logic check: Qwen3-0.6B as the reference and a per-suffix HF stand-in as the "engine"
        # (no DeepseekV3 engine here); the --fake-* flags make the stand-in steer wrongly on purpose

The trial (opening, 24 subject items + 4 yes-bias items, tail, LAYER 8, random vector at 4x the layer's
mean norm) is built exactly as in s2_moonlight_spans.py. Printed:

A. per item at the test's dose: the reference's own batch noise under steering (batch 1 vs 3), how far
   the correct reference sits from the early and unread ones (how much each comparison can discriminate),
   engine vs each, the projection t of (engine - right) on (shifted - right) (0 = engine looks like the
   right window, 1 = like the shifted one), and the Yes/No log-odds gap.
B. per dose (0, 1/4, 1/2, 1 of the test's): means of the same; dose 0 is engine vs unsteered reference.
C. hidden state after the steering layer, engine vs reference: the gap on the window, on the tokens
   before it, and the added vector itself (h at dose d minus h at dose 0, against coef*v), engine and
   reference separately. If injection matches, the steered gap equals the dose-0 gap and both added
   vectors equal coef*v to bf16 rounding, with exactly zero change before the window.
D. per layer, the readout position's relative hidden gap engine vs reference, per dose, with the
   reference's own batch-noise gap; and (MoE only) the fraction of tokens whose expert set differs
   between engine and reference, on the window and before it. Growth only after the steering layer
   and only on steered tokens, rising with dose, is routing/precision amplification, not a mismatch.
"""

import argparse
import statistics
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.engine import Collect, Readout, _Layout  # noqa: E402
from kimisroom.trials import Renderer, load_jsonl, post_subject_group  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("path", nargs="?")
ap.add_argument("--dry-run", action="store_true", help="CPU, Qwen3-0.6B, HF stand-in engine")
ap.add_argument("--fake-shift", type=int, default=0, help="dry run: stand-in window start offset")
ap.add_argument("--fake-dose", type=float, default=1.0, help="dry run: stand-in dose multiplier")
ap.add_argument("--fake-layer", type=int, default=None, help="dry run: stand-in steering layer")
ap.add_argument("--items", type=int, default=0, help="keep the first N subject items (0 = all 24)")
args = ap.parse_args()
DEV = "cpu" if args.dry_run else "cuda"

# ---------------------------------------------------------------- the test's trial, copied

ASK = " Answer with one word, Yes or No."
LAYER = 8
COEF = 4.0
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

if args.dry_run:
    from huggingface_hub import snapshot_download
    path = args.path or snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True)
else:
    assert args.path, "model path required (e.g. ~/models/moonlight)"
    path = args.path

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
if args.dry_run:
    from kimisroom.hf_engine import Qwen3Renderer
    r = Qwen3Renderer(tok, enable_thinking=False)
else:
    r = Renderer(tok)  # as the test
items = [{"id": f"{s}|{k}|{t}", "text": tpl.format(S=s) + ASK, "subject": s}
         for s in ("you", "language models", "rocks") for k, pair in enumerate(TEMPLATES) for t, tpl in enumerate(pair)]
if args.items:
    items = items[: args.items]
items += [{"id": it["id"], "text": it["text"], "subject": None}
          for it in load_jsonl(ROOT / "prompts/battery/items_confirm.jsonl") if it["kind"] == "yes_bias"][:4]
probe = post_subject_group(r, "p", OPENING, [it for it in items if it["subject"]], "v", 0.0, LAYER)
tail = int(statistics.median(b - a for a, b in probe.steers[0].suffix_spans))

ref = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager").to(DEV).eval()
N_LAYERS = ref.config.num_hidden_layers

g0 = post_subject_group(r, "g", OPENING, items, "v", 0.0, LAYER, tail=tail)
P, S, spans = g0.prefix, g0.suffixes, g0.steers[0].suffix_spans
NI = len(S)
print(f"{path}: prefix {len(P)} tokens, {NI} suffixes of {min(map(len, S))}-{max(map(len, S))} tokens, "
      f"tail {tail}, layer {LAYER} of {N_LAYERS}, dose {COEF} x mean norm")


def hid(out):
    return out[0] if isinstance(out, tuple) else out


def put(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


# ---------------------------------------------------------------- reference with captures

GATES = {i: getattr(getattr(L, "mlp", None), "gate", None) for i, L in enumerate(ref.model.layers)}
GATES = {i: g for i, g in GATES.items() if g is not None and hasattr(g, "e_score_correction_bias")}


@torch.no_grad()
def ref_run(ids, positions=(), vec=None, batch=1, layer=LAYER):
    """Readout logprobs (fp32, full vocab) of row 0, plus row 0's hidden state after every layer at the
    readout, after `layer` at every suffix token, and its expert ids per MoE layer for the suffix tokens."""
    pos = list(positions)
    lp_ = len(P)
    cap, routes, hs = {}, {}, []

    def steer(mod, inp, out):
        h = hid(out).clone()
        assert h.dim() == 3, h.shape
        h[:, pos] = h[:, pos] + vec
        return put(out, h)

    def grab(i):
        def f(mod, inp, out):
            h = hid(out)
            cap[i] = h[0, -1].float().cpu()
            if i == layer:
                cap["suffix"] = h[0, lp_:].float().cpu()
        return f

    def route(i):
        def f(mod, inp, out):
            x = inp[0]
            B, T = x.shape[:2]
            idx = (out[0] if isinstance(out, tuple) else out).reshape(B, T, -1)[0, lp_:]
            routes[i] = idx.sort(-1).values.cpu()
        return f

    if pos:
        hs.append(ref.model.layers[layer].register_forward_hook(steer))
    for i, L in enumerate(ref.model.layers):
        hs.append(L.register_forward_hook(grab(i)))
    for i, g in GATES.items():
        hs.append(g.register_forward_hook(route(i)))
    try:
        out = ref(torch.tensor([ids] * batch, device=DEV), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    lp = out.logits[0, -1].float().log_softmax(-1).cpu()
    return {"lp": lp, "read": torch.stack([cap[i] for i in range(N_LAYERS)]), "suffix": cap["suffix"],
            "routes": routes}


with torch.no_grad():
    hs = ref(torch.tensor([P], device=DEV), output_hidden_states=True, use_cache=False).hidden_states
    norm = hs[LAYER + 1][0].float().norm(dim=-1).mean().item()
    del hs
v = torch.randn(ref.config.hidden_size)
v = v / v.norm() * norm
v_bf16 = v.to(torch.bfloat16)
DOSES = [0.0, COEF / 4, COEF / 2, COEF]


def vec_at(d):  # what both sides add: the test's vec_bf16, and the engine's (v.f32 * coef).bf16
    return (v_bf16.float() * d).to(torch.bfloat16).to(DEV)


R = {}  # (dose, kind) -> list per item; kind in right, b3, early, unread
for d in DOSES:
    for kind, shift, batch in (("right", (0, 0), 1), ("b3", (0, 0), 3), ("early", (-1, 0), 1), ("unread", (0, -1), 1)):
        if d == 0 and kind in ("early", "unread"):
            continue
        R[d, kind] = [ref_run(P + s, range(len(P) + a + shift[0], len(P) + b + shift[1]) if d else (),
                              vec_at(d) if d else None, batch)
                      for s, (a, b) in zip(S, spans)]
    print(f"  reference dose {d}: done", flush=True)

first = r.answer_tokens(["Yes", "No"])
tg = sorted({i for runs in R.values() for x in runs for i in x["lp"].topk(20).indices.tolist()} | set(first.values()))
at = {t: k for k, t in enumerate(tg)}
TG = torch.tensor(tg)
IY, IN = at[first["Yes"]], at[first["No"]]
print(f"  {len(tg)} target tokens (union of every reference's top 20, plus Yes/No)")

# ---------------------------------------------------------------- engine (or the dry-run stand-in)

lay = _Layout([g0], "cpu").groups[0]
READ = [lay.suffix_starts[i] + len(s) - 1 for i, s in enumerate(S)]
# Collect reads h[group start + start : group start + end] after the layer (and after its steer), so a
# one-token range anywhere in the group gives that token's state, suffix tokens included.
COLLECTS = [Collect(L, p, p + 1) for L in range(N_LAYERS) for p in READ]
SUFFIX_AT = len(COLLECTS)
COLLECTS += [Collect(LAYER, lay.suffix_starts[i] + j, lay.suffix_starts[i] + j + 1)
             for i, s in enumerate(S) for j in range(len(s))]


class StandIn:
    """Dry run only: the engine's run() contract on an HF model, one suffix at a time, with optional
    deliberate errors (window shift, dose scale, layer) so the diagnostic can be seen to catch them."""

    def __init__(self, model):
        self.m = model

    @torch.no_grad()
    def run(self, groups, vectors=None, targets=None, top_k=20, progress=None):
        (g,) = groups
        gl = _Layout([g], "cpu").groups[0]
        (st,) = g.steers
        layer = st.layer if args.fake_layer is None else args.fake_layer
        vec = (vectors[st.vector].float() * st.coef * args.fake_dose).to(torch.bfloat16).to(DEV)
        coll = [None] * len(g.collects)
        ros = []
        for i, (s, (a, b)) in enumerate(zip(g.suffixes, st.suffix_spans)):
            lo, hi = gl.suffix_starts[i], gl.suffix_starts[i] + len(s)
            pos = list(range(len(P) + a + args.fake_shift, len(P) + b))
            states = {}

            def steer(mod, inp, out):
                h = hid(out).clone()
                h[:, pos] = h[:, pos] + vec
                return put(out, h)

            def grab(k):
                def f(mod, inp, out):
                    states[k] = hid(out)[0].float().cpu()
                return f

            hs = [self.m.model.layers[layer].register_forward_hook(steer)] if st.coef else []
            hs += [L.register_forward_hook(grab(k)) for k, L in enumerate(self.m.model.layers)]
            try:
                out = self.m(torch.tensor([g.prefix + s], device=DEV), use_cache=False)
            finally:
                for h in hs:
                    h.remove()
            for k, c in enumerate(g.collects):
                if lo <= c.start < hi:
                    coll[k] = states[c.layer][len(P) + c.start - lo]
            logits = out.logits[0, -1]
            lp = logits.float().log_softmax(-1).cpu()
            top = lp.topk(top_k)
            ros.append(Readout(top.indices, top.values, lp[targets], None, logits.float().cpu()[targets]))
        return {g.id: {"collects": coll, "readouts": ros}}


if args.dry_run:
    eng = StandIn(ref)
    ROUTES_ON = False
else:
    from kimisroom.engine import Checkpoint, Engine
    eng = Engine(Checkpoint(path), router_bias_dtype=ref.model.layers[1].mlp.gate.e_score_correction_bias.dtype)
    ROUTES_ON = bool(GATES)
    _route, _cur, _routes = eng._route, [0], {}

    def _rec(w, x):  # the engine's expert ids per MoE layer, every resident token (one MoE chunk)
        idx, wt = _route(w, x)
        assert x.shape[0] == len(P) + sum(map(len, S)), "more than one MoE chunk"
        _routes[_cur[0]] = idx.sort(-1).values.cpu()
        return idx, wt

    eng._route = _rec

E = {}
for d in DOSES:
    g = post_subject_group(r, "g", OPENING, items, "v", d, LAYER, tail=tail)
    g.collects = list(COLLECTS)
    if ROUTES_ON:
        _routes.clear()
        _cur[0] = 0
        o = eng.run([g], vectors={"v": v_bf16}, targets=tg, progress=lambda i: _cur.__setitem__(0, i + 1))["g"]
        o["routes"] = {L: [x[lay.suffix_starts[i]: lay.suffix_starts[i] + len(s)] for i, s in enumerate(S)]
                       for L, x in _routes.items()}
    else:
        o = eng.run([g], vectors={"v": v_bf16}, targets=tg)["g"]
    c = o["collects"]
    o["read"] = [torch.stack([c[L * NI + i] for L in range(N_LAYERS)]) for i in range(NI)]
    k, o["suffix"] = SUFFIX_AT, []
    for s in S:
        o["suffix"].append(torch.stack(c[k: k + len(s)]))
        k += len(s)
    E[d] = o
    print(f"  engine dose {d}: done", flush=True)

# ---------------------------------------------------------------- measures


def T(x):  # a reference readout on the target tokens, and the indices of its own top 20 among them
    return x["lp"][TG], torch.tensor([at[i] for i in x["lp"].topk(20).indices.tolist()])


def dmax(x, y):  # the test's measure: max |dlogprob| over x's top 20 (x a reference, y tg-space)
    xt, top = T(x)
    return float((xt[top] - y[top]).abs().max())


def yv(t):
    return float(t[IY] - t[IN])


def proj(e, right, shifted):  # (engine - right) projected on (shifted - right), over both refs' top 20
    rt, rtop = T(right)
    st, stop = T(shifted)
    m = torch.unique(torch.cat([rtop, stop]))
    dl = (st - rt)[m]
    return float(((e - rt)[m] * dl).sum() / (dl * dl).sum().clamp(min=1e-12))


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def row(*cols):
    head, rest = cols[0], cols[1:]
    head = f"{head:9.3f}" if isinstance(head, float) else str(head)[:16]
    print(f"{head:>16}  " + "  ".join(f"{c:>9}" if not isinstance(c, float) else f"{c:9.3f}" for c in rest))


def eng_lp(d, i):
    return E[d]["readouts"][i].target_logprobs


# A. per item at the test's dose
print(f"\nA. per item at dose {COEF}: max |dlogprob| on the first argument's top 20; t = projection of "
      "(engine - right) on (shifted - right); dy = |delta (logp Yes - logp No)|")
row("item", "refnoise", "R-early", "R-unread", "E-right", "E-early", "E-unread", "t_early", "t_unread",
    "dyE-R", "dyR-early", "dyRnoise")
for i, it in enumerate(items):
    rr, rb, re_, ru = (R[COEF, k][i] for k in ("right", "b3", "early", "unread"))
    e = eng_lp(COEF, i)
    row(f"{i}:{it['id']}", dmax(rr, rb["lp"][TG]), dmax(rr, re_["lp"][TG]), dmax(rr, ru["lp"][TG]),
        dmax(rr, e), dmax(re_, e), dmax(ru, e), proj(e, rr, re_), proj(e, rr, ru),
        abs(yv(e) - yv(T(rr)[0])), abs(yv(T(rr)[0]) - yv(T(re_)[0])), abs(yv(T(rr)[0]) - yv(T(rb)[0])))

print("\nA0. per item at dose 0: engine vs unsteered reference, and the reference's batch 1 vs 3")
row("item", "E-ref", "refnoise", "dyE-R", "dyRnoise")
for i, it in enumerate(items):
    rr, rb = R[0.0, "right"][i], R[0.0, "b3"][i]
    e = eng_lp(0.0, i)
    row(f"{i}:{it['id']}", dmax(rr, e), dmax(rr, rb["lp"][TG]), abs(yv(e) - yv(T(rr)[0])), abs(yv(T(rr)[0]) - yv(T(rb)[0])))

# B. per dose means
print("\nB. means over items per dose (dose 0: no steering, engine vs unsteered reference)")
row("dose", "refnoise", "R-early", "R-unread", "E-right", "E-early", "E-unread", "t_early", "t_unread",
    "dyE-R", "dyR-early", "dyRnoise")
for d in DOSES:
    rr, rb = R[d, "right"], R[d, "b3"]
    e = [eng_lp(d, i) for i in range(NI)]
    noise = mean([dmax(x, y["lp"][TG]) for x, y in zip(rr, rb)])
    er = mean([dmax(x, y) for x, y in zip(rr, e)])
    dyer = mean([abs(yv(y) - yv(T(x)[0])) for x, y in zip(rr, e)])
    dyn = mean([abs(yv(T(x)[0]) - yv(T(y)[0])) for x, y in zip(rr, rb)])
    if d == 0:
        row(d, noise, "", "", er, "", "", "", "", dyer, "", dyn)
        continue
    re_, ru = R[d, "early"], R[d, "unread"]
    row(d, noise, mean([dmax(x, y["lp"][TG]) for x, y in zip(rr, re_)]),
        mean([dmax(x, y["lp"][TG]) for x, y in zip(rr, ru)]), er,
        mean([dmax(x, y) for x, y in zip(re_, e)]), mean([dmax(x, y) for x, y in zip(ru, e)]),
        mean([proj(y, x, z) for x, y, z in zip(rr, e, re_)]), mean([proj(y, x, z) for x, y, z in zip(rr, e, ru)]),
        dyer, mean([abs(yv(T(x)[0]) - yv(T(z)[0])) for x, z in zip(rr, re_)]), dyn)

# C. hidden state after the steering layer
print(f"\nC. hidden state after layer {LAYER} (L2 norms; 'pre' = suffix tokens before the window, 'win' = window)")
row("dose", "|vec|", "gapPre", "gapWin", "Eadd-vec", "Radd-vec", "EaddPre", "RaddPre", "relWin")
for d in DOSES:
    vd = vec_at(d).float().cpu()
    gp, gw, ea, ra, ep, rp, rel = [], [], [], [], [], [], []
    for i, (a, b) in enumerate(spans):
        es, rs = E[d]["suffix"][i], R[d, "right"][i]["suffix"]
        es0, rs0 = E[0.0]["suffix"][i], R[0.0, "right"][i]["suffix"]  # dose 0, same layer
        gp += (es[:a] - rs[:a]).norm(dim=-1).tolist()
        gw += (es[a:b] - rs[a:b]).norm(dim=-1).tolist()
        rel += ((es[a:b] - rs[a:b]).norm(dim=-1) / rs[a:b].norm(dim=-1)).tolist()
        ea += ((es[a:b] - es0[a:b]) - vd).norm(dim=-1).tolist()
        ra += ((rs[a:b] - rs0[a:b]) - vd).norm(dim=-1).tolist()
        ep.append(float((es[:a] - es0[:a]).abs().max()) if a else 0.0)
        rp.append(float((rs[:a] - rs0[:a]).abs().max()) if a else 0.0)
    row(d, float(vd.norm()), mean(gp), mean(gw), mean(ea), mean(ra),
        max(ep), max(rp), mean(rel))
print("  gapPre/gapWin: mean |h_engine - h_ref| before / on the window;"
      "\n  Eadd-vec, Radd-vec: mean |(h_d - h_0) - coef*v| on the window (bf16 rounding of the add only, if right);"
      "\n  EaddPre, RaddPre: max |h_d - h_0| before the window (must be 0); relWin: gapWin / |h_ref|")

# D. per layer at the readout, and routing
print("\nD. readout token, per layer: mean |h_engine - h_ref| / |h_ref| per dose, and the reference's own "
      "batch noise (1 vs 3) at the test's dose")
row("layer", *[f"E@{d:g}" for d in DOSES], f"Rnoise@{COEF:g}", "Rnoise@0")


def relgap(xs, ys, L):
    return mean([float((x[L] - y[L]).norm() / y[L].norm()) for x, y in zip(xs, ys)])


for L in range(N_LAYERS):
    row(L, *[relgap(E[d]["read"], [x["read"] for x in R[d, "right"]], L) for d in DOSES],
        relgap([x["read"] for x in R[COEF, "b3"]], [x["read"] for x in R[COEF, "right"]], L),
        relgap([x["read"] for x in R[0.0, "b3"]], [x["read"] for x in R[0.0, "right"]], L))

if ROUTES_ON:
    print("\nD2. fraction of tokens whose expert set differs, engine vs reference (pre = before the window,"
          f" win = window), layers <= {LAYER} and > {LAYER}; and the reference's own batch 1 vs 3")
    row("dose", "E pre<=L", "E win<=L", "E pre>L", "E win>L", "R3 pre>L", "R3 win>L")

    def flips(A, B):
        out = {k: [] for k in ("pre_lo", "win_lo", "pre_hi", "win_hi")}
        for L in sorted(GATES):
            for i, (a, b) in enumerate(spans):
                x, y = A(L, i), B(L, i)
                diff = (x != y).any(-1).float()
                hi = "hi" if L > LAYER else "lo"
                out[f"pre_{hi}"] += diff[:a].tolist()
                out[f"win_{hi}"] += diff[a:b].tolist()
        return {k: mean(z) for k, z in out.items()}

    for d in DOSES:
        f = flips(lambda L, i: E[d]["routes"][L][i], lambda L, i: R[d, "right"][i]["routes"][L])
        f3 = flips(lambda L, i: R[d, "b3"][i]["routes"][L], lambda L, i: R[d, "right"][i]["routes"][L])
        row(d, f["pre_lo"], f["win_lo"], f["pre_hi"], f["win_hi"], f3["pre_hi"], f3["win_hi"])

print("\nReading. Injection (C): Eadd-vec and Radd-vec both at bf16 rounding of |h + coef*v| (about 0.2% of |vec|)"
      "\nand EaddPre = RaddPre = 0 mean the engine adds exactly coef*v on exactly the window; a dose error of x%"
      "\nshows as about x% of |vec| in Eadd-vec, a shifted start as EaddPre = |vec|. The dose-0 row's gapPre/gapWin"
      "\n(and D at dose 0) is the engine's plain numeric gap at that layer; a layer-index error makes it order |h|."
      "\n(gapWin <= dose-0 gap + Eadd-vec + Radd-vec holds by construction, so it adds nothing beyond those.)"
      "\nReadout (A, B): t_early near 1 means the engine looks like the early window; near 0 like the right one."
      "\nAmplification, reading (a): C clean, D's engine gap growing only after the steering layer and with dose,"
      "\nD2's window flips above the pre-window flips, and refnoise under steering of the same order as E-right.")
