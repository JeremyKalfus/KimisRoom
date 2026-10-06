"""CPU check of the GLM-5.3 port (kimisroom/glm.py) on a tiny random FP8 checkpoint, against
transformers' own GlmMoeDsaForCausalLM. Needs transformers >= 5.6 (`glm_moe_dsa` with
`indexer_types`); no GPU and no download:

    python tests/glm_tiny_check.py [--keep DIR]

The checkpoint has GLM-5.3's layout at toy sizes: e4m3 linears with 128x128 block scales
(kv_a_proj_with_mqa ends in a partial block, as in the real one), a dense first layer, MoE layers
with a sigmoid router, correction bias and a shared expert, interleaved RoPE, and DeepSeek Sparse
Attention with index_topk 16, so 20-60 token sequences run the indexer on "full" and "shared"
layers, in a prefix and in suffixes. Its two shards split a layer, as the real shards do.

Asserted:
1. dequant: glm.dequant_fp8 == glm_ref.dequant (explicit block loop) on every quantized matrix,
   == transformers' Fp8Dequantize on every whole-block one, and the expert path (_dequant_blocks,
   compiled or not) == dequant_fp8.
2. selection: the engine's indexer picks exactly the keys transformers' GlmMoeDsaIndexer picks,
   on every row of a 60-token input, on every full layer.
3. harness: tests' layer-by-layer reference (glm_ref.LayerwiseRef) == GlmMoeDsaForCausalLM.
4. whole model: GlmEngine.run (prefix_all readouts, suffixes, steering, collects) against the
   reference in bf16 and fp32: same top-1 everywhere, and the engine's max |dlogprob| to fp32
   at most 2x transformers-bf16's (the bf16 noise floor) plus 0.01. A dense reference (DSA off) is
   the sensitivity control: the engine must sit far closer to the sparse reference than to it.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kimisroom import glm, glm_ref  # noqa: E402
from kimisroom.engine import Collect, Group, Steer, _Layout, linear, rms_norm  # noqa: E402

F32, BF16, FP8 = torch.float32, torch.bfloat16, torch.float8_e4m3fn
TOPK = 16
CFG = {
    "architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa", "attention_bias": False,
    "attention_dropout": 0.0, "dtype": "bfloat16", "eos_token_id": [1], "pad_token_id": 0, "ep_size": 1,
    "first_k_dense_replace": 1, "head_dim": 128, "hidden_act": "silu", "hidden_size": 256,
    "index_head_dim": 128, "index_n_heads": 4, "index_share_for_mtp_iteration": True,
    "index_skip_topk_offset": 2, "index_topk": TOPK, "index_topk_freq": 2, "index_topk_pattern": None,
    "indexer_rope_interleave": True, "indexer_types": ["full", "full", "shared", "full", "shared", "shared"],
    "initializer_range": 0.02, "intermediate_size": 384, "kv_lora_rank": 128, "max_position_embeddings": 4096,
    "mlp_layer_types": ["dense"] + ["sparse"] * 5, "moe_intermediate_size": 128, "moe_layer_freq": 1,
    "moe_router_dtype": "float32", "n_group": 1, "n_routed_experts": 16, "n_shared_experts": 1,
    "norm_topk_prob": True, "num_attention_heads": 4, "num_experts_per_tok": 4, "num_hidden_layers": 6,
    "num_key_value_heads": 4, "num_nextn_predict_layers": 0, "pretraining_tp": 1, "q_lora_rank": 128,
    "qk_head_dim": 128, "qk_nope_head_dim": 64, "qk_rope_head_dim": 64, "rms_norm_eps": 1e-05,
    "rope_interleave": True, "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
    "routed_scaling_factor": 2.5, "scoring_func": "sigmoid", "tie_word_embeddings": False, "topk_group": 1,
    "topk_method": "noaux_tc", "use_cache": False, "v_head_dim": 64, "vocab_size": 512,
    "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                            "weight_block_size": [128, 128]},
}


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 (R, C) -> e4m3 values and one fp32 scale per 128x128 block (partial edge blocks kept)."""
    R, C = w.shape
    nr, nc = -(-R // 128), -(-C // 128)
    s = torch.empty(nr, nc)
    q = torch.empty(R, C, dtype=FP8)
    for i in range(nr):
        for j in range(nc):
            blk = w[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128]
            s[i, j] = blk.abs().amax().clamp(min=1e-12) / 448.0
            q[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128] = (blk / s[i, j]).to(FP8)
    return q, s


def write_checkpoint(d: Path) -> None:
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(0)
    c = CFG
    H, nh, dn, dr, dv = c["hidden_size"], c["num_attention_heads"], c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]

    def rnd(*shape, gain=1.0):
        return torch.randn(*shape, generator=g) * gain / shape[-1] ** 0.5

    t: dict[str, torch.Tensor] = {}

    def lin(name, rows, cols, gain=1.0):
        t[name + ".weight"], t[name + ".weight_scale_inv"] = quantize(rnd(rows, cols, gain=gain))

    def norm(name, n, bias=False):
        t[name + ".weight"] = (1 + 0.1 * torch.randn(n, generator=g)).to(BF16)
        if bias:
            t[name + ".bias"] = (0.1 * torch.randn(n, generator=g)).to(BF16)

    t["model.embed_tokens.weight"] = torch.randn(c["vocab_size"], H, generator=g).to(BF16)
    t["lm_head.weight"] = (rnd(c["vocab_size"], H, gain=2.0)).to(BF16)
    norm("model.norm", H)
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        norm(p + "input_layernorm", H)
        norm(p + "post_attention_layernorm", H)
        lin(p + "self_attn.q_a_proj", c["q_lora_rank"], H)
        norm(p + "self_attn.q_a_layernorm", c["q_lora_rank"])
        lin(p + "self_attn.q_b_proj", nh * (dn + dr), c["q_lora_rank"])
        lin(p + "self_attn.kv_a_proj_with_mqa", c["kv_lora_rank"] + dr, H)  # 192 rows: a partial block
        norm(p + "self_attn.kv_a_layernorm", c["kv_lora_rank"])
        lin(p + "self_attn.kv_b_proj", nh * (dn + dv), c["kv_lora_rank"])
        lin(p + "self_attn.o_proj", H, nh * dv)
        if c["indexer_types"][i] == "full":
            lin(p + "self_attn.indexer.wq_b", c["index_n_heads"] * c["index_head_dim"], c["q_lora_rank"])
            lin(p + "self_attn.indexer.wk", c["index_head_dim"], H)
            norm(p + "self_attn.indexer.k_norm", c["index_head_dim"], bias=True)
            t[p + "self_attn.indexer.weights_proj.weight"] = rnd(c["index_n_heads"], H, gain=1.0).to(BF16)
        if c["mlp_layer_types"][i] == "dense":
            for m, (r, k) in {"gate_proj": (384, H), "up_proj": (384, H), "down_proj": (H, 384)}.items():
                lin(p + f"mlp.{m}", r, k)
        else:
            I = c["moe_intermediate_size"]
            t[p + "mlp.gate.weight"] = rnd(c["n_routed_experts"], H, gain=1.0).to(BF16)
            t[p + "mlp.gate.e_score_correction_bias"] = 0.1 * torch.randn(c["n_routed_experts"], generator=g)
            for m, (r, k) in {"gate_proj": (I, H), "up_proj": (I, H), "down_proj": (H, I)}.items():
                lin(p + f"mlp.shared_experts.{m}", r, k)
                for e in range(c["n_routed_experts"]):
                    lin(p + f"mlp.experts.{e}.{m}", r, k)
    # two shards that split layer 3 down the middle of its experts, as the real shards split layers
    names = list(t)
    cut = names.index("model.layers.3.mlp.experts.7.up_proj.weight")
    shards = {"model-00001-of-00002.safetensors": names[:cut], "model-00002-of-00002.safetensors": names[cut:]}
    wm = {}
    for f, ks in shards.items():
        save_file({k: t[k].contiguous() for k in ks}, str(d / f))
        wm.update({k: f for k in ks})
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (d / "config.json").write_text(json.dumps(CFG))


def check_dequant(ck: glm.GlmCheckpoint) -> None:
    try:
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize
        tf_deq = Fp8Dequantize(None)._dequantize_one
    except Exception as e:  # noqa: BLE001
        print(f"  transformers' Fp8Dequantize unavailable ({type(e).__name__}: {e}); skipping that comparison")
        tf_deq = None
    n = n_tf = 0
    for i in (0, 1):
        got = ck.read_layer(i)
        for k in [k for k in got if k.endswith(".weight_scale_inv")]:
            w, s = got[k[: -len("_scale_inv")]], got[k]
            mine = glm.dequant_fp8(w, s)
            assert torch.equal(mine, glm_ref.dequant(w, s, BF16, "cpu")), k
            assert torch.equal(glm.dequant_fp8(w, s, F32), glm_ref.dequant(w, s, F32, "cpu")), k
            if tf_deq is not None and w.shape[0] % 128 == 0 and w.shape[1] % 128 == 0:
                assert torch.equal(mine, tf_deq(w.view(FP8), s, BF16)), k
                n_tf += 1
            if ".experts." in k:
                assert torch.equal(glm._dequant_blocks(w, s), mine), k
                assert torch.equal(glm.dequant_expert(w, s), mine), k
                assert torch.equal(glm.dequant_expert(w, s), mine), k  # second call: the cached (maybe compiled) path
            n += 1
    print(f"1. dequant: {n} matrices equal to the block-loop reference, {n_tf} to transformers' Fp8Dequantize, "
          f"expert path {'compiled' if glm._dequant_fn and glm._compiled in glm._dequant_fn.values() else 'eager'}: ok")


def compare_selection(have, want, scores, k):
    """have/want (R, K) bool selections; scores (R, K) the engine's index scores. Returns (rows that
    differ beyond ties, rows that differ only among keys tied with the k-th score). transformers
    takes torch.topk's order among tied keys, which on the CPU depends on the row's length; the
    engine takes the earlier key."""
    bad, ties = [], 0
    for r in (have != want).any(-1).nonzero().flatten().tolist():
        kth = scores[r].sort(descending=True).values[k - 1]
        d = (have[r] != want[r]).nonzero().flatten()
        if int(have[r].sum()) == int(want[r].sum()) == k and bool((scores[r, d] == kth).all()):
            ties += 1
        else:
            bad.append(r)
    return bad, ties


def check_selection(ck: glm.GlmCheckpoint, cfg) -> None:
    eng = glm.GlmEngine(ck, device="cpu")
    g = torch.Generator().manual_seed(1)
    T = 60
    x = (torch.randn(T, CFG["hidden_size"], generator=g)).to(BF16)
    pos = torch.arange(T)
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    rot = glm_ref.rotary(cfg, "cpu")
    n_rows = n_ties = 0
    for i in [i for i, t in enumerate(CFG["indexer_types"]) if t == "full"]:
        got = ck.read_layer(i)
        layer = glm_ref.build_layer(cfg, ck, got, i, BF16, "cpu")
        w = eng._to_device(ck.pack_layer(got, i))
        cos, sin = rot(x[None], pos[None])
        attn = layer.self_attn
        q_resid = attn.q_a_layernorm(attn.q_a_proj(x[None]))
        ref = attn.indexer(x[None], q_resid, (cos, sin), causal[None], pos[None])[0]  # (T, topk)
        ref_sel = torch.zeros(T, T, dtype=torch.bool).scatter(-1, ref.long(), True) & causal
        qa = rms_norm(linear(x, w["self_attn.q_a_proj.weight"], glm.TILE_ATTN), w["self_attn.q_a_layernorm.weight"], 1e-6)
        assert torch.equal(qa, q_resid[0]), f"layer {i}: q_resid differs"
        c2, s2 = glm.rope_cos_sin(eng.inv_freq, pos)
        assert torch.equal(c2, cos[0]) and torch.equal(s2, sin[0]), f"layer {i}: rotary differs"
        iq, ik, iw = eng._index_qkw(w, x, qa, c2, s2)
        sel = eng._select(iq, iw, ik[None], causal)
        bad, ties = compare_selection(sel[TOPK:], ref_sel[TOPK:], eng._index_scores(iq, iw, ik[None], causal)[TOPK:], TOPK)
        assert not bad, f"layer {i}: prefix selection differs on rows {[r + TOPK for r in bad]}"
        assert bool((sel[:TOPK] == causal[:TOPK]).all()) and bool((ref_sel[:TOPK] == causal[:TOPK]).all())
        n_rows, n_ties = n_rows + T - TOPK, n_ties + ties

        # the suffix path: what _attention stores per suffix chunk == the indexer on prefix + that suffix
        lens = [9, 20, 3, 31]
        grp = Group("s", list(range(2, 2 + 12)), [list(range(2, 2 + L)) for L in lens])
        gl = _Layout([grp], torch.device("cpu")).groups[0]
        xs = torch.randn(grp.n_tokens, CFG["hidden_size"], generator=g).to(BF16)
        eng._dsa = {}
        w["layer"] = i
        eng._attention(w, xs, gl)
        qa_s = rms_norm(linear(xs, w["self_attn.q_a_proj.weight"], glm.TILE_ATTN), w["self_attn.q_a_layernorm.weight"], 1e-6)
        c3, s3 = glm.rope_cos_sin(eng.inv_freq, gl.positions)
        jq, jk, jw = eng._index_qkw(w, xs, qa_s, c3, s3)
        Lp = 12
        for n_chunk, (idx, valid, _) in enumerate(gl.suffix_chunks(eng.suffix_chunk)):
            sel = eng._dsa.get((gl.start, n_chunk))
            assert sel is not None, f"layer {i}: chunk {n_chunk} skipped DSA"
            for row in range(idx.shape[0]):
                L = int(valid[row].sum())
                if L == 0:
                    continue
                own = idx[row, :L]
                seq = torch.cat([xs[:Lp], xs[own]])
                S = Lp + L
                p = torch.arange(S)
                cs, sn = rot(seq[None], p[None])
                qr = attn.q_a_layernorm(attn.q_a_proj(seq[None]))
                m = torch.ones(S, S, dtype=torch.bool).tril()
                tk = attn.indexer(seq[None], qr, (cs, sn), m[None], p[None])[0]
                want = (torch.zeros(S, S, dtype=torch.bool).scatter(-1, tk.long(), True) & m)[Lp:]  # (L, S)
                have = torch.cat([sel[row, :L, :Lp], sel[row, :L, Lp : Lp + L]], dim=-1)
                long_rows = (m[Lp:].sum(-1) > TOPK).nonzero().flatten()
                assert bool((have[m[Lp:].sum(-1) <= TOPK] == m[Lp:][m[Lp:].sum(-1) <= TOPK]).all())
                scores = eng._index_scores(jq[own], jw[own], torch.cat([jk[:Lp], jk[own]])[None], m[Lp:])
                bad, ties = compare_selection(have[long_rows], want[long_rows], scores[long_rows], TOPK)
                assert not bad, f"layer {i}: suffix chunk {n_chunk} row {row} differs at offsets {bad}"
                n_rows, n_ties = n_rows + len(long_rows), n_ties + ties
    print(f"2. selection: engine indexer == GlmMoeDsaIndexer on {n_rows} rows past topk (prefix and suffix "
          f"chunks, every full layer); {n_ties} differ only in the order of keys tied at the k-th score: ok")


def run_reference(ck, cfg, dtype, seqs, steer_plan, collect_plan):
    """LayerwiseRef over seqs; steer_plan[layer] = [(seq, positions, vec, coef)];
    collect_plan[layer] = [(key, seq, start, end)]. Returns (logprobs per seq, collects)."""
    ref = glm_ref.LayerwiseRef(cfg, ck, dtype, "cpu")
    ref.embed(seqs, ck.load_global("model.embed_tokens.weight"))
    collects = {}
    for i in range(cfg.num_hidden_layers):
        ref.step(i, ck.read_layer(i))
        for s, p, v, coef in steer_plan.get(i, []):
            ref.steer(s, p, v, coef)
        for key, s, a, b in collect_plan.get(i, []):
            collects[key] = ref.hs[s][0, a:b].to(F32).mean(0)
    norm_w, head_w = ck.load_global("model.norm.weight"), ck.load_global("lm_head.weight")
    lps = [glm_ref.final_logprobs(h[0], norm_w, head_w, cfg.rms_norm_eps, dtype) for h in ref.hs]
    return lps, collects


def isolated_layers(ck, cfg, cfg_dense, groups, seqs, rows, steer_plan):
    """Per layer: the fp32 reference's chained input fed to the engine (as bf16, through the
    real multi-group layout), to transformers in bf16 and to a dense fp32 reference. Returns
    (layer, engine, bf16, dense) errors relative to the fp32 reference's own update."""
    eng = glm.GlmEngine(ck, device="cpu")
    layout = _Layout(groups, torch.device("cpu"))
    emb = ck.load_global("model.embed_tokens.weight")
    r32, r16, rd = (glm_ref.LayerwiseRef(c, ck, dt, "cpu") for c, dt in ((cfg, F32), (cfg, BF16), (cfg_dense, F32)))
    for r in (r32, r16, rd):
        r.embed(seqs, emb)

    def gather(hs):
        return torch.stack([hs[s][0, p].to(F32) for s, p in rows])

    res = []
    for i in range(cfg.num_hidden_layers):
        got = ck.read_layer(i)
        ins = [h.clone() for h in r32.hs]
        x = gather(ins)
        o32 = gather(r32.step(i, got))
        o16 = gather(r16.step(i, got, inputs=[h.to(BF16) for h in ins]))
        od = gather(rd.step(i, got, inputs=ins))
        h = x.to(BF16)
        eng.layer(eng._to_device(ck.pack_layer(got, i)), h, layout)
        upd = (o32 - x).norm()
        res.append((i, ((h.to(F32) - o32).norm() / upd).item(), ((o16 - o32).norm() / upd).item(),
                    ((od - o32).norm() / upd).item()))
        for s, p, v, coef in steer_plan.get(i, []):
            r32.steer(s, p, v, coef)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", help="write the tiny checkpoint here and keep it")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.cuda.synchronize = lambda *args, **kw: None  # Engine.run syncs its device; this run is on the CPU
    tmp = tempfile.TemporaryDirectory()
    d = Path(a.keep or tmp.name)
    d.mkdir(parents=True, exist_ok=True)
    write_checkpoint(d)
    ck = glm.GlmCheckpoint(d)
    assert ck._file_layers["model-00001-of-00002.safetensors"] >= {3} and 3 in ck._file_layers["model-00002-of-00002.safetensors"]
    cfg = glm_ref.hf_config(d)
    check_dequant(ck)
    check_selection(ck, cfg)

    # ---------------- groups: a 60-token prefix read everywhere, suffixes crossing topk, steering
    g = torch.Generator().manual_seed(2)

    def toks(n):
        return torch.randint(2, CFG["vocab_size"], (n,), generator=g).tolist()

    vec = torch.randn(CFG["hidden_size"], generator=g) * 2.0
    gA = Group("A", toks(60), readout="prefix_all")
    gB = Group("B", toks(12), [toks(3), toks(9), toks(20)])
    gC = Group("C", toks(30), [toks(5), toks(33), toks(14)],
               steers=[Steer(1, "v", 3.0, prefix=(4, 10), suffix=(0, None))],
               collects=[Collect(2, 0, 30), Collect(4, 10, 30)])
    groups = [gA, gB, gC]
    targets = [5, 17, 100]
    eng = glm.GlmEngine(ck, device="cpu")
    out = eng.run(groups, {"v": vec}, targets, top_k=20)
    assert eng.dsa_rows > 0, "no query was long enough to use DSA"

    # the same thing as plain sequences for the reference
    seqs, where, steer_plan, collect_plan = [], [], {}, {}
    rows = []  # the engine layout's rows as (sequence, position)
    seqs.append(gA.prefix)
    rows += [(0, p) for p in range(len(gA.prefix))]
    where += [("A", r, 0, r) for r in range(len(gA.prefix) - 1)]  # (group, readout index, seq, position)
    for gr in (gB, gC):
        first = len(seqs)
        rows += [(first, p) for p in range(len(gr.prefix))]
        for k, s in enumerate(gr.suffixes):
            seqs.append(gr.prefix + s)
            rows += [(len(seqs) - 1, len(gr.prefix) + j) for j in range(len(s))]
    si = 1
    for k, s in enumerate(gB.suffixes):
        where.append(("B", k, si, len(gB.prefix) + len(s) - 1))
        si += 1
    for k, s in enumerate(gC.suffixes):
        where.append(("C", k, si, len(gC.prefix) + len(s) - 1))
        Lp = len(gC.prefix)
        steer_plan.setdefault(1, []).append((si, list(range(4, 10)) + list(range(Lp, Lp + len(s))), vec, 3.0))
        if k == 0:
            collect_plan.setdefault(2, []).append((("C", 0), si, 0, 30))
            collect_plan.setdefault(4, []).append((("C", 1), si, 10, 30))
        si += 1

    refs = {dt: run_reference(ck, cfg, dt, seqs, steer_plan, collect_plan) for dt in (BF16, F32)}
    cfg_dense = glm_ref.hf_config(d, index_topk=10**6)
    dense = run_reference(ck, cfg_dense, F32, seqs, steer_plan, collect_plan)

    # 3. the layer-by-layer harness against transformers' whole model
    from transformers import GlmMoeDsaForCausalLM

    for dt in (BF16, F32):
        with torch.device("meta"):
            model = GlmMoeDsaForCausalLM(cfg)
        model.load_state_dict(glm_ref.model_state_dict(cfg, ck, dt), strict=True, assign=True)
        model.model.rotary_emb = glm_ref.rotary(cfg, "cpu")
        model.eval()
        worst = 0.0
        for si, s in enumerate(seqs):
            hooks = []
            for li, plan in steer_plan.items():
                plan = [x for x in plan if x[0] == si]

                def hook(mod, inp, outp, plan=plan, dt=dt):  # the steer after layer li, in place
                    h = outp[0] if isinstance(outp, tuple) else outp
                    for _, p, v, coef in plan:
                        vv = (v.to(F32) * coef).to(BF16) if dt == BF16 else v.to(F32) * coef
                        h[0, p] = (h[0, p].to(vv.dtype) + vv).to(dt)

                hooks.append(model.model.layers[li].register_forward_hook(hook))
            with torch.no_grad():
                logits = model(torch.tensor([s])).logits[0].to(F32)
            for hk in hooks:
                hk.remove()
            lp = torch.log_softmax(logits, -1)
            worst = max(worst, (lp - refs[dt][0][si]).abs().max().item())
        print(f"3. harness ({str(dt)[6:]}): LayerwiseRef vs GlmMoeDsaForCausalLM max |dlogprob| {worst:.2e}")
        assert worst < (1e-2 if dt == BF16 else 1e-4), worst
        del model

    # 4. per layer, isolated: every implementation gets the fp32 reference's input to that layer
    res = isolated_layers(ck, cfg, cfg_dense, groups, seqs, rows, steer_plan)
    print("4. per layer, ||out - fp32 ref|| / ||fp32 ref update|| on the fp32 reference's inputs:")
    print("   layer  type          engine   transformers-bf16   dense-fp32 (DSA off)")
    for i, e, b, dd in res:
        kind = f"{CFG['mlp_layer_types'][i]}/{CFG['indexer_types'][i]}"
        print(f"   {i:5d}  {kind:12s}  {e:.5f}  {b:.5f}             {dd:.5f}")
    assert all(e <= 1.5 * b + 0.002 for _, e, b, _ in res), "engine error above the bf16 reference's"
    assert min(dd for *_, dd in res) > 3 * max(e for _, e, _, _ in res), "DSA off is as close: not tested"

    # 5. whole model, chained: the engine's own pass against the chained references
    def stats(lpA, lpB):
        top1, dmax, dsum, m = 0, 0.0, 0.0, 0
        for gid, r, si, p in where:
            ro = out[gid]["readouts"][r]
            ids = ro.top_ids
            a = ro.top_logprobs if lpA is None else lpA[si][p, ids]
            d = (a - lpB[si][p, ids]).abs()
            dmax, dsum, m = max(dmax, d.max().item()), dsum + d.sum().item(), m + d.numel()
            mine = int(ids[0]) if lpA is None else int(lpA[si][p].argmax())
            top1 += int(mine == int(lpB[si][p].argmax()))
        return top1, dmax, dsum / m

    n = len(where)
    t_e32, d_e32, m_e32 = stats(None, refs[F32][0])
    t_e16, d_e16, m_e16 = stats(None, refs[BF16][0])
    t_1632, d_1632, m_1632 = stats(refs[BF16][0], refs[F32][0])
    t_ed, d_ed, m_ed = stats(None, dense[0])
    nxt = max(abs(out["A"]["readouts"][r].next_logprob - refs[F32][0][0][r, gA.prefix[r + 1]].item())
              for r in range(len(gA.prefix) - 1))
    col = max((out["C"]["collects"][k] - refs[F32][1][("C", k)]).norm().item() /
              refs[F32][1][("C", k)].norm().item() for k in (0, 1))
    print(f"5. whole model, chained, {n} readouts ({eng.dsa_rows} DSA-cut queries at layer 0); top-1, max and mean |dlogprob| over the engine's top 20:")
    print(f"   engine   vs fp32 ref : {t_e32}/{n}, {d_e32:.4f}, {m_e32:.4f} (next-token logprob max {nxt:.4f})")
    print(f"   engine   vs bf16 ref : {t_e16}/{n}, {d_e16:.4f}, {m_e16:.4f}")
    print(f"   bf16 ref vs fp32 ref : {t_1632}/{n}, {d_1632:.4f}, {m_1632:.4f}  (bf16 noise floor)")
    print(f"   engine   vs dense fp32 ref (DSA off): {t_ed}/{n}, {d_ed:.4f}, {m_ed:.4f}  (sensitivity control)")
    print(f"   collects vs fp32 ref: max relative error {col:.4f}")
    assert t_e32 >= t_1632 - max(2, n // 20) and m_e32 <= 2 * m_1632 + 0.01, "engine further from fp32 than bf16 is"
    assert m_ed > 2.5 * m_e32, "the dense reference is as close as the sparse one: DSA is not being tested"
    print("glm tiny check: ok")


if __name__ == "__main__":
    main()
