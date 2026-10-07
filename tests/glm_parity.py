"""GLM-5.3 port gates on the real checkpoint (GPU box, ~/glmenv, weights in /nvme/glm53).

    python tests/glm_parity.py ref    MODEL OUT --layers 8   # gate 1: layers 0-7, one at a time
    python tests/glm_parity.py ref    MODEL OUT              # gate 2: all 78 layers (GPUs 1-3)
    python tests/glm_parity.py engine MODEL OUT              # gate 2: GlmEngine.run (GPU 0, alongside ref)
    python tests/glm_parity.py runner MODEL OUT              # kimisroom.run on every GPU, the production path
    python tests/glm_parity.py tinker MODEL OUT              # zai-org/GLM-5.3 on Tinker (kv exec TINKER_API_KEY=tinker)
    python tests/glm_parity.py report MODEL OUT              # the numbers and the verdicts

The inputs are real trials: dev task t04 at D0 and at D1.4 with GLM's frozen assent (about 2,500
tokens, so DeepSeek Sparse Attention cuts keys on every suffix row and on prefix rows past
index_topk), four battery items each (self_state, identification, belief_revision, yes_bias); the
D1.4 trial again with a random unit vector on the answer span after layer 37 at coefficients 8
and 64, and with a zero vector, which must reproduce the unsteered readouts bit for bit; and
neutral stories as plain text (about 2,600 tokens, every position read out, mean states
collected at layers 26/37/49/58 the way the vector extraction collects them).

`ref` runs transformers' GlmMoeDsaDecoderLayer (kimisroom/glm_ref.py) one layer at a time: an
fp32 chain, the truth, and a bf16 chain, steered and collected like the engine. At every layer
each implementation also gets the fp32 chain's input to that layer: the bf16 layer, the fp32
layer with DSA off (the sensitivity control) and the engine's layer, through the real
multi-group layout; error = ||out - fp32 out|| / ||fp32 update||. At the full-indexer layers the
engine's key selection is compared with the fp32 and bf16 references' on every row past
index_topk. `--smoke` runs everything on the CPU against a tiny checkpoint written by
`tests/glm_tiny_check.py --keep DIR` (token ids folded into its vocabulary).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kimisroom import glm  # noqa: E402
from kimisroom.engine import Collect, Group, Steer, _Layout, linear, rms_norm  # noqa: E402

F32, BF16 = torch.float32, torch.bfloat16
TASK = "t04"
KINDS = ("self_state", "identification", "belief_revision", "yes_bias")
STEER_LAYER, COEFS = 37, (8.0, 64.0)
COLLECT_LAYERS = (26, 37, 49, 58)
NAT_TOKENS = 2600
TOP_K, REF_TOP = 20, 50
TINKER_MODEL = "zai-org/GLM-5.3:peft:262144"


# ---------------------------------------------------------------- the trials

def build(model: str, tok_dir: str | None = None, smoke_cfg: dict | None = None):
    """(groups, zero-vector group, targets, steer layer, collect layers) for this checkpoint."""
    from transformers import AutoTokenizer

    from kimisroom.battery import load_battery
    from kimisroom.texts import load_assent, load_denial
    from kimisroom.trials import Turns, carry_over_group, load_jsonl

    r = glm.GlmRenderer(AutoTokenizer.from_pretrained(tok_dir or model))
    battery = load_battery(ROOT / "prompts/battery/items_idctl.jsonl", entity="GLM")
    items = [next(it for it in battery if it["kind"] == k) for k in KINDS]
    task = next(t for t in load_jsonl(ROOT / "prompts/tasks/neutral_tasks.jsonl") if t["id"] == TASK)
    glm.assert_not_confirmatory([task])
    cond = [{"role": "user", "content": load_denial(ROOT / "prompts/denial/d1_argument_4.md")},
            {"role": "assistant", "content": load_assent("glm53", "d1_argument_4")}]
    n_layers = (smoke_cfg or json.loads((Path(model) / "config.json").read_text()))["num_hidden_layers"]
    steer_layer = min(STEER_LAYER, n_layers - 2)
    collect_layers = tuple(sorted({min(L, n_layers - 1) for L in COLLECT_LAYERS}))
    d0 = carry_over_group(r, "D0", Turns(None, [], task["task"], task["answer"]), items)
    t14 = Turns(None, cond, task["task"], task["answer"])
    d14 = carry_over_group(r, "D1.4", t14, items)
    span = carry_over_group(r, "x", t14, items[:1], "answer", "rand", 1.0, steer_layer).steers[0].prefix
    stories = [s["text"] for s in load_jsonl(ROOT / "data/stimuli/k26-v2/stories.jsonl") if s["condition"] == "neutral"]
    text = ""
    for s in stories:
        text = f"{text}\n\n{s}" if text else s
        if len(r.ids(text)) >= NAT_TOKENS:
            break
    ids = r.ids(text)
    nat = Group("nat", ids, collects=[Collect(L, 0, len(ids)) for L in collect_layers], readout="prefix_all")
    groups = [nat, d0, d14] + [replace(d14, id=f"D1.4-s{c:g}", steers=[Steer(steer_layer, "rand", c, prefix=span)])
                               for c in COEFS]
    zero = replace(d14, id="D1.4-z", steers=[Steer(steer_layer, "zero", 1.0, prefix=span)])
    words = sorted({w for it in battery if isinstance(it["answer_format"], list) for w in it["answer_format"]} | {"Yes", "No"})
    targets = r.answer_tokens(words)
    if smoke_cfg:  # fold every token into the tiny vocabulary
        V = smoke_cfg["vocab_size"]

        def fold(g):
            return replace(g, prefix=[t % V for t in g.prefix], suffixes=[[t % V for t in s] for s in g.suffixes])

        groups, zero = [fold(g) for g in groups], fold(zero)
        targets = {w: t % V for w, t in targets.items()}
    return groups, zero, targets, steer_layer, collect_layers


def unit_vector(H: int) -> torch.Tensor:
    v = torch.randn(H, generator=torch.Generator().manual_seed(7))
    return v / v.norm()


def sequences(groups: list[Group]):
    """Each group as whole sequences for the reference: the prefix alone when it has no
    suffixes, else prefix + suffix for every suffix. Returns the sequences, the engine layout's
    rows as (sequence, start, end) blocks, the readouts as (group, index, sequence, position),
    the steers as (sequence, positions, vector, coef, layer) and the collects as
    (group, index, sequence, start, end, layer), and each group's first sequence."""
    seqs, blocks, where, steers, collects, firsts = [], [], [], [], [], {}
    for g in groups:
        first = firsts[g.id] = len(seqs)
        if not g.suffixes:
            seqs.append(list(g.prefix))
        for s in g.suffixes:
            seqs.append(list(g.prefix) + list(s))
        own = list(range(first, len(seqs)))
        blocks.append((first, 0, len(g.prefix)))
        Lp = len(g.prefix)
        for k, s in enumerate(g.suffixes):
            blocks.append((own[k], Lp, Lp + len(s)))
            if g.readout == "suffix_last":
                where.append((g.id, k, own[k], Lp + len(s) - 1))
        if g.readout == "prefix_all":
            where += [(g.id, p, first, p) for p in range(Lp - 1)]
        for st in g.steers:
            assert st.suffix is None
            for s in own:
                steers.append((s, list(range(*st.prefix)), st.vector, st.coef, st.layer))
        for k, c in enumerate(g.collects):
            collects.append((g.id, k, first, c.start, c.end, c.layer))
    return seqs, blocks, where, steers, collects, firsts


# ---------------------------------------------------------------- ref

def gather(hs: list[torch.Tensor], blocks, device=None) -> torch.Tensor:
    return torch.cat([hs[s][0, a:b].to(device or hs[s].device) for s, a, b in blocks])


def selection_rows(topk: torch.Tensor, S: int, k: int, device) -> torch.Tensor:
    """(S - k, S) bool: the keys a reference selected for every row past k (causal keys only)."""
    rows = topk[0, k:].long().to(device)
    allow = torch.ones(S, S, dtype=torch.bool, device=device).tril()[k:]
    return torch.zeros(S - k, S, dtype=torch.bool, device=device).scatter(-1, rows, True) & allow


@torch.no_grad()
def ref(a) -> None:
    from kimisroom import glm_ref

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    devs = [torch.device(x) for x in a.ref_devices.split(",")]
    d16, d32, de = devs[:3]
    dx16 = devs[3] if len(devs) > 3 else None  # the bf16 ref's routed experts, when one GPU cannot hold the layer
    ck = glm.GlmCheckpoint(a.model)
    cfg = glm_ref.hf_config(a.model)
    smoke = json.loads((Path(a.model) / "config.json").read_text()) if a.smoke else None
    groups, _, targets, steer_layer, collect_layers = build(a.model, a.tokenizer, smoke)
    seqs, blocks, where, steers, collects, firsts = sequences(groups)
    n = min(a.layers or cfg.num_hidden_layers, cfg.num_hidden_layers)
    K = cfg.index_topk
    H = cfg.hidden_size
    vecs = {"rand": unit_vector(H)}
    long_seqs = [s for s, q in enumerate(seqs) if len(q) > K]
    sel_seqs = [firsts[g.id] for g in groups if not g.steers and len(seqs[firsts[g.id]]) > K]
    d14, span = firsts["D1.4"], list(range(*next(g for g in groups if g.steers).steers[0].prefix))
    rows = [(s, p) for s, a0, b0 in blocks for p in range(a0, b0)]
    long_rows = torch.tensor([j for j, (s, p) in enumerate(rows) if p >= K], dtype=torch.long)

    emb = ck.load_global("model.embed_tokens.weight")
    h32 = [F.embedding(torch.tensor(q), emb).to(F32)[None] for q in seqs]  # chains live on the CPU
    h16 = [x.to(BF16) for x in h32]
    tk32: list = [None] * len(seqs)
    tk16: list = [None] * len(seqs)  # the bf16 chain's own selections
    tk16i: list = [None] * len(seqs)  # the bf16 layer's selections on the fp32 chain's inputs
    rot32, rot16 = glm_ref.rotary(cfg, d32), glm_ref.rotary(cfg, d16)
    eng = glm.GlmEngine(ck, device=de)
    eng._dsa, eng.dsa_rows = {}, 0
    layout = _Layout(groups, de)
    assert layout.token_ids == [seqs[s][p] for s, p in rows], "row map differs from the engine's layout"
    col32, col16, norms = {}, {}, []
    per_layer = []
    pool = ThreadPoolExecutor(1)
    nxt = pool.submit(ck.read_layer, 0)
    t_start = time.time()

    def run_all(layer, rot, dev, dtype, xs, tks, which):
        outs, sel = [None] * len(xs), list(tks)
        for s in which:
            o, t = glm_ref.run_layer(layer, rot, xs[s].to(dev, dtype), tks[s])
            outs[s], sel[s] = o.to("cpu"), t
        return outs, sel

    every = list(range(len(seqs)))
    for i in range(n):
        t0 = time.time()
        got = nxt.result()
        if i + 1 < n:
            nxt = pool.submit(ck.read_layer, i + 1)
        full = cfg.indexer_types[i] == "full"
        x_in = h32
        L32 = glm_ref.build_layer(cfg, ck, got, i, F32, d32)
        o32, t32 = run_all(L32, rot32, d32, F32, x_in, tk32, every)
        od = [None] * len(seqs)
        if long_seqs:  # DSA off, on the same inputs
            if full:
                L32.self_attn.indexer.index_topk = 10 ** 9
            for s in long_seqs:
                S = x_in[s].shape[1]
                prev = None if full else torch.arange(S, device=d32, dtype=torch.int32).expand(1, S, S)
                od[s] = glm_ref.run_layer(L32, rot32, x_in[s].to(d32), prev)[0].to("cpu")
        del L32
        L16 = glm_ref.build_layer(cfg, ck, got, i, BF16, d16, expert_device=dx16)
        o16i, t16i = run_all(L16, rot16, d16, BF16, [x.to(BF16) for x in x_in], tk16i, every)
        o16, t16 = run_all(L16, rot16, d16, BF16, h16, tk16, every)
        del L16
        if "cuda" in (d32.type, d16.type):  # with the fp32 ref on the CPU, the bf16 ref's cache still needs freeing
            torch.cuda.empty_cache()

        # the engine's layer on the fp32 chain's inputs, through the real layout
        w = eng._to_device(ck.pack_layer(got, i))
        he = gather(x_in, blocks, de).to(BF16)
        sel_stats = None
        if full and sel_seqs:
            sel_stats = {"rows": 0, "eng_same": 0, "bf16_same": 0, "eng_diff_keys": 0, "bf16_diff_keys": 0,
                         "kth_ties": 0}
            for s in sel_seqs:
                S = x_in[s].shape[1]
                x = rms_norm(x_in[s][0].to(de, BF16), w["input_layernorm.weight"], cfg.rms_norm_eps)  # the layer's attention input
                qa = rms_norm(linear(x, w["self_attn.q_a_proj.weight"], glm.TILE_ATTN), w["self_attn.q_a_layernorm.weight"], 1e-6)
                cos, sin = glm.rope_cos_sin(eng.inv_freq, torch.arange(S, device=de))
                iq, ik, iw = eng._index_qkw(w, x, qa, cos, sin)
                allow = torch.ones(S, S, dtype=torch.bool, device=de).tril()[K:]
                mine = eng._select(iq[K:], iw[K:], ik[None], allow)
                sc = eng._index_scores(iq[K:], iw[K:], ik[None], allow)
                kth = sc.sort(dim=-1, descending=True).values[:, K - 1 : K + 1]
                want = selection_rows(t32[s], S, K, de)
                other = selection_rows(t16i[s], S, K, de)
                sel_stats["rows"] += S - K
                sel_stats["eng_same"] += int((mine == want).all(-1).sum())
                sel_stats["bf16_same"] += int((other == want).all(-1).sum())
                sel_stats["eng_diff_keys"] += int((mine != want).sum())
                sel_stats["bf16_diff_keys"] += int((other != want).sum())
                sel_stats["kth_ties"] += int((kth[:, 0] == kth[:, 1]).sum())
                del iq, ik, iw, sc, mine, want, other
        eng.layer(w, he, layout)
        del w
        X, O32 = gather(x_in, blocks), gather(o32, blocks)
        upd = (O32 - X).norm()

        def err(o, idx=None):
            d, u = o.to(F32) - O32, O32 - X
            if idx is not None:
                d, u = d[idx], u[idx]
            return ((d.norm() / u.norm()).item()) if d.numel() else None

        rec = {"layer": i, "full": full, "moe": ck.is_moe(i), "update_rel": (upd / X.norm()).item(),
               "eng": err(he.cpu()), "bf16": err(gather(o16i, blocks)),
               "eng_long": err(he.cpu(), long_rows), "bf16_long": err(gather(o16i, blocks), long_rows),
               "dense_long": err(gather([od[s] if od[s] is not None else o32[s] for s in range(len(seqs))], blocks), long_rows)
               if long_seqs else None, "selection": sel_stats, "seconds": round(time.time() - t0, 1)}
        per_layer.append(rec)
        h32, tk32, h16, tk16, tk16i = o32, t32, o16, t16, t16i

        for s, pos, vname, coef, L in steers:
            if L != i:
                continue
            v = vecs[vname] * coef
            h32[s][0, pos] = h32[s][0, pos] + v
            h16[s][0, pos] = (h16[s][0, pos] + v.to(BF16)).to(BF16)
        for gid, k, s, a0, b0, L in collects:
            if L == i:
                col32[(gid, k)] = h32[s][0, a0:b0].to(F32).mean(0)
                col16[(gid, k)] = h16[s][0, a0:b0].to(F32).mean(0)
        # the residual's size where the steer goes, unsteered (for reading the coefficients)
        norms.append(h32[d14][0, span].norm(dim=-1).mean().item())
        print(f"layer {i:2d} {'full' if full else 'shrd'} {'moe' if rec['moe'] else 'dns'}  engine {rec['eng']:.5f}  "
              f"bf16 {rec['bf16']:.5f}  long rows: engine {rec['eng_long'] or 0:.5f} bf16 {rec['bf16_long'] or 0:.5f} "
              f"dense {rec['dense_long'] or 0:.5f}"
              + (f"  sel same eng {sel_stats['eng_same']}/{sel_stats['rows']} bf16 {sel_stats['bf16_same']}/{sel_stats['rows']}"
                 f" ties {sel_stats['kth_ties']}" if sel_stats else "") + f"  {rec['seconds']}s", flush=True)
    res = {"per_layer": per_layer, "answer_span_norm": norms, "dsa_rows_layer0": eng.dsa_rows, "layers": n,
           "seconds": round(time.time() - t_start), "targets": targets}
    if n == cfg.num_hidden_layers:
        norm_w, head_w = ck.load_global("model.norm.weight"), ck.load_global("lm_head.weight")
        tgt = torch.tensor(list(targets.values()))
        for name, hs, dt, dev in (("r32", h32, F32, d32), ("r16", h16, BF16, d16)):
            nw, hw = norm_w.to(dev), head_w.to(dev, dt)
            reads = {}
            for gid, k, s, p in where:
                reads.setdefault(s, []).append((gid, k, p))
            got_r = {}
            for s, want in reads.items():
                pos = torch.tensor([p for _, _, p in want])
                lp = glm_ref.final_logprobs(hs[s][0, pos].to(dev), nw, hw, cfg.rms_norm_eps, dt)
                top = lp.topk(REF_TOP, dim=-1)
                for j, (gid, k, p) in enumerate(want):
                    rec_r = {"top_ids": top.indices[j].cpu(), "top_lp": top.values[j].cpu(),
                             "target_lp": lp[j, tgt.to(dev)].cpu()}
                    if p + 1 < len(seqs[s]):
                        rec_r["next_lp"] = lp[j, seqs[s][p + 1]].item()
                    if gid != "nat":
                        rec_r["lp"] = lp[j].cpu()
                    got_r[(gid, k)] = rec_r
                del lp
            res[name] = got_r
            del hw
        res["collects32"], res["collects16"] = col32, col16
    torch.save(res, out / ("ref.pt" if n == cfg.num_hidden_layers else f"ref_{n}.pt"))
    print(f"ref: {n} layers in {res['seconds']}s -> {out}")


# ---------------------------------------------------------------- engine, runner

def engine(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = glm.GlmCheckpoint(a.model)
    smoke = json.loads((Path(a.model) / "config.json").read_text()) if a.smoke else None
    groups, zero, targets, *_ = build(a.model, a.tokenizer, smoke)
    H = ck.config.hidden_size
    eng = glm.GlmEngine(ck, device=a.engine_device)
    t0 = time.time()
    res = eng.run(groups + [zero], {"rand": unit_vector(H), "zero": torch.zeros(H)}, list(targets.values()), top_k=TOP_K)
    secs = time.time() - t0
    torch.save({"res": res, "dsa_rows": eng.dsa_rows, "seconds": secs, "timings": eng.timings,
                "tokens": sum(g.n_tokens for g in groups + [zero])}, out / "engine.pt")
    print(f"engine: {sum(g.n_tokens for g in groups + [zero]):,} tokens in {secs:.0f}s, "
          f"{eng.dsa_rows} DSA-cut queries per layer, timings {({k: round(v) for k, v in eng.timings.items()})}")


def runner(a) -> None:
    out = Path(a.out)
    ck = glm.GlmCheckpoint(a.model)
    groups, zero, targets, *_ = build(a.model, a.tokenizer)
    H = ck.config.hidden_size
    job = {"groups": groups + [zero], "vectors": {"rand": unit_vector(H), "zero": torch.zeros(H)},
           "targets": list(targets.values()), "top_k": TOP_K}
    torch.save(job, out / "runner.job.pt")
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "kimisroom.run", str(out / "runner.job.pt"), str(out / "runner.pt"),
                    "--model", a.model], check=True, cwd=ROOT)
    print(f"runner: {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- tinker

def tinker_ref(a) -> None:
    import tinker
    from tinker import types

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    groups, _, targets, *_ = build(a.model, a.tokenizer)
    seqs, _, where, _, _, _ = sequences([g for g in groups if not g.steers])
    tgt = list(targets.values())
    sampler = tinker.ServiceClient().create_sampling_client(base_model=TINKER_MODEL)
    params = types.SamplingParams(max_tokens=1, temperature=0.0)
    reads: dict[int, list] = {}
    for gid, k, s, p in where:
        reads.setdefault(s, []).append((gid, k, p))
    futs = {}
    for s, want in reads.items():
        prompt = seqs[s] + [tgt[0]]  # one token more, so the readout position is a prompt position
        T = torch.full((len(prompt) - 1, len(tgt)), -1, dtype=torch.long)
        for _, _, p in want:
            T[p] = torch.tensor(tgt)
        futs[s] = sampler.sample(prompt=types.ModelInput.from_ints(prompt), num_samples=1, sampling_params=params,
                                 include_prompt_logprobs=True, topk_prompt_logprobs=TOP_K,
                                 target_prompt_logprobs=types.TensorData.from_torch_sparse(T, pad_value=-1))
    res, n_prompt, n_cached = {}, 0, 0
    for s, f in futs.items():
        r = f.result()
        n_prompt += len(seqs[s]) + 1
        n_cached += r.prompt_cache_hit_tokens
        topk, plp, T = r.topk_prompt_logprobs, r.prompt_logprobs, r.target_prompt_logprobs.to_torch()
        for gid, k, p in reads[s]:
            pairs = topk[p + 1]
            res[(gid, k)] = {"top_ids": torch.tensor([t for t, _ in pairs]), "top_lp": torch.tensor([v for _, v in pairs]),
                             "target_lp": T[p].to(F32), "next_lp": plp[p + 1] if p + 1 < len(seqs[s]) else None}
    cost = (n_prompt - n_cached) * 4.86e-6 + n_cached * 0.972e-6 + len(futs) * 12.15e-6
    torch.save({"res": res, "prompt_tokens": n_prompt, "cached_tokens": n_cached, "usd": cost, "model": TINKER_MODEL},
               out / "tinker.pt")
    print(f"tinker: {len(futs)} prompts, {n_prompt:,} tokens ({n_cached:,} cached), about ${cost:.2f}")


# ---------------------------------------------------------------- report

def compare(A: dict, B: dict, keys, full_b: bool) -> dict:
    """A's readouts against B's: top-1 agreement, |dlogprob| on A's top ids (B's full vector when
    it has one, else only ids in B's top list), on the targets, and on the next token."""
    top1 = n = 0
    d_top, d_tgt, d_next = [], [], []
    for key in keys:
        a, b = A[key], B[key]
        n += 1
        top1 += int(int(a["top_ids"][0]) == int(b["top_ids"][0]))
        ids = a["top_ids"][:TOP_K]
        if full_b and "lp" in b:
            d_top.append((a["top_lp"][:TOP_K] - b["lp"][ids]).abs())
        else:
            bl = dict(zip(b["top_ids"].tolist(), b["top_lp"].tolist()))
            m = [(float(x), bl[int(t)]) for t, x in zip(ids, a["top_lp"][:TOP_K]) if int(t) in bl]
            if m:
                d_top.append(torch.tensor([abs(x - y) for x, y in m]))
        keep = (a["target_lp"] > -15) | (b["target_lp"] > -15)
        d_tgt.append((a["target_lp"] - b["target_lp"].to(F32))[keep].abs())
        if a.get("next_lp") is not None and b.get("next_lp") is not None:
            d_next.append(abs(a["next_lp"] - b["next_lp"]))
    cat = torch.cat(d_top) if d_top else torch.zeros(1)
    tg = torch.cat(d_tgt) if d_tgt else torch.zeros(1)
    return {"n": n, "top1": top1, "top_mean": cat.mean().item(), "top_max": cat.max().item(),
            "target_mean": tg.mean().item(), "target_max": tg.max().item(),
            "next_mean": (sum(d_next) / len(d_next)) if d_next else None}


def as_read(r) -> dict:
    return {"top_ids": r.top_ids, "top_lp": r.top_logprobs, "target_lp": r.target_logprobs, "next_lp": r.next_logprob}


def report(a) -> None:
    out = Path(a.out)
    summary: dict = {}
    ok = True
    # A gate passes only on a finished reference: ref_N.pt for gate 1 (report --layers N), ref.pt with every
    # layer and the engine's whole-model run for gate 2. A crashed ref writes no file, which used to pass.
    n_all = json.loads((Path(a.model) / "config.json").read_text())["num_hidden_layers"]
    want = out / (f"ref_{a.layers}.pt" if a.layers else "ref.pt")
    if not want.exists():
        print(f"missing {want.name}: the reference did not finish")
        ok = False
    for f in sorted(out.glob("ref*.pt")):
        R = torch.load(f, weights_only=False)
        pl = R["per_layer"]
        expected = int(f.stem.split("_")[1]) if "_" in f.stem else n_all
        if len(pl) != expected or not pl:
            print(f"{f.name}: {len(pl)} layers, expected {expected}")
            ok = False
            continue
        bad = [r["layer"] for r in pl if r["eng"] > 1.5 * r["bf16"] + 0.002]
        ratio = sorted(r["eng"] / max(r["bf16"], 1e-12) for r in pl)
        sel = [r["selection"] for r in pl if r["selection"]]
        rows = sum(s["rows"] for s in sel)
        long_ = [r for r in pl if r["eng_long"] is not None]
        summ = {"layers": len(pl), "engine_over_bf16_median": ratio[len(ratio) // 2], "engine_over_bf16_max": ratio[-1],
                "engine_err_max": max(r["eng"] for r in pl), "bf16_err_max": max(r["bf16"] for r in pl),
                "layers_failing": bad,
                "selection_rows": rows, "selection_engine_same": sum(s["eng_same"] for s in sel),
                "selection_bf16_same": sum(s["bf16_same"] for s in sel),
                "selection_engine_diff_keys": sum(s["eng_diff_keys"] for s in sel),
                "selection_bf16_diff_keys": sum(s["bf16_diff_keys"] for s in sel),
                "selection_kth_ties": sum(s["kth_ties"] for s in sel),
                "long_rows_engine_max": max((r["eng_long"] for r in long_), default=None),
                "long_rows_bf16_max": max((r["bf16_long"] for r in long_), default=None),
                "long_rows_dense_median": sorted(r["dense_long"] for r in long_)[len(long_) // 2] if long_ else None,
                "seconds": R["seconds"]}
        ok &= not bad
        print(f"{f.name}: per layer, isolated, {len(pl)} layers: engine/bf16 error ratio median {summ['engine_over_bf16_median']:.2f}, "
              f"max {summ['engine_over_bf16_max']:.2f}; worst engine {summ['engine_err_max']:.5f} vs worst bf16 "
              f"{summ['bf16_err_max']:.5f}; failing layers {bad or 'none'}")
        print(f"   DSA selection, {rows} rows past index_topk at the full layers: engine identical to fp32 on "
              f"{summ['selection_engine_same']}, bf16 on {summ['selection_bf16_same']}; keys differing "
              f"{summ['selection_engine_diff_keys']} (engine) vs {summ['selection_bf16_diff_keys']} (bf16); rows with "
              f"a tie at the k-th score {summ['selection_kth_ties']}")
        if long_:
            print(f"   rows past index_topk: engine max {summ['long_rows_engine_max']:.5f}, bf16 max "
                  f"{summ['long_rows_bf16_max']:.5f}, DSA off median {summ['long_rows_dense_median']:.5f}")
        summary[f.stem] = summ
        if "r32" not in R:
            continue
        E = torch.load(out / "engine.pt", weights_only=False) if (out / "engine.pt").exists() else None
        if E is None:
            print("missing engine.pt: gate 2 needs the engine's whole-model run")
            ok = False
        r32, r16 = R["r32"], R["r16"]
        trial = [k for k in r32 if k[0] != "nat"]
        nat = [k for k in r32 if k[0] == "nat"]
        res = {"bf16 vs fp32 (trials)": compare(r16, r32, trial, True), "bf16 vs fp32 (nat)": compare(r16, r32, nat, False)}
        if E:
            eng = {(gid, k): as_read(x) for gid, v in E["res"].items() for k, x in enumerate(v["readouts"])}
            res["engine vs fp32 (trials)"] = compare(eng, r32, trial, True)
            res["engine vs fp32 (nat)"] = compare(eng, r32, nat, False)
            res["engine vs bf16 (trials)"] = compare(eng, r16, trial, True)
            z = all(torch.equal(x.top_logprobs, y.top_logprobs) and torch.equal(x.target_logprobs, y.target_logprobs)
                    for x, y in zip(E["res"]["D1.4"]["readouts"], E["res"]["D1.4-z"]["readouts"]))
            res["zero vector reproduces unsteered"] = z
            for c in COEFS:
                sid = f"D1.4-s{c:g}"
                eff_e = torch.cat([(as_read(x)["target_lp"] - as_read(y)["target_lp"]).abs() for x, y in
                                   zip(E["res"][sid]["readouts"], E["res"]["D1.4"]["readouts"])]).mean().item()
                eff_r = torch.cat([(r32[(sid, k)]["target_lp"] - r32[("D1.4", k)]["target_lp"]).abs()
                                   for k in range(len(KINDS))]).mean().item()
                res[f"steer {c:g}: mean |change| in target logprobs, engine / fp32"] = (eff_e, eff_r)
            col_e = E["res"]["nat"]["collects"]
            res["collects: engine vs fp32, bf16 vs fp32 (max relative)"] = (
                max(((col_e[k] - R["collects32"][("nat", k)]).norm() / R["collects32"][("nat", k)].norm()).item()
                    for k in range(len(col_e))),
                max(((R["collects16"][("nat", k)] - R["collects32"][("nat", k)]).norm() / R["collects32"][("nat", k)].norm()).item()
                    for k in range(len(col_e))))
            ppl = {name: math.exp(-sum(d[k]["next_lp"] for k in nat if d[k].get("next_lp") is not None) / len(nat))
                   for name, d in (("fp32", r32), ("bf16", r16), ("engine", eng))}
            res["nat perplexity"] = ppl
            t, f16 = res["engine vs fp32 (trials)"], res["bf16 vs fp32 (trials)"]
            tn, f16n = res["engine vs fp32 (nat)"], res["bf16 vs fp32 (nat)"]
            g2 = (t["n"] - t["top1"] <= 1.5 * (f16["n"] - f16["top1"]) + 1 and t["top_mean"] <= 2 * f16["top_mean"] + 0.01
                  and tn["n"] - tn["top1"] <= 1.5 * (f16n["n"] - f16n["top1"]) + 2 and tn["top_mean"] <= 2 * f16n["top_mean"] + 0.01
                  and z)
            res["gate 2 (whole model)"] = g2
            ok &= g2
            if (out / "runner.pt").exists():
                Rn = torch.load(out / "runner.pt", weights_only=False)
                same = all(torch.equal(x.top_logprobs, y.top_logprobs) and torch.equal(x.top_ids, y.top_ids)
                           and torch.equal(x.target_logprobs, y.target_logprobs)
                           for gid in E["res"] for x, y in zip(E["res"][gid]["readouts"], Rn[gid]["readouts"]))
                same &= all(torch.equal(x, y) for x, y in zip(E["res"]["nat"]["collects"], Rn["nat"]["collects"]))
                res["runner (4 GPUs) == engine (1 GPU), bitwise"] = same
                ok &= same
            if (out / "tinker.pt").exists():
                T = torch.load(out / "tinker.pt", weights_only=False)
                keys = [k for k in T["res"] if k in eng]
                tr = [k for k in keys if k[0] != "nat"]
                nk = [k for k in keys if k[0] == "nat"]
                res["engine vs tinker (trials)"] = compare(eng, T["res"], tr, False)
                res["fp32 vs tinker (trials)"] = compare(r32, T["res"], tr, False)
                res["engine vs tinker (nat)"] = compare(eng, T["res"], nk, False)
                res["fp32 vs tinker (nat)"] = compare(r32, T["res"], nk, False)
                res["tinker nat perplexity"] = math.exp(-sum(T["res"][k]["next_lp"] for k in nk) / len(nk))
                res["tinker usd"] = T["usd"]
        for k, v in res.items():
            print(f"   {k}: {v}")
        summary["whole_model"] = res
    summary["verdict"] = "pass" if ok else "FAIL"
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"verdict: {summary['verdict']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "engine", "runner", "tinker", "report"])
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=0, help="ref: only the first N layers (gate 1); report: gate 1's N")
    ap.add_argument("--tokenizer", help="tokenizer files, when MODEL has none (--smoke)")
    ap.add_argument("--ref-devices", default="cuda:1,cuda:2,cuda:3", help="bf16 ref, fp32 ref, isolated engine[, bf16 ref's experts]")
    ap.add_argument("--engine-device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true", help="CPU, tiny checkpoint")
    a = ap.parse_args()
    if a.smoke:
        a.ref_devices, a.engine_device = "cpu,cpu,cpu", "cpu"
        torch.cuda.synchronize = lambda *args, **kw: None
    {"ref": ref, "engine": engine, "runner": runner, "tinker": tinker_ref, "report": report}[a.mode](a)


if __name__ == "__main__":
    main()
