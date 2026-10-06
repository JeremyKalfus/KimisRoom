"""GLM-5.3 (zai-org/GLM-5.3, model_type `glm_moe_dsa`) on the layer-wise prefill engine.

GLM-5.3 keeps DeepSeek-V3's layout: MLA attention, three dense layers, then 75 MoE layers with
a sigmoid router, a correction bias, 256 experts (top 8, normalized, scaled by 2.5) and one shared
expert. `engine.Engine` already does all of that, so this module only adds what differs, by
subclassing. `engine.py` itself is untouched (it is hash-frozen in prereg/manifest.txt).

1. **FP8 weights.** Every linear layer except the router, the norms, the embeddings, lm_head and
   `indexer.weights_proj` is e4m3 with one fp32 `weight_scale_inv` per 128x128 block. A weight is
   the fp8 value times its block's scale in fp32, rounded once to bf16, which is what DeepSeek's
   reference `weight_dequant` and transformers' `Fp8Dequantize` compute. Non-expert weights are
   dequantized when a layer reaches the GPU; experts one at a time, when they run, like K2.6's
   INT4 experts. Activations stay bf16 (weight-only dequant, DeepSeek's default `gemm_impl`).
2. **No remote code.** The architecture is in-tree in transformers 5.x, so the config is read from
   config.json (`rope_parameters`, `indexer_types`, `mlp_layer_types`) and the rotary embedding
   is implemented here op for op after transformers' `GlmMoeDsaRotaryEmbedding` and
   `apply_rotary_pos_emb_interleave` (rope_theta 8e6, interleaved pairs, no scaling).
3. **Head sizes.** qk is 192 + 64 (softmax scale 256 ** -0.5) and v_head_dim is 256; the
   engine's attention already reads them from the config.
4. **DeepSeek Sparse Attention.** On the 21 "full" layers a lightning indexer scores each query's
   causal keys and the query attends to its top `index_topk` (2048) only; the 57 "shared" layers
   reuse the last full layer's selection. A query with at most 2048 causal keys keeps all of them,
   so DSA is dense attention there and the indexer is skipped; past that it runs exactly as
   transformers' `GlmMoeDsaIndexer` (fp32 scores, ReLU, per-head weights, top-k). A D1.4 trial is
   about 2,050-2,100 tokens, so the last tokens of its longest suffixes need it.

The MTP head (layer 78) is never loaded. Weights are read per layer straight from the original
shards: a layer's tensors are read as merged contiguous byte runs, so each byte of the checkpoint
is read once per pass even though most shards hold parts of two layers.

The readout renderer (`GlmRenderer`) is the other GLM-specific piece: GLM-5.3 always thinks (its
generation prompt ends in `<think>`, with no switch), so the first answer token is read after an
empty think block, `<think></think>`, the form the template gives every earlier assistant turn.
That choice is a PREREG amendment item (drafts/prereg-amendment-glm.md).
"""

from __future__ import annotations

import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .engine import (BF16, F32, TILE_ATTN, Checkpoint, Engine, _GroupLayout, linear, rms_norm)
from .trials import Renderer

FP8 = torch.float8_e4m3fn
BLOCK = 128
PROJ = ("gate_proj", "up_proj", "down_proj")
# safetensors dtype -> how a layer's tensors are held on the CPU; fp8 stays raw bytes until it
# reaches the GPU, so no CPU fp8 kernel (stack, copy into shared memory) is ever needed
_CPU_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64,
               "F8_E4M3": torch.uint8, "U8": torch.uint8}
CHUNK = 64 << 20


# ---------------------------------------------------------------- config


def glm_config(raw: dict) -> SimpleNamespace:
    """The engine's view of a glm_moe_dsa config.json, with every assumption the port makes checked."""
    assert raw["model_type"] == "glm_moe_dsa", raw["model_type"]
    q = raw.get("quantization_config") or {}
    assert q.get("quant_method") == "fp8" and q.get("fmt") == "e4m3" and q.get("weight_block_size") == [BLOCK, BLOCK], q
    rp = raw.get("rope_parameters") or {}
    assert rp.get("rope_type", "default") == "default", f"only default RoPE is implemented: {rp}"
    assert raw.get("rope_interleave", True) and raw.get("indexer_rope_interleave", True)
    assert raw["num_key_value_heads"] == raw["num_attention_heads"]
    assert raw["n_group"] == 1 and raw["topk_group"] == 1 and raw["scoring_func"] == "sigmoid"
    assert raw["topk_method"] == "noaux_tc" and raw["hidden_act"] == "silu" and not raw.get("attention_bias")
    c = SimpleNamespace(**{k: v for k, v in raw.items() if k != "quantization_config"})
    n, k = c.num_hidden_layers, c.first_k_dense_replace
    c.rope_theta = rp["rope_theta"]
    c.rope_scaling = None
    c.moe_layer_freq = 1
    c.mlp_layer_types = raw.get("mlp_layer_types") or ["dense"] * k + ["sparse"] * (n - k)
    assert c.mlp_layer_types == ["dense"] * k + ["sparse"] * (n - k), "dense layers must come first"
    if raw.get("indexer_types") is None:  # transformers' rule from index_topk_freq / index_skip_topk_offset
        f, off = raw["index_topk_freq"], raw["index_skip_topk_offset"]
        c.indexer_types = ["full" if (max(i - off + 1, 0) % f) == 0 else "shared" for i in range(n)]
    assert len(c.indexer_types) == n and c.indexer_types[0] == "full", "layer 0 must run its own indexer"
    return c


# ---------------------------------------------------------------- fp8


def dequant_fp8(w: torch.Tensor, s: torch.Tensor, dtype: torch.dtype = BF16, block: int = BLOCK) -> torch.Tensor:
    """(R, C) e4m3 weight (or its raw bytes) times its (ceil(R/128), ceil(C/128)) block scales,
    in fp32, then one rounding to `dtype`. Any R and C (kv_a_proj_with_mqa has 576 rows)."""
    R, C = w.shape
    x = (w.view(FP8) if w.dtype == torch.uint8 else w).to(F32)
    sf = s.to(F32).repeat_interleave(block, 0)[:R].repeat_interleave(block, 1)[:, :C]
    return (x * sf).to(dtype)


def _dequant_blocks(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """dequant_fp8 for whole blocks only (every expert matrix), as reshapes, so it compiles to one kernel."""
    R, C = w.shape
    x = w.view(FP8).to(F32).view(R // BLOCK, BLOCK, C // BLOCK, BLOCK)
    return (x * s.view(R // BLOCK, 1, C // BLOCK, 1)).view(R, C).to(BF16)


_compiled = None
_dequant_fn: dict[tuple, object] = {}  # matrix shape -> the dequant checked bit-identical for it


def dequant_expert(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """One expert matrix, fused by torch.compile when the compiled kernel reproduces the eager one
    bit for bit on the first matrix of that shape (checked here, and in tests/glm_fp8_check.py),
    eager otherwise."""
    global _compiled
    fn = _dequant_fn.get(tuple(w.shape))
    if fn is not None:
        return fn(w, s)
    ref = _dequant_blocks(w, s)
    fn = _dequant_blocks
    try:
        if _compiled is None:
            _compiled = torch.compile(_dequant_blocks, dynamic=False, fullgraph=True)
        if torch.equal(_compiled(w, s), ref):
            fn = _compiled
        else:
            print(f"glm: compiled fp8 dequant differs from eager for {tuple(w.shape)}, using eager", flush=True)
    except Exception as e:  # noqa: BLE001 (any compile failure: fall back, same numbers)
        print(f"glm: torch.compile of the fp8 dequant failed ({type(e).__name__}: {e}), using eager", flush=True)
    _dequant_fn[tuple(w.shape)] = fn
    return ref


# ---------------------------------------------------------------- rope


def rope_interleave(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """transformers' apply_rotary_pos_emb_interleave for one tensor: pairs (x0, x1), (x2, x3), ...
    each rotated by one frequency, output de-interleaved (evens then odds). cos/sin are
    cat(freqs, freqs) already broadcastable to x except in the last dim."""
    c = cos[..., : cos.shape[-1] // 2]
    s = sin[..., : sin.shape[-1] // 2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def rope_inv_freq(c: SimpleNamespace) -> torch.Tensor:
    """GlmMoeDsaRotaryEmbedding.compute_default_rope_parameters (head_dim is qk_rope_head_dim), on the CPU."""
    dim = c.qk_rope_head_dim
    return 1.0 / (c.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))


def rope_cos_sin(inv_freq: torch.Tensor, positions: torch.Tensor, dtype: torch.dtype = BF16):
    """(T, dim) cos and sin for these positions: fp32 angles, cos/sin in fp32, one cast, as transformers."""
    freqs = positions.to(F32)[:, None] * inv_freq.to(positions.device)[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


# ---------------------------------------------------------------- checkpoint


class GlmCheckpoint(Checkpoint):
    """Lazy access to a local snapshot of zai-org/GLM-5.3 (FP8)."""

    def __init__(self, path: str | os.PathLike):  # noqa: D107 (no remote code: Checkpoint.__init__ is not called)
        self.path = Path(path)
        raw = json.loads((self.path / "config.json").read_text())
        self.config = glm_config(raw)
        self.prefix = ""
        self.int4_group = None
        self.fp8_block = BLOCK
        index = json.loads((self.path / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._handles: dict[str, object] = {}
        self._lock = threading.Lock()
        self._headers: dict[str, tuple[int, dict]] = {}
        self._file_layers: dict[str, set] = {}
        for k, f in self.weight_map.items():
            parts = k.split(".")
            self._file_layers.setdefault(f, set()).add(int(parts[2]) if parts[:2] == ["model", "layers"] else None)
        self._io = ThreadPoolExecutor(8)

    def is_moe(self, i: int) -> bool:
        return self.config.mlp_layer_types[i] == "sparse"

    def _header(self, fname: str) -> tuple[int, dict]:
        with self._lock:
            h = self._headers.get(fname)
        if h is None:
            with open(self.path / fname, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                h = (8 + n, json.loads(f.read(n)))
            with self._lock:
                self._headers[fname] = h
        return h

    def _read_tensors(self, fname: str, keys: list[str]) -> dict[str, torch.Tensor]:
        """These tensors of one shard, read as merged contiguous byte runs with parallel preads."""
        base, header = self._header(fname)
        spans = sorted((base + header[k]["data_offsets"][0], base + header[k]["data_offsets"][1], k) for k in keys)
        runs: list[list] = []  # [file start, file end, [(start, end, key), ...]]
        for a, b, k in spans:
            if runs and a == runs[-1][1]:
                runs[-1][1] = b
                runs[-1][2].append((a, b, k))
            else:
                runs.append([a, b, [(a, b, k)]])
        buf = torch.empty(sum(b - a for a, b, _ in runs), dtype=torch.uint8)
        mv = memoryview(buf.numpy())
        jobs, off = [], 0
        for a, b, _ in runs:
            jobs += [(s, off + s - a, min(s + CHUNK, b) - s) for s in range(a, b, CHUNK)]
            off += b - a
        fd = os.open(self.path / fname, os.O_RDONLY)

        def read(job) -> None:
            pos, at, n = job
            done = 0
            while done < n:
                got = os.preadv(fd, [mv[at + done : at + n]], pos + done) if hasattr(os, "preadv") else \
                    _pread_into(fd, mv[at + done : at + n], pos + done)
                assert got > 0, f"short read in {fname} at {pos + done}"
                done += got

        try:
            list(self._io.map(read, jobs))
        finally:
            os.close(fd)
        out, off = {}, 0
        for a, b, members in runs:
            for ta, tb, k in members:
                meta = header[k]
                dt = _CPU_DTYPES[meta["dtype"]]
                lo = off + ta - a
                raw = buf[lo : lo + tb - ta]
                if lo % dt.itemsize:
                    raw = raw.clone()
                out[k] = raw.view(dt).view(meta["shape"])
            off += b - a
        return out

    def layer_names(self, i: int) -> tuple[list[str], list[str]]:
        """(plain tensors, fp8 linear weights without suffix) of decoder layer i, experts excluded."""
        c = self.config
        plain = ["input_layernorm.weight", "post_attention_layernorm.weight",
                 "self_attn.q_a_layernorm.weight", "self_attn.kv_a_layernorm.weight"]
        lin = ["self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_a_proj_with_mqa",
               "self_attn.kv_b_proj", "self_attn.o_proj"]
        if c.indexer_types[i] == "full":
            plain += ["self_attn.indexer.k_norm.weight", "self_attn.indexer.k_norm.bias",
                      "self_attn.indexer.weights_proj.weight"]
            lin += ["self_attn.indexer.wq_b", "self_attn.indexer.wk"]
        if self.is_moe(i):
            plain += ["mlp.gate.weight", "mlp.gate.e_score_correction_bias"]
            lin += [f"mlp.shared_experts.{m}" for m in PROJ]
        else:
            lin += [f"mlp.{m}" for m in PROJ]
        return plain, lin

    def read_layer(self, i: int) -> dict[str, torch.Tensor]:
        """Every tensor of decoder layer i under its checkpoint name minus the layer prefix,
        fp8 as raw bytes; views into the read buffers (callers copy what they keep)."""
        assert 0 <= i < self.config.num_hidden_layers, f"layer {i} (the MTP head is {self.config.num_hidden_layers})"
        p = f"model.layers.{i}."
        plain, lin = self.layer_names(i)
        names = plain + [f"{n}.weight" for n in lin] + [f"{n}.weight_scale_inv" for n in lin]
        if self.is_moe(i):
            names += [f"mlp.experts.{e}.{m}.{s}" for e in range(self.config.n_routed_experts) for m in PROJ
                      for s in ("weight", "weight_scale_inv")]
        by_file: dict[str, list[str]] = {}
        for n in names:
            by_file.setdefault(self.weight_map[p + n], []).append(p + n)
        got: dict[str, torch.Tensor] = {}
        for f, ks in sorted(by_file.items()):
            got.update(self._read_tensors(f, ks))
        return {k[len(p):]: v for k, v in got.items()}

    def load_layer_cpu(self, i: int, shared: bool = False) -> dict[str, torch.Tensor]:
        """Decoder layer i in CPU memory: plain tensors, fp8 linears as (bytes, scale) pairs, experts
        stacked along dim 0 (`experts.<proj>.weight` bytes, `experts.<proj>.scale`), and its index."""
        return self.pack_layer(self.read_layer(i), i, shared)

    def pack_layer(self, got: dict[str, torch.Tensor], i: int, shared: bool = False) -> dict[str, torch.Tensor]:
        """read_layer's tensors in the engine's layout, as private (or shared-memory) copies."""

        def empty(shape, dtype: torch.dtype) -> torch.Tensor:
            if not shared:
                return torch.empty(shape, dtype=dtype)
            st = torch.UntypedStorage._new_shared(math.prod(shape) * dtype.itemsize)
            return torch.empty(0, dtype=dtype).set_(st, 0, shape)

        def own(x: torch.Tensor) -> torch.Tensor:
            return empty(tuple(x.shape), x.dtype).copy_(x)

        def stack(xs: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(xs, out=empty((len(xs), *xs[0].shape), xs[0].dtype))

        plain, lin = self.layer_names(i)
        t = {n: own(got[n]) for n in plain}
        for n in lin:
            t[f"{n}.weight"] = own(got[f"{n}.weight"])
            t[f"{n}.weight_scale_inv"] = own(got[f"{n}.weight_scale_inv"])
        if self.is_moe(i):
            E = self.config.n_routed_experts
            for m in PROJ:
                t[f"experts.{m}.weight"] = stack([got[f"mlp.experts.{e}.{m}.weight"] for e in range(E)])
                t[f"experts.{m}.scale"] = stack([got[f"mlp.experts.{e}.{m}.weight_scale_inv"] for e in range(E)])
        t["layer_index"] = own(torch.tensor([i], dtype=torch.int64))
        return t


def _pread_into(fd: int, view: memoryview, pos: int) -> int:
    data = os.pread(fd, len(view), pos)
    view[: len(data)] = data
    return len(data)


# ---------------------------------------------------------------- engine


class GlmEngine(Engine):
    """engine.Engine with GLM-5.3's weights, rope and sparse attention; everything else (layout,
    steering, collects, readouts, batch-invariant tiling, routing, expert combine) is inherited."""

    def __init__(self, ckpt: GlmCheckpoint, device: str | torch.device = "cuda", moe_chunk: int = 65536,
                 suffix_chunk: int = 16, router_bias_dtype: torch.dtype | None = None):  # noqa: D107
        self.ckpt = ckpt
        self.c = c = ckpt.config
        self.device = torch.device(device)
        self.moe_chunk = moe_chunk
        self.suffix_chunk = suffix_chunk
        self.router_bias_dtype = router_bias_dtype  # None keeps the checkpoint's fp32 bias, as transformers
        self.q_head_dim = c.qk_nope_head_dim + c.qk_rope_head_dim
        self.softmax_scale = self.q_head_dim ** (-0.5)
        self.inv_freq = rope_inv_freq(c).to(self.device)
        self.topk = c.index_topk
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._dsa: dict = {}
        self.dsa_rows = 0  # queries whose keys DSA cut in the last run, counted at layer 0 (a diagnostic)

    def run(self, *args, **kwargs):  # noqa: D102 (Engine.run, with the sparse-attention state reset)
        self._dsa = {}
        self.dsa_rows = 0
        return super().run(*args, **kwargs)

    # ---------------- weights

    def _to_device(self, cpu: dict[str, torch.Tensor]) -> dict:
        dev = self.device
        w: dict = {}
        for k, v in cpu.items():
            if k.endswith(".weight_scale_inv") or k == "layer_index":
                continue
            w[k] = v.to(dev)
        for k in list(w):
            sk = k[: -len(".weight")] + ".weight_scale_inv" if k.endswith(".weight") else None
            if sk in cpu:
                w[k] = dequant_fp8(w[k], cpu[sk].to(dev))
        if "mlp.gate.weight" in w:
            w["gate.weight"] = w.pop("mlp.gate.weight")
            bias = w.pop("mlp.gate.e_score_correction_bias")
            w["gate.bias"] = bias.to(self.router_bias_dtype or bias.dtype).to(F32)
        w["layer"] = int(cpu["layer_index"][0])
        return w

    def _expert(self, w: dict, e: int):
        return [dequant_expert(w[f"experts.{m}.weight"][e], w[f"experts.{m}.scale"][e]) for m in PROJ]

    # ---------------- sparse attention

    def _index_qkw(self, w: dict, x: torch.Tensor, qa: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """GlmMoeDsaIndexer's projections for every token: q (T, H, D) and k (T, D) with interleaved
        rope on their first qk_rope_head_dim dims, and the per-head weights (T, H) in fp32."""
        c = self.c
        T, H, D, dr = x.shape[0], c.index_n_heads, c.index_head_dim, c.qk_rope_head_dim
        q = linear(qa, w["self_attn.indexer.wq_b.weight"], TILE_ATTN).view(T, H, D)
        k = linear(x, w["self_attn.indexer.wk.weight"], TILE_ATTN)
        k = F.layer_norm(k, (D,), w["self_attn.indexer.k_norm.weight"], w["self_attn.indexer.k_norm.bias"], 1e-6)
        q_rot, q_pass = torch.split(q, [dr, D - dr], dim=-1)
        k_rot, k_pass = torch.split(k, [dr, D - dr], dim=-1)
        q = torch.cat([rope_interleave(q_rot, cos[:, None, :], sin[:, None, :]), q_pass], dim=-1)
        k = torch.cat([rope_interleave(k_rot, cos, sin), k_pass], dim=-1)
        wts = F.linear(x.to(F32), w["self_attn.indexer.weights_proj.weight"].to(F32)) * (H ** -0.5)
        return q, k, wts

    @staticmethod
    def _index_scores(q: torch.Tensor, wts: torch.Tensor, keys: torch.Tensor, allow: torch.Tensor) -> torch.Tensor:
        """GlmMoeDsaIndexer's scores for some queries: q (..., H, D), wts (..., H), keys (..., K, D)
        with the leading dims broadcastable, allow (..., K) bool; -inf where a key is not allowed."""
        D = q.shape[-1]
        scores = torch.matmul(q.to(F32), keys.to(F32).transpose(-1, -2)) * (D ** -0.5)  # (..., H, K)
        scores = F.relu(scores)
        index = torch.matmul(wts.to(F32).unsqueeze(-2), scores).squeeze(-2)  # (..., K)
        return index.masked_fill(~allow, float("-inf"))

    def _select(self, q: torch.Tensor, wts: torch.Tensor, keys: torch.Tensor, allow: torch.Tensor) -> torch.Tensor:
        """True where a key is allowed and among the query's topk by index score. Rows with at most
        topk allowed keys keep them all. Equal scores (the ReLU makes exact zeros common) go to the
        earlier key: a stable sort, so the choice never depends on a row's padding or length, and
        it is the order CUDA's topk takes tied values in (gatherTopK writes them in index order)."""
        index = self._index_scores(q, wts, keys, allow)
        k = min(self.topk, index.shape[-1])
        idx = torch.sort(index, dim=-1, descending=True, stable=True).indices[..., :k]
        sel = torch.zeros_like(allow).scatter(-1, idx, True)
        return sel & allow

    def _attention(self, w: dict, x: torch.Tensor, g: _GroupLayout) -> torch.Tensor:
        """Engine._attention with GLM's rope and DSA. Shapes and kernels match the dense engine's for
        every query that keeps all its keys, so a group without long rows runs the dense path."""
        c = self.c
        T = x.shape[0]
        nh, dn, dr, dv = c.num_attention_heads, c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim
        qa = rms_norm(linear(x, w["self_attn.q_a_proj.weight"], TILE_ATTN), w["self_attn.q_a_layernorm.weight"], 1e-6)
        q = linear(qa, w["self_attn.q_b_proj.weight"], TILE_ATTN).view(1, T, nh, dn + dr).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [dn, dr], dim=-1)
        ckv = linear(x, w["self_attn.kv_a_proj_with_mqa.weight"], TILE_ATTN)
        ckv, k_pe = torch.split(ckv, [c.kv_lora_rank, dr], dim=-1)
        k_pe = k_pe.reshape(1, T, 1, dr).transpose(1, 2)
        kv = linear(rms_norm(ckv, w["self_attn.kv_a_layernorm.weight"], 1e-6), w["self_attn.kv_b_proj.weight"], TILE_ATTN)
        kv = kv.view(1, T, nh, dn + dv).transpose(1, 2)
        k_nope, v = torch.split(kv, [dn, dv], dim=-1)
        cos, sin = rope_cos_sin(self.inv_freq, g.positions)
        q_pe = rope_interleave(q_pe, cos, sin)
        k_pe = rope_interleave(k_pe, cos, sin)
        qs = torch.cat([q_nope, q_pe], dim=-1)[0]  # (nh, T, d)
        ks = torch.cat([k_nope, k_pe.expand(1, nh, T, dr)], dim=-1)[0]
        vs = v[0]
        out = torch.empty(nh, T, dv, dtype=BF16, device=x.device)
        Lp = g.prefix_len
        layer = w["layer"]
        full = c.indexer_types[layer] == "full"
        long_suffix = Lp + max(g.suffix_lens, default=0) > self.topk
        idx_qkw = self._index_qkw(w, x, qa, cos, sin) if full and (Lp > self.topk or long_suffix) else None

        # prefix: causal attention; queries past position topk keep only their selected keys
        if Lp:
            if Lp > self.topk:
                key = (g.start, "prefix")
                if full:
                    iq, ik, iw = idx_qkw
                    rows = torch.arange(self.topk, Lp, device=x.device)
                    allow = torch.arange(Lp, device=x.device)[None, :] <= rows[:, None]  # (R, Lp)
                    self._dsa[key] = self._select(iq[rows], iw[rows], ik[None, :Lp], allow)
                sel = self._dsa[key]
                mask = torch.ones(Lp, Lp, dtype=torch.bool, device=x.device).tril()
                mask[self.topk:] = sel
                if layer == 0:
                    self.dsa_rows += Lp - self.topk
                out[:, :Lp] = F.scaled_dot_product_attention(
                    qs[None, :, :Lp], ks[None, :, :Lp], vs[None, :, :Lp], attn_mask=mask[None, None],
                    scale=self.softmax_scale)[0]
            else:
                out[:, :Lp] = F.scaled_dot_product_attention(
                    qs[None, :, :Lp], ks[None, :, :Lp], vs[None, :, :Lp], is_causal=True, scale=self.softmax_scale
                )[0]
        # suffixes: all prefix keys plus their own causal keys, cut by DSA where a query has more than topk
        for n_chunk, (idx, valid, mask) in enumerate(g.suffix_chunks(self.suffix_chunk)):
            m, Ls = idx.shape
            safe = idx.clamp(min=0)
            qi = qs[:, safe].permute(1, 0, 2, 3)  # (m, nh, Ls, d)
            ki = ks[:, safe].permute(1, 0, 2, 3)
            vi = vs[:, safe].permute(1, 0, 2, 3)
            allow = ~mask[:, None]  # (m, 1, Ls, Ls)
            if Lp:
                ki = torch.cat([ks[None, :, :Lp].expand(m, -1, -1, -1), ki], dim=2)
                vi = torch.cat([vs[None, :, :Lp].expand(m, -1, -1, -1), vi], dim=2)
                allow = torch.cat([allow.new_ones(m, 1, Ls, Lp), allow], dim=-1)
            # a query at suffix offset j sees Lp + j + 1 keys; only those past topk need the indexer
            if Lp + Ls > self.topk and bool(((Lp + torch.arange(Ls, device=x.device) + 1 > self.topk)[None, :] & valid).any()):
                key = (g.start, n_chunk)
                if full:
                    iq, ik, iw = idx_qkw
                    keys = torch.cat([ik[None, :Lp].expand(m, -1, -1), ik[safe]], dim=1)  # (m, Lp + Ls, D)
                    self._dsa[key] = self._select(iq[safe], iw[safe], keys[:, None], allow[:, 0])
                sel = self._dsa[key]  # (m, Ls, Lp + Ls)
                if layer == 0:
                    self.dsa_rows += int(((sel.sum(-1) < allow[:, 0].sum(-1)) & valid).sum())
                allow = allow & sel[:, None]
            o = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=allow, scale=self.softmax_scale)
            out[:, idx[valid]] = o.permute(1, 0, 2, 3)[:, valid]
        o = out.transpose(0, 1).reshape(T, nh * dv)
        return linear(o, w["self_attn.o_proj.weight"], TILE_ATTN)


# ---------------------------------------------------------------- rendering

# Reasoning effort written into GLM-5.3's system header for every render. The template has no
# off switch; "low" is what Z.ai's migration notes give for clients that used to disable thinking,
# and the closest header to an empty think block. A PREREG amendment item, see the module docstring.
READOUT_EFFORT = "low"
THINK, THINK_END = "<think>", "</think>"


class GlmRenderer(Renderer):
    """GLM-5.3's chat template with the think block closed before the answer.

    The template renders every earlier assistant turn as `<|assistant|><think></think>content`
    and ends the generation prompt in `<|assistant|><think>`, with no switch to turn thinking
    off. A first-token readout needs the answer to be the next token, so a render with a
    generation prompt gets `</think>` appended, giving `<|assistant|><think></think>`: the token
    pattern of K2.6's `thinking=False` generation prompt, and the form GLM's own template gives
    every non-final assistant turn. Renders without a generation prompt are untouched (the
    template is prefix-stable: an assistant turn renders the same whether or not it is last).
    """

    def __init__(self, tok, reasoning_effort: str | None = READOUT_EFFORT, close_think: bool = True):  # noqa: D107
        super().__init__(tok, **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}))
        self.close_think = close_think
        self.think_id = tok.convert_tokens_to_ids(THINK)
        self.think_end_id = tok.convert_tokens_to_ids(THINK_END)

    def text(self, messages: list[dict], gen: bool = False) -> str:
        t = super().text(messages, gen)
        if gen and self.close_think:
            assert t.endswith(THINK), f"generation prompt does not end in {THINK}: {t[-60:]!r}"
            t += THINK_END
        return t

    def reference_ids(self, messages: list[dict], gen: bool) -> list[int]:
        ids = super().reference_ids(messages, gen)
        if gen and self.close_think:
            assert ids[-1] == self.think_id, ids[-5:]
            ids = ids + [self.think_end_id]
        return ids


# ---------------------------------------------------------------- the confirmatory guard


# The GLM amendment's OpenTimestamps proof, made the way prereg/manifest.txt.ots was. Until it exists
# GLM sees no confirmatory task (drafts/prereg-amendment-glm.md, "Registration").
AMENDMENT_STAMP = "prereg/manifest-glm.txt.ots"


def assert_not_confirmatory(tasks: list[dict]) -> None:
    """GLM may not see the confirmatory tasks (c01-c40), by id or by text, until its PREREG
    amendment is timestamped."""
    from .texts import ROOT, load_jsonl

    if (ROOT / AMENDMENT_STAMP).exists():
        return
    conf = load_jsonl(ROOT / "prompts/tasks/confirmatory_tasks.jsonl")
    reserved, reserved_text = {t["id"] for t in conf}, {t["task"] for t in conf}
    hit = [t["id"] for t in tasks if t["id"] in reserved or t["task"] in reserved_text]
    assert not hit, (f"confirmatory tasks {hit} are reserved: GLM sees none until its PREREG amendment "
                     f"is timestamped ({AMENDMENT_STAMP} missing; drafts/prereg-amendment-glm.md)")
