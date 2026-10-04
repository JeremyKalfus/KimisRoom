"""Layer-by-layer prefill for DeepseekV3-family checkpoints (Moonlight, Kimi K2.6).

Each pass embeds a batch of token groups, then streams the model one decoder
layer at a time: the layer's weights come off disk, every resident token goes
through it, and the weights are dropped before the next layer loads. GPU memory
holds one layer plus the residual stream, so a 600 GB model runs on one 48 GB
card.

A group is one prefix plus any number of suffixes. Each suffix attends to the
whole prefix and causally to itself, never to other suffixes, so a battery of
questions asked after one (steered) conversation pays for the prefix once.

The math mirrors the checkpoint's own `modeling_deepseek.py` (loaded through
transformers' remote-code loader): the router, rotary embedding and RoPE
application are the reference objects; norms, MLA attention and the expert
combine are re-implemented op for op, including bf16/fp32 casts.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open

BF16 = torch.bfloat16
F32 = torch.float32
_ST_DTYPES = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64,
    "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}


# ---------------------------------------------------------------- job spec


@dataclass
class Steer:
    """Add coef * vectors[vector] to the residual stream after decoder layer `layer`.

    `prefix` is a [start, end) token range inside the prefix; `suffix` is a
    [start, end) range inside every suffix (end=None means to the suffix end).
    """

    layer: int
    vector: str
    coef: float
    prefix: tuple[int, int] | None = None
    suffix: tuple[int, int | None] | None = None


@dataclass
class Collect:
    """Mean of the residual stream after decoder layer `layer` over prefix tokens [start, end)."""

    layer: int
    start: int
    end: int


@dataclass
class Group:
    id: str
    prefix: list[int]
    suffixes: list[list[int]] = field(default_factory=list)
    steers: list[Steer] = field(default_factory=list)
    collects: list[Collect] = field(default_factory=list)
    # "suffix_last": next-token distribution after each suffix; "prefix_last": after the
    # prefix; "prefix_all": at every prefix position, with the logprob of the actual next
    # prefix token (teacher-forced perplexity); "none": no readout.
    readout: str = "suffix_last"

    @property
    def n_tokens(self) -> int:
        return len(self.prefix) + sum(len(s) for s in self.suffixes)


@dataclass
class Readout:
    top_ids: torch.Tensor  # (k,) int64
    top_logprobs: torch.Tensor  # (k,) fp32
    target_logprobs: torch.Tensor  # (n_targets,) fp32
    next_logprob: float | None = None  # "prefix_all" only: logprob of the token that actually follows


# ---------------------------------------------------------------- checkpoint


class Checkpoint:
    """Lazy access to a local HF snapshot of a DeepseekV3-family model."""

    def __init__(self, path: str | os.PathLike):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        self.path = Path(path)
        raw = json.loads((self.path / "config.json").read_text())
        text = raw.get("text_config", raw)
        self.prefix = "language_model." if "text_config" in raw else ""

        def remote(name):
            return get_class_from_dynamic_module(f"modeling_deepseek.{name}", str(self.path))

        config_cls = get_class_from_dynamic_module("configuration_deepseek.DeepseekV3Config", str(self.path))
        self.config = config_cls(**{k: v for k, v in text.items() if k != "quantization_config"})
        self.MoEGate = remote("MoEGate")
        self.apply_rotary_pos_emb = remote("apply_rotary_pos_emb")
        self.Attention = remote("DeepseekV3Attention")
        quant = text.get("quantization_config") or raw.get("quantization_config")
        self.int4_group = None
        if quant:
            w = quant["config_groups"]["group_0"]["weights"]
            assert quant["format"] == "pack-quantized" and w["num_bits"] == 4 and w["symmetric"]
            self.int4_group = w["group_size"]

        index = json.loads((self.path / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._handles: dict[str, object] = {}
        self._lock = threading.Lock()
        # Shards holding one layer only (all of K2.6's) are read whole with parallel
        # sequential reads: mmap'ing ~3,500 small tensors out of them page-faults at a
        # fraction of what the disk can stream.
        self._file_layers: dict[str, set] = {}
        for k, f in self.weight_map.items():
            parts = k[len(self.prefix) :].split(".")
            layer = int(parts[2]) if parts[:2] == ["model", "layers"] else None
            self._file_layers.setdefault(f, set()).add(layer)
        self._io = ThreadPoolExecutor(8)

    def _read_file(self, fname: str) -> dict[str, torch.Tensor]:
        path = self.path / fname
        size = path.stat().st_size
        buf = torch.empty(size, dtype=torch.uint8)
        mv = memoryview(buf.numpy())
        fd = os.open(path, os.O_RDONLY)
        chunk = 64 << 20

        def read(off: int) -> None:
            end = min(off + chunk, size)
            while off < end:
                off += os.preadv(fd, [mv[off:end]], off)

        try:
            list(self._io.map(read, range(0, size, chunk)))
        finally:
            os.close(fd)
        n = int.from_bytes(bytes(mv[:8]), "little")
        header = json.loads(bytes(mv[8 : 8 + n]))
        base = 8 + n
        out = {}
        for k, meta in header.items():
            if k == "__metadata__":
                continue
            a, b = (base + o for o in meta["data_offsets"])
            dt = _ST_DTYPES[meta["dtype"]]
            raw = buf[a:b] if a % dt.itemsize == 0 else buf[a:b].clone()
            out[k] = raw.view(dt).view(meta["shape"])
        return out

    def _get(self, key: str) -> torch.Tensor:
        fname = self.weight_map[self.prefix + key]
        with self._lock:
            h = self._handles.get(fname)
            if h is None:
                h = safe_open(self.path / fname, framework="pt", device="cpu")
                self._handles[fname] = h
            return h.get_tensor(self.prefix + key)

    def is_moe(self, i: int) -> bool:
        c = self.config
        return c.n_routed_experts is not None and i >= c.first_k_dense_replace and i % c.moe_layer_freq == 0

    def load_layer_cpu(self, i: int, shared: bool = False) -> dict[str, torch.Tensor]:
        """All tensors of decoder layer i in CPU memory, experts stacked along dim 0."""
        c = self.config
        p = f"model.layers.{i}."
        names = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight",
        ]
        if c.q_lora_rank is None:
            names.append("self_attn.q_proj.weight")
        else:
            names += ["self_attn.q_a_proj.weight", "self_attn.q_a_layernorm.weight", "self_attn.q_b_proj.weight"]
        moe = self.is_moe(i)
        if moe:
            names += ["mlp.gate.weight", "mlp.gate.e_score_correction_bias"]
            if c.n_shared_experts is not None:
                names += [f"mlp.shared_experts.{m}.weight" for m in ("gate_proj", "up_proj", "down_proj")]
        else:
            names += [f"mlp.{m}.weight" for m in ("gate_proj", "up_proj", "down_proj")]
        whole: dict[str, torch.Tensor] = {}
        for f in {f for k, f in self.weight_map.items() if k.startswith(self.prefix + p)}:
            if self._file_layers[f] == {i}:
                whole.update(self._read_file(f))

        def get(key: str) -> torch.Tensor:
            t = whole.get(self.prefix + key)
            return self._get(key) if t is None else t

        def empty(shape, dtype: torch.dtype) -> torch.Tensor:  # allocated in shared memory if asked
            if not shared:
                return torch.empty(shape, dtype=dtype)
            st = torch.UntypedStorage._new_shared(math.prod(shape) * dtype.itemsize)
            return torch.empty(0, dtype=dtype).set_(st, 0, shape)

        def own(x: torch.Tensor) -> torch.Tensor:  # a private copy
            return empty(tuple(x.shape), x.dtype).copy_(x)

        def stack(xs: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(xs, out=empty((len(xs), *xs[0].shape), xs[0].dtype))

        t = {n: own(get(p + n)) for n in names}
        if moe:
            for m in ("gate_proj", "up_proj", "down_proj"):
                base = [f"{p}mlp.experts.{e}.{m}." for e in range(c.n_routed_experts)]
                if self.int4_group:
                    t[f"experts.{m}.packed"] = stack([get(b + "weight_packed") for b in base])
                    t[f"experts.{m}.scale"] = stack([get(b + "weight_scale") for b in base])
                    t[f"experts.{m}.shape"] = own(get(base[0] + "weight_shape"))
                else:
                    t[f"experts.{m}.weight"] = stack([get(b + "weight") for b in base])
        return t

    def load_global(self, name: str) -> torch.Tensor:
        return self._get(name)


# ---------------------------------------------------------------- int4


def unpack_int4(packed: torch.Tensor, cols: int) -> torch.Tensor:
    """compressed-tensors pack-quantized int4 -> int8 in [-8, 7]; column c is word c//8, nibble c%8."""
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    q = (packed.unsqueeze(-1) >> shifts) & 0xF
    q = q.reshape(*packed.shape[:-1], -1)[..., :cols]
    return (q - 8).to(torch.int8)


def dequant_int4(packed: torch.Tensor, scale: torch.Tensor, rows: int, cols: int, group: int) -> torch.Tensor:
    q = unpack_int4(packed, cols)[:rows]
    w = q.to(scale.dtype).view(rows, cols // group, group) * scale[:rows].unsqueeze(-1)
    return w.view(rows, cols).to(BF16)


# One fused kernel instead of ~6 full-size elementwise passes per expert; integer unpacking
# and the bf16 scale multiply round the same way, so it is bit-identical to dequant_int4.
dequant_int4_fused = torch.compile(dequant_int4, dynamic=False, fullgraph=True)


# ---------------------------------------------------------------- math


# Batch invariance: a GEMM or row reduction can pick a different kernel (and so a
# different summation order) when the number of rows changes, which makes a token's
# result depend on what else is in the batch. Every row-wise op below therefore runs
# on fixed-size row tiles, padding the last tile, so each token sees the same kernels
# whatever it is batched with. Tile sizes are fixed per op, never chosen from data.
TILE_ATTN, TILE_MLP, TILE_EXPERT, TILE_ROUTER, TILE_NORM, TILE_HEAD = 512, 1024, 256, 1024, 1024, 64
SUFFIX_BUCKET = 32


def tiled(fn, x: torch.Tensor, tile: int) -> torch.Tensor:
    n = x.shape[0]
    if n == 0:
        return fn(x)
    pad = (-n) % tile
    if pad:
        x = torch.cat([x, x.new_zeros(pad, *x.shape[1:])])
    return torch.cat([fn(x[a : a + tile]) for a in range(0, x.shape[0], tile)])[:n]


def linear(x: torch.Tensor, w: torch.Tensor, tile: int) -> torch.Tensor:
    return tiled(lambda t: F.linear(t, w), x, tile)


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x32 = x.to(F32)
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return w * x32.to(dtype)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return tiled(lambda t: _rms(t, w, eps), x, TILE_NORM)


def mlp(x: torch.Tensor, g: torch.Tensor, u: torch.Tensor, d: torch.Tensor, tile: int) -> torch.Tensor:
    return tiled(lambda t: F.linear(F.silu(F.linear(t, g)) * F.linear(t, u), d), x, tile)


class Engine:
    def __init__(
        self,
        ckpt: Checkpoint,
        device: str | torch.device = "cuda",
        moe_chunk: int = 65536,
        suffix_chunk: int = 16,
        router_bias_dtype: torch.dtype | None = None,
    ):
        self.ckpt = ckpt
        self.c = ckpt.config
        self.device = torch.device(device)
        self.moe_chunk = moe_chunk
        self.suffix_chunk = suffix_chunk
        # None keeps the checkpoint dtype; BF16 reproduces HF from_pretrained(torch_dtype=bf16).
        self.router_bias_dtype = router_bias_dtype
        c = self.c
        self.q_head_dim = c.qk_nope_head_dim + c.qk_rope_head_dim
        self.softmax_scale = self.q_head_dim ** (-0.5)
        if c.rope_scaling is not None:
            mscale_all_dim = c.rope_scaling.get("mscale_all_dim", 0)
            if mscale_all_dim:
                factor = c.rope_scaling["factor"]
                m = 0.1 * mscale_all_dim * math.log(factor) + 1.0 if factor > 1 else 1.0
                self.softmax_scale = self.softmax_scale * m * m
        ns = SimpleNamespace(
            config=c,
            qk_rope_head_dim=c.qk_rope_head_dim,
            max_position_embeddings=c.max_position_embeddings,
            rope_theta=c.rope_theta,
        )
        with torch.device("cpu"):
            ckpt.Attention._init_rope(ns)
        self.rotary = ns.rotary_emb.to(self.device)
        self._pool = ThreadPoolExecutor(max_workers=1)

    # ---------------- weights

    def _to_device(self, cpu: dict[str, torch.Tensor]) -> dict:
        dev = self.device
        w = {k: v.to(dev) for k, v in cpu.items() if not k.endswith(".shape")}
        if "mlp.gate.weight" in w:
            w["gate.weight"] = w.pop("mlp.gate.weight")
            bias = w.pop("mlp.gate.e_score_correction_bias")
            # The reference adds the bias to fp32 scores; a bf16 bias (HF's cast) is rounded first.
            w["gate.bias"] = bias.to(self.router_bias_dtype or bias.dtype).to(F32)
            if self.ckpt.int4_group:
                for m in ("gate_proj", "up_proj", "down_proj"):
                    rows, cols = (int(x) for x in cpu[f"experts.{m}.shape"])
                    w[f"experts.{m}.dims"] = (rows, cols)
        return w

    def _expert(self, w: dict, e: int):
        out = []
        for m in ("gate_proj", "up_proj", "down_proj"):
            if self.ckpt.int4_group:
                rows, cols = w[f"experts.{m}.dims"]
                packed, scale = w[f"experts.{m}.packed"][e], w[f"experts.{m}.scale"][e]
                out.append(dequant_int4_fused(packed, scale, rows, cols, self.ckpt.int4_group))
            else:
                out.append(w[f"experts.{m}.weight"][e])
        return out

    # ---------------- layer pieces

    def _attention(self, w: dict, x: torch.Tensor, g: "_GroupLayout") -> torch.Tensor:
        """MLA over one group's tokens x (T, H), already input-normed. Returns o_proj output (T, H)."""
        c = self.c
        T = x.shape[0]
        nh, dn, dr, dv = c.num_attention_heads, c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim
        if c.q_lora_rank is None:
            q = linear(x, w["self_attn.q_proj.weight"], TILE_ATTN)
        else:
            qa = rms_norm(linear(x, w["self_attn.q_a_proj.weight"], TILE_ATTN), w["self_attn.q_a_layernorm.weight"], 1e-6)
            q = linear(qa, w["self_attn.q_b_proj.weight"], TILE_ATTN)
        q = q.view(1, T, nh, dn + dr).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [dn, dr], dim=-1)
        ckv = linear(x, w["self_attn.kv_a_proj_with_mqa.weight"], TILE_ATTN)
        ckv, k_pe = torch.split(ckv, [c.kv_lora_rank, dr], dim=-1)
        k_pe = k_pe.view(1, T, 1, dr).transpose(1, 2)
        kv = linear(rms_norm(ckv, w["self_attn.kv_a_layernorm.weight"], 1e-6), w["self_attn.kv_b_proj.weight"], TILE_ATTN)
        kv = kv.view(1, T, nh, dn + dv).transpose(1, 2)
        k_nope, v = torch.split(kv, [dn, dv], dim=-1)
        cos, sin = self.rotary(v, seq_len=g.max_pos + 1)
        q_pe, k_pe = self.ckpt.apply_rotary_pos_emb(q_pe, k_pe, cos, sin, g.positions.unsqueeze(0))
        qs = torch.cat([q_nope, q_pe], dim=-1)[0]  # (nh, T, d)
        ks = torch.cat([k_nope, k_pe.expand(1, nh, T, dr)], dim=-1)[0]
        vs = v[0]
        out = torch.empty(nh, T, dv, dtype=BF16, device=x.device)
        Lp = g.prefix_len
        # Fused attention (fp32 accumulation). Each call holds one group's prefix or one
        # fixed-shape suffix chunk, so no other group or chunk-mate can change its kernel.
        if Lp:  # prefix: plain causal attention
            out[:, :Lp] = F.scaled_dot_product_attention(
                qs[None, :, :Lp], ks[None, :, :Lp], vs[None, :, :Lp], is_causal=True, scale=self.softmax_scale
            )[0]
        # suffixes: all prefix keys plus their own causal keys
        for idx, valid, mask in g.suffix_chunks(self.suffix_chunk):
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
            o = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=allow, scale=self.softmax_scale)
            out[:, idx[valid]] = o.permute(1, 0, 2, 3)[:, valid]
        o = out.transpose(0, 1).reshape(T, nh * dv)
        return linear(o, w["self_attn.o_proj.weight"], TILE_ATTN)

    def _route(self, w: dict, x: torch.Tensor):
        """The checkpoint's MoEGate (sigmoid scores, noaux_tc top-k with the correction
        bias, normalized, scaled), with experts sorted by id per token and the
        normalizing sum taken in that fixed order."""
        c = self.c
        assert c.n_group == 1 and c.topk_group == 1 and c.scoring_func == "sigmoid" and c.topk_method == "noaux_tc"
        logits = linear(x.to(F32), w["gate.weight"].to(F32), TILE_ROUTER)
        scores = logits.sigmoid()
        choice = scores + w["gate.bias"].unsqueeze(0)
        idx = torch.topk(choice, k=c.num_experts_per_tok, dim=-1, sorted=False)[1]
        idx = idx.sort(dim=-1).values
        weight = scores.gather(1, idx)
        if c.num_experts_per_tok > 1 and c.norm_topk_prob:
            denom = weight[:, 0].clone()
            for j in range(1, weight.shape[1]):
                denom = denom + weight[:, j]
            weight = weight / (denom + 1e-20).unsqueeze(-1)
        return idx, weight * c.routed_scaling_factor

    def _ffn(self, w: dict, x: torch.Tensor) -> torch.Tensor:
        """Dense MLP or MoE on post-attention-normed tokens x (T, H). Each token's expert
        outputs are added in ascending expert order (no duplicate rows per index_add_)."""
        if "gate.weight" not in w:
            return mlp(x, w["mlp.gate_proj.weight"], w["mlp.up_proj.weight"], w["mlp.down_proj.weight"], TILE_MLP)
        T, H = x.shape
        k = self.c.num_experts_per_tok
        topk_idx, topk_w = self._route(w, x)
        flat = topk_idx.reshape(-1)
        order = flat.argsort(stable=True)
        counts = torch.bincount(flat, minlength=self.c.n_routed_experts).tolist()
        token_of = order // k
        w_sorted = topk_w.reshape(-1)[order]
        y = torch.zeros(T, H, dtype=F32, device=x.device)
        start = 0
        for e, n in enumerate(counts):
            if n == 0:
                continue
            idx = token_of[start : start + n]
            g, u, d = self._expert(w, e)
            out = mlp(x[idx], g, u, d, TILE_EXPERT)
            y.index_add_(0, idx, out.to(F32) * w_sorted[start : start + n, None])
            start += n
        y = y.to(BF16)
        if "mlp.shared_experts.gate_proj.weight" in w:
            y = y + mlp(
                x,
                w["mlp.shared_experts.gate_proj.weight"],
                w["mlp.shared_experts.up_proj.weight"],
                w["mlp.shared_experts.down_proj.weight"],
                TILE_MLP,
            )
        return y

    def layer(self, w: dict, h: torch.Tensor, layout: "_Layout") -> None:
        """Apply one decoder layer to the residual stream h (N, H) in place."""
        eps = self.c.rms_norm_eps
        for g in layout.groups:
            sl = slice(g.start, g.end)
            xn = rms_norm(h[sl], w["input_layernorm.weight"], eps)
            h[sl] = h[sl] + self._attention(w, xn, g)
        for a in range(0, h.shape[0], self.moe_chunk):
            sl = slice(a, min(a + self.moe_chunk, h.shape[0]))
            xn = rms_norm(h[sl], w["post_attention_layernorm.weight"], eps)
            h[sl] = h[sl] + self._ffn(w, xn)

    # ---------------- pass

    @torch.no_grad()
    def run(
        self,
        groups: list[Group],
        vectors: dict[str, torch.Tensor] | None = None,
        targets: list[int] | None = None,
        top_k: int = 20,
        progress=None,
        layer_source=None,
    ) -> dict[str, dict]:
        """One pass over all layers for these groups; the caller sizes the batch to fit."""
        c, dev = self.c, self.device
        eps = c.rms_norm_eps
        self.timings: dict[str, float] = {}  # seconds per stage, summed over layers (read it in progress)
        vectors = {k: v.to(dev) for k, v in (vectors or {}).items()}
        layout = _Layout(groups, dev)
        embed = self.ckpt.load_global("model.embed_tokens.weight").to(dev)
        h = F.embedding(torch.tensor(layout.token_ids, device=dev), embed)
        del embed
        steer_at: dict[int, list] = {}
        collect_at: dict[int, list] = {}
        for gi, g in enumerate(groups):
            for s in g.steers:
                steer_at.setdefault(s.layer, []).append((gi, s))
            for k, col in enumerate(g.collects):
                collect_at.setdefault(col.layer, []).append((gi, k, col))
        collected = {g.id: [None] * len(g.collects) for g in groups}
        norms = {g.id: [None] * len(g.collects) for g in groups}  # mean per-token L2 norm over each span
        n = c.num_hidden_layers
        load = layer_source or self.ckpt.load_layer_cpu  # layer_source(i): layer i's CPU tensors, called in order
        nxt = self._pool.submit(load, 0)
        for i in range(n):
            t0 = time.perf_counter()
            cpu = nxt.result()
            t1 = time.perf_counter()
            if i + 1 < n:
                nxt = self._pool.submit(load, i + 1)
            w = self._to_device(cpu)
            del cpu
            torch.cuda.synchronize(dev)
            t2 = time.perf_counter()
            self.layer(w, h, layout)
            torch.cuda.synchronize(dev)
            t3 = time.perf_counter()
            for k, dt in (("wait_disk", t1 - t0), ("to_gpu", t2 - t1), ("compute", t3 - t2)):
                self.timings[k] = self.timings.get(k, 0.0) + dt
            del w
            for gi, s in steer_at.get(i, []):
                pos = layout.groups[gi].steer_positions(s)
                vec = (vectors[s.vector].to(F32) * s.coef).to(BF16)
                h[pos] = h[pos] + vec
            for gi, k, col in collect_at.get(i, []):
                g0 = layout.groups[gi].start
                span = h[g0 + col.start : g0 + col.end].to(F32)
                collected[groups[gi].id][k] = span.mean(0).cpu()
                norms[groups[gi].id][k] = span.norm(dim=-1).mean().item()
            if progress:
                progress(i)
        out: dict[str, dict] = {
            g.id: {"collects": collected[g.id], "collect_norms": norms[g.id], "readouts": []} for g in groups
        }
        read_pos, read_owner, read_next = layout.readout_positions(groups)
        if read_pos:
            norm_w = self.ckpt.load_global("model.norm.weight").to(dev)
            head = self.ckpt.load_global("lm_head.weight").to(dev)
            hr = rms_norm(h[torch.tensor(read_pos, device=dev)], norm_w, eps)
            tgt_idx = torch.tensor(targets or [], dtype=torch.long, device=dev)

            nxt = torch.tensor(read_next, device=dev)
            R, pad = hr.shape[0], (-hr.shape[0]) % TILE_HEAD  # fixed row tiles, as in tiled()
            hr = torch.cat([hr, hr.new_zeros(pad, hr.shape[1])])
            nxt = torch.cat([nxt, nxt.new_full((pad,), -1)])
            res = []
            for a in range(0, hr.shape[0], TILE_HEAD):
                lp = F.log_softmax(F.linear(hr[a : a + TILE_HEAD], head).to(F32), dim=-1)
                top = lp.topk(top_k, dim=-1)
                nl = lp.gather(1, nxt[a : a + TILE_HEAD].clamp(min=0)[:, None])
                res.append(torch.cat([top.indices.to(F32), top.values, lp[:, tgt_idx], nl], dim=-1).cpu())
            res = torch.cat(res)[:R]
            for r, gid in enumerate(read_owner):
                ids, vals, tgt, nl = res[r].split([top_k, top_k, len(tgt_idx), 1])
                nl = nl.item() if read_next[r] >= 0 else None
                out[gid]["readouts"].append(Readout(ids.long(), vals, tgt, nl))
        return out


# ---------------------------------------------------------------- layout


class _GroupLayout:
    def __init__(self, g: Group, start: int, device):
        self.start = start
        self.device = device
        self.prefix_len = len(g.prefix)
        self.suffix_lens = [len(s) for s in g.suffixes]
        self.suffix_starts = []
        pos = list(range(self.prefix_len))
        off = self.prefix_len
        for L in self.suffix_lens:
            self.suffix_starts.append(off)
            pos += range(self.prefix_len, self.prefix_len + L)
            off += L
        self.end = start + off
        self.positions = torch.tensor(pos, device=device)
        self.max_pos = max(pos)
        self._chunks = None

    def causal(self, L: int) -> torch.Tensor:
        return torch.ones(L, L, dtype=torch.bool, device=self.device).triu(1)

    def suffix_chunks(self, n: int):
        """(idx, valid, mask) per chunk of exactly n suffix rows. Suffixes are bucketed by
        their own length rounded up to SUFFIX_BUCKET and chunks never mix buckets, so a
        suffix's attention shapes depend only on itself, never on its chunk-mates. idx
        holds group-relative token indices padded with -1; mask is True where a query may
        not attend to an own key."""
        if self._chunks is None:
            self._chunks = []
            buckets: dict[int, list[int]] = {}
            for r, L in enumerate(self.suffix_lens):
                buckets.setdefault(-(-L // SUFFIX_BUCKET) * SUFFIX_BUCKET, []).append(r)
            for Ls, rows in sorted(buckets.items()):
                for a in range(0, len(rows), n):
                    idx = torch.full((n, Ls), -1, dtype=torch.long)
                    for i, r in enumerate(rows[a : a + n]):
                        s0, L = self.suffix_starts[r], self.suffix_lens[r]
                        idx[i, :L] = torch.arange(s0, s0 + L)
                    idx = idx.to(self.device)
                    valid = idx >= 0
                    mask = self.causal(Ls).unsqueeze(0) | ~valid.unsqueeze(1)
                    self._chunks.append((idx, valid, mask))
        return self._chunks

    def steer_positions(self, s: Steer) -> torch.Tensor:
        pos: list[int] = []
        if s.prefix is not None:
            pos += range(self.start + s.prefix[0], self.start + s.prefix[1])
        if s.suffix is not None:
            for s0, L in zip(self.suffix_starts, self.suffix_lens):
                a, b = s.suffix
                b = L if b is None else min(b, L)
                pos += range(self.start + s0 + a, self.start + s0 + b)
        return torch.tensor(pos, dtype=torch.long, device=self.device)


class _Layout:
    def __init__(self, groups: list[Group], device):
        self.groups: list[_GroupLayout] = []
        self.token_ids: list[int] = []
        start = 0
        for g in groups:
            gl = _GroupLayout(g, start, device)
            self.groups.append(gl)
            self.token_ids += g.prefix
            for s in g.suffixes:
                self.token_ids += s
            start = gl.end

    def readout_positions(self, groups: list[Group]):
        """Token positions to read, the group id owning each, and the id of the token that
        follows each position (-1 when not asked for)."""
        pos, owner, nxt = [], [], []
        for g, gl in zip(groups, self.groups):
            if g.readout == "suffix_last":
                for s0, L in zip(gl.suffix_starts, gl.suffix_lens):
                    pos.append(gl.start + s0 + L - 1)
                    owner.append(g.id)
                    nxt.append(-1)
            elif g.readout == "prefix_last":
                pos.append(gl.start + gl.prefix_len - 1)
                owner.append(g.id)
                nxt.append(-1)
            elif g.readout == "prefix_all":
                pos += range(gl.start, gl.start + gl.prefix_len - 1)
                owner += [g.id] * (gl.prefix_len - 1)
                nxt += g.prefix[1:]
            else:
                assert g.readout == "none", g.readout
        return pos, owner, nxt
