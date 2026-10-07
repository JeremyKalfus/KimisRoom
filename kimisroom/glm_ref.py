"""transformers' own GLM-5.3 modules, one decoder layer at a time: the reference the engine port is
gated on (tests/glm_layer_parity.py, tests/glm_chain.py). Needs transformers >= 5.6 (`glm_moe_dsa`
with `indexer_types`); on the boxes that is the overlay in scripts/glm_box.sh, since the main venv
keeps 4.57.1 for the Qwen backend.

A reference layer is `GlmMoeDsaDecoderLayer` with transformers' own attention (SDPA), indexer,
router, experts ("eager", the per-expert loop) and norms. Its weights come from the same FP8
checkpoint, dequantized to the reference dtype:
- bf16: the fp8 value times its block scale in fp32, rounded once to bf16 (what the engine and
  transformers' Fp8Dequantize hold), with the fp32 router bias and fp32 indexer.weights_proj that
  transformers keeps (`_keep_in_fp32_modules`);
- fp32: the same product left in fp32, every other weight upcast, all math in fp32. It is the
  precision floor both bf16 implementations are measured against.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .glm import BLOCK, FP8, PROJ, GlmCheckpoint

F32, BF16 = torch.float32, torch.bfloat16


def hf_config(path: str | Path, attn: str = "sdpa", **override):
    from transformers import GlmMoeDsaConfig

    raw = json.loads((Path(path) / "config.json").read_text())
    raw = {k: v for k, v in raw.items() if k not in ("quantization_config", "transformers_version", "architectures")}
    cfg = GlmMoeDsaConfig(**{**raw, **override}, attn_implementation=attn, experts_implementation="eager")
    # any other attention implementation passes the top-k to a kernel instead of masking: dense under SDPA
    assert cfg._attn_implementation in ("eager", "sdpa"), cfg._attn_implementation
    return cfg


def dequant(w: torch.Tensor, s: torch.Tensor, dtype: torch.dtype, device) -> torch.Tensor:
    """Block dequant on `device`, written independently of glm.dequant_fp8 (explicit block loop)."""
    w = w.to(device).view(FP8).to(F32)
    s = s.to(device).to(F32)
    out = torch.empty(w.shape, dtype=F32, device=device)
    R, C = w.shape
    for bi in range(s.shape[0]):
        for bj in range(s.shape[1]):
            r0, c0 = bi * BLOCK, bj * BLOCK
            out[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = w[r0 : r0 + BLOCK, c0 : c0 + BLOCK] * s[bi, bj]
    return out.to(dtype)


def dequant_fast(w: torch.Tensor, s: torch.Tensor, dtype: torch.dtype, device) -> torch.Tensor:
    """The same product for whole-block matrices, vectorized (experts; checked equal to `dequant`)."""
    R, C = w.shape
    x = w.to(device).view(FP8).to(F32).view(R // BLOCK, BLOCK, C // BLOCK, BLOCK)
    return (x * s.to(device).to(F32).view(R // BLOCK, 1, C // BLOCK, 1)).view(R, C).to(dtype)


@torch.no_grad()
def build_layer(cfg, ck: GlmCheckpoint, got: dict[str, torch.Tensor], i: int, dtype: torch.dtype, device,
                expert_device=None):
    """GlmMoeDsaDecoderLayer i with the checkpoint's weights in `dtype` on `device`. With `expert_device`,
    the routed experts' weights live there instead and their inputs and output are moved across: the same
    ops on another device of the same kind, for GPUs too small for a whole bf16 MoE layer (24 GB L4)."""
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M

    with torch.device("meta"):
        layer = M.GlmMoeDsaDecoderLayer(cfg, i)
    plain, lin = ck.layer_names(i)
    sd: dict[str, torch.Tensor] = {}
    for n in plain:
        t = got[n].to(device)
        keep32 = n.endswith("e_score_correction_bias") or n.endswith("indexer.weights_proj.weight")
        sd[n] = t.to(F32 if (keep32 or dtype == F32) else dtype)
    for n in lin:
        sd[f"{n}.weight"] = dequant(got[f"{n}.weight"], got[f"{n}.weight_scale_inv"], dtype, device)
    if ck.is_moe(i):
        E, I, H = ck.config.n_routed_experts, ck.config.moe_intermediate_size, ck.config.hidden_size
        edev = device if expert_device is None else expert_device
        gu = torch.empty(E, 2 * I, H, dtype=dtype, device=edev)
        dn = torch.empty(E, H, I, dtype=dtype, device=edev)
        for e in range(E):
            q = {m: dequant_fast(got[f"mlp.experts.{e}.{m}.weight"], got[f"mlp.experts.{e}.{m}.weight_scale_inv"],
                                 dtype, edev) for m in PROJ}
            gu[e, :I], gu[e, I:] = q["gate_proj"], q["up_proj"]
            dn[e] = q["down_proj"]
        sd["mlp.experts.gate_up_proj"], sd["mlp.experts.down_proj"] = gu, dn
    missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected, unexpected
    assert not [m for m in missing if not m.endswith("inv_freq")], missing
    if ck.is_moe(i) and expert_device is not None and torch.device(expert_device) != torch.device(device):
        experts, fwd, edev = layer.mlp.experts, layer.mlp.experts.forward, torch.device(expert_device)

        def across(*args, **kw):
            mv = lambda t: t.to(edev) if torch.is_tensor(t) else t  # noqa: E731
            out = fwd(*[mv(x) for x in args], **{k: mv(v) for k, v in kw.items()})
            return out.to(device)

        experts.forward = across
    layer.eval()
    return layer


def rotary(cfg, device):
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M

    return M.GlmMoeDsaRotaryEmbedding(cfg).to(device)


@torch.no_grad()
def run_layer(layer, rot, h: torch.Tensor, prev_topk: torch.Tensor | None):
    """One reference layer on one sequence h (1, S, H): causal bool mask, positions 0..S-1.
    Returns (h_out, topk_indices)."""
    S = h.shape[1]
    pos = torch.arange(S, device=h.device)[None]
    cos, sin = rot(h, pos)
    mask = torch.ones(S, S, dtype=torch.bool, device=h.device).tril()[None, None]
    out, topk = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=(cos, sin),
                      prev_topk_indices=prev_topk)
    return out, topk


@torch.no_grad()
def final_logprobs(h: torch.Tensor, norm_w: torch.Tensor, head_w: torch.Tensor, eps: float, dtype: torch.dtype):
    """transformers' final RMSNorm then lm_head, as fp32 log-probabilities (the engine's readout)."""
    x = h.to(F32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    x = norm_w.to(dtype) * x.to(dtype)
    return F.log_softmax(F.linear(x, head_w.to(dtype)).to(F32), dim=-1)


class LayerwiseRef:
    """transformers' GLM-5.3 run one decoder layer at a time over a list of sequences (each its own
    batch of one, so nothing is padded): build layer i from the checkpoint, run every sequence
    through it, drop it. This is GlmMoeDsaModel.forward's loop (causal mask, rotary embedding,
    top-k handed from each layer to the next), so a 753B model needs one layer of memory per
    dtype. Steering adds a vector to a layer's output at given positions, as the engine does."""

    def __init__(self, cfg, ck: GlmCheckpoint, dtype: torch.dtype, device):
        self.cfg, self.ck, self.dtype, self.device = cfg, ck, dtype, torch.device(device)
        self.rot = rotary(cfg, self.device)
        self.hs: list[torch.Tensor] = []
        self.topk: list[torch.Tensor | None] = []

    @torch.no_grad()
    def embed(self, seqs: list[list[int]], embed_w: torch.Tensor) -> None:
        e = embed_w.to(self.device)
        self.hs = [F.embedding(torch.tensor(s, device=self.device), e).to(self.dtype)[None] for s in seqs]
        self.topk = [None] * len(seqs)

    @torch.no_grad()
    def step(self, i: int, got: dict[str, torch.Tensor], inputs: list[torch.Tensor] | None = None) -> list[torch.Tensor]:
        """Layer i on every sequence. `inputs` replaces the running states (isolated per-layer
        errors: someone else's states in, this layer's outputs back, the running states kept);
        the top-k selection is handed on either way. Returns the outputs."""
        layer = build_layer(self.cfg, self.ck, got, i, self.dtype, self.device)
        outs, tks = [], []
        for h, tk in zip(inputs if inputs is not None else self.hs, self.topk):
            o, t = run_layer(layer, self.rot, h.to(self.device, self.dtype), tk)
            outs.append(o)
            tks.append(t)
        del layer
        self.topk = tks
        if inputs is None:
            self.hs = outs
        return outs

    @torch.no_grad()
    def steer(self, seq: int, positions: list[int], vec: torch.Tensor, coef: float) -> None:
        h = self.hs[seq]
        v = vec.to(self.device, F32) * coef
        v = v.to(BF16) if self.dtype == BF16 else v  # the engine adds the bf16-rounded vector
        h[0, positions] = (h[0, positions].to(v.dtype) + v).to(self.dtype)


def model_state_dict(cfg, ck: GlmCheckpoint, dtype: torch.dtype, device="cpu") -> dict[str, torch.Tensor]:
    """A whole GlmMoeDsaForCausalLM state dict (small checkpoints only: tests/glm_tiny_check.py)."""
    sd = {n: ck.load_global(n).to(device).to(dtype) for n in ("model.embed_tokens.weight", "model.norm.weight",
                                                              "lm_head.weight")}
    for i in range(cfg.num_hidden_layers):
        layer = build_layer(cfg, ck, ck.read_layer(i), i, dtype, device)
        sd.update({f"model.layers.{i}.{k}": v for k, v in layer.state_dict().items()})
    return sd
