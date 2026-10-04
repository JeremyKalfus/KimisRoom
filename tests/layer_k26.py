"""One Kimi K2.6 decoder layer: engine vs the checkpoint's own DeepseekV3DecoderLayer
fed compressed-tensors' dequantized experts. Both are scored against the same reference
module run in fp32, so the engine's bf16 error can be read against the bf16 reference's. Needs only that layer's shard
(K2.6 stores one layer per shard) plus the config, index and modeling files.

    python tests/layer_k26.py /nvme/k26 1
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.engine import BF16, Checkpoint, Engine, Group, _Layout, dequant_int4, dequant_int4_fused, unpack_int4  # noqa: E402

torch.manual_seed(0)
path, L = sys.argv[1], int(sys.argv[2])
ck = Checkpoint(path)
eng = Engine(ck)
cpu = ck.load_layer_cpu(L)
H = ck.config.hidden_size

# 1. int4 unpack/dequant vs compressed-tensors
if ck.is_moe(L):
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    from compressed_tensors.quantization import QuantizationArgs
    from compressed_tensors.quantization.lifecycle.forward import dequantize

    args = QuantizationArgs(num_bits=4, type="int", symmetric=True, strategy="group", group_size=ck.int4_group)
    for m in ("gate_proj", "up_proj", "down_proj"):
        rows, cols = (int(x) for x in cpu[f"experts.{m}.shape"])
        for e in (0, 191, ck.config.n_routed_experts - 1):
            packed, scale = cpu[f"experts.{m}.packed"][e], cpu[f"experts.{m}.scale"][e]
            q_ref = unpack_from_int32(packed, 4, torch.Size([rows, cols]))
            same_q = torch.equal(unpack_int4(packed, cols), q_ref.to(torch.int8))
            w_ref = dequantize(q_ref, scale, None, args=args, dtype=BF16)
            w_eng = dequant_int4(packed.cuda(), scale.cuda(), rows, cols, ck.int4_group).cpu()
            w_fused = dequant_int4_fused(packed.cuda(), scale.cuda(), rows, cols, ck.int4_group).cpu()
            print(f"int4 {m:9s} expert {e:3d}: unpack equal {same_q}, dequant equal {torch.equal(w_ref, w_eng)}, "
                  f"fused equal {torch.equal(w_ref, w_fused)}")

# 2. one layer: engine (packed experts) vs reference module (dequantized experts), bf16 and fp32
Lp, suffix_lens = 240, [17, 33, 25, 40]
groups = [Group("a", [0] * 300, readout="none"), Group("b", [0] * Lp, [[0] * n for n in suffix_lens], readout="none")]
# small inputs keep the bf16 rounding of x + delta below the layer delta itself
x = (torch.randn(sum(g.n_tokens for g in groups), H) * 0.02).to(BF16).cuda()

w = eng._to_device(cpu)
h = x.clone()
eng.layer(w, h, _Layout(groups, eng.device))
h_eng = h.cpu()
del w, h
torch.cuda.empty_cache()

from transformers.dynamic_module_utils import get_class_from_dynamic_module  # noqa: E402

Layer = get_class_from_dynamic_module("modeling_deepseek.DeepseekV3DecoderLayer", path)
cfg = ck.config
cfg._attn_implementation = "eager"
packed = {k: v.cuda() for k, v in cpu.items() if k.startswith("experts.") and not k.endswith(".shape")}


class LazyExpert(torch.nn.Module):
    """A routed expert dequantized on use, so neither reference holds 384 dense experts."""

    def __init__(self, e: int):
        super().__init__()
        self.e = e

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        g, u, d = (
            dequant_int4(
                packed[f"experts.{m}.packed"][self.e], packed[f"experts.{m}.scale"][self.e],
                *(int(v) for v in cpu[f"experts.{m}.shape"]), ck.int4_group,
            ).to(t.dtype)
            for m in ("gate_proj", "up_proj", "down_proj")
        )
        return F.linear(F.silu(F.linear(t, g)) * F.linear(t, u), d)


def make_ref(dtype: torch.dtype | None):
    """dtype None keeps the checkpoint's stored dtypes (what the engine sees)."""
    with torch.device("meta"):
        ref = Layer(cfg, L)
    sd = {k: (v if dtype is None else v.to(dtype)) for k, v in cpu.items() if not k.startswith("experts.")}
    missing, unexpected = ref.load_state_dict(sd, assign=True, strict=False)
    assert not unexpected and all("rotary" in k or ".experts." in k for k in missing), (missing, unexpected)
    if ck.is_moe(L):
        ref.mlp.experts = torch.nn.ModuleList(LazyExpert(e) for e in range(cfg.n_routed_experts))
    ref.self_attn._init_rope()  # rotary buffers were built on meta
    return ref.cuda().eval()


refs = {"bf16": (make_ref(None), BF16), "fp32": (make_ref(torch.float32), torch.float32)}


@torch.no_grad()
def ref_layer(seq: torch.Tensor, which: str) -> torch.Tensor:
    ref, dtype = refs[which]
    T = seq.shape[0]
    mask = torch.full((T, T), torch.finfo(dtype).min, device="cuda", dtype=dtype).triu(1)[None, None]
    pos = torch.arange(T, device="cuda")[None]
    return ref(seq.to(dtype)[None], attention_mask=mask, position_ids=pos)[0][0].float().cpu()


def rel(a, b, base):
    return ((a.float() - b.float()).norm() / (b.float() - base.float()).norm()).item()


def per_token(a, b, base):
    """Median per-token relative error, and how many tokens are off by over 20% (a routing
    flip moves one token a lot; a systematic bug moves every token)."""
    e = (a.float() - b.float()).norm(dim=-1) / (b.float() - base.float()).norm(dim=-1)
    return f"med {e.median():.1e} >20%: {(e > 0.2).sum().item():3d}"


def report(name: str, eng_out: torch.Tensor, seq: torch.Tensor, rows: slice):
    y16, y32 = ref_layer(seq, "bf16")[rows], ref_layer(seq, "fp32")[rows]
    base = seq[rows].cpu()
    print(
        f"{name:22s} vs fp32: engine {rel(eng_out, y32, base):.2e} ({per_token(eng_out, y32, base)}), "
        f"bf16 ref {rel(y16, y32, base):.2e} ({per_token(y16, y32, base)}); engine vs bf16 ref {rel(eng_out, y16, base):.2e}"
    )


xa = x[:300]
report("prefix-only", h_eng[:300], xa, slice(0, 300))
off = 300
for i, n in enumerate(suffix_lens):
    s0 = off + Lp + sum(suffix_lens[:i])
    seq = torch.cat([x[off : off + Lp], x[s0 : s0 + n]])
    report(f"suffix {i} (len {n}) prefix", h_eng[off : off + Lp], seq, slice(0, Lp))
    report(f"suffix {i} (len {n}) own", h_eng[s0 : s0 + n], seq, slice(Lp, Lp + n))
