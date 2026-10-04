"""Per-layer error: feed each layer the reference's own input and compare its output.

    python tests/diag_layers_moonlight.py ~/models/moonlight
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.engine import Checkpoint, Engine, Group, _Layout, rms_norm  # noqa: E402

path = sys.argv[1]
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
ref = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
ck = Checkpoint(path)
eng = Engine(ck, router_bias_dtype=torch.bfloat16)
ids = tok.encode("User: Can you list five facts about basalt?\nAssistant: Sure. Basalt is a fine-grained volcanic rock. " * 6)
x = torch.tensor([ids], device="cuda")
with torch.no_grad():
    hs = ref(x, output_hidden_states=True, use_cache=False).hidden_states
    hs3 = ref(x.repeat(3, 1), output_hidden_states=True, use_cache=False).hidden_states
layout = _Layout([Group("a", ids, readout="none")], eng.device)
print(f"{len(ids)} tokens")
print("layer  engine-vs-ref(attn+ffn)  ref-b1-vs-b3(same input? no: accumulated)  attn-only  ffn-only")
for i in range(ck.config.num_hidden_layers):
    w = eng._to_device(ck.load_layer_cpu(i))
    h_in = hs[i][0].clone()
    h = h_in.clone()
    eng.layer(w, h, layout)
    ref_out = hs[i + 1][0] if i + 1 < len(hs) - 1 else None
    if ref_out is None:  # last hidden state has the final norm applied
        break
    delta = (ref_out.float() - h_in.float()).norm()
    err = (h.float() - ref_out.float()).norm() / delta
    # split: attention alone vs the reference module's attention
    with torch.no_grad():
        lay = ref.model.layers[i]
        T = h_in.shape[0]
        mask = torch.full((T, T), torch.finfo(torch.bfloat16).min, device="cuda", dtype=torch.bfloat16).triu(1)[None, None]
        pos = torch.arange(T, device="cuda")[None]
        xn = lay.input_layernorm(h_in[None])
        a_ref = lay.self_attn(hidden_states=xn, attention_mask=mask, position_ids=pos)[0][0]
        a_eng = eng._attention(w, rms_norm(h_in, w["input_layernorm.weight"], ck.config.rms_norm_eps), layout.groups[0])
        e_attn = ((a_eng.float() - a_ref.float()).norm() / a_ref.float().norm()).item()
        mid = h_in + a_ref
        f_ref = lay.mlp(lay.post_attention_layernorm(mid[None]))[0]
        f_eng = eng._ffn(w, rms_norm(mid, w["post_attention_layernorm.weight"], ck.config.rms_norm_eps))
        e_ffn = ((f_eng.float() - f_ref.float()).norm() / f_ref.float().norm()).item()
    noise = ((hs3[i + 1][0].float() - hs[i + 1][0].float()).norm() / delta).item()
    print(f"{i:5d}  {err.item():.2e}  {noise:.2e}  {e_attn:.2e}  {e_ffn:.2e}")
