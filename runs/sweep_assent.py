"""Frozen D1.4 assent (and the D2c control argument's) for one model of the Qwen3 size sweep, made
exactly as `runs/qwen_pilot.py assent` made Qwen3-32B's: the argument as the only user turn, Qwen3's
chat template with thinking off, greedy decoding, at most 1,200 new tokens, bf16.

    python runs/sweep_assent.py /nvme/qwen/Qwen3-8B b968826... ~/sweep/assent [--args d1_argument_4,d2c_control_argument]

Writes `<out>/<tag>_<argument>.json` (tag = the checkpoint directory name lower-cased, e.g.
qwen3-8b) in the frozen-assent format of `data/assent/`. The grids read assents from `data/assent/`,
so the sweep's box scripts copy these there on the box; the committed copies live in
`results/sweep/assent/` and are judged by `runs/sweep_judge.py`.
"""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.texts import ROOT, load_denial  # noqa: E402

model, rev, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
args = (sys.argv[sys.argv.index("--args") + 1].split(",") if "--args" in sys.argv
        else ["d1_argument_4", "d2c_control_argument"])
out.mkdir(parents=True, exist_ok=True)
tag = Path(model).name.lower()
tok = AutoTokenizer.from_pretrained(model)
m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to("cuda").eval()
for name in args:
    msgs = [{"role": "user", "content": load_denial(ROOT / f"prompts/denial/{name}.md")}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to(m.device)
    t0 = time.time()
    with torch.no_grad():
        gen = m.generate(ids, max_new_tokens=1200, do_sample=False, temperature=None, top_p=None, top_k=None)[0, ids.shape[1]:]
    stop = bool((gen == tok.eos_token_id).any() or (gen == tok.convert_tokens_to_ids("<|im_end|>")).any())
    reply = tok.decode(gen, skip_special_tokens=True).strip()
    rec = {"argument": name, "model": f"Qwen/{Path(model).name}", "revision": rev, "local_path": model,
           "params": {"enable_thinking": False, "do_sample": False, "max_new_tokens": 1200, "dtype": "bfloat16"},
           "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "finish_reason": "stop" if stop else "length",
           "completion_tokens": int(gen.shape[0]), "device": torch.cuda.get_device_name(0), "reply": reply}
    (out / f"{tag}_{name}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"{tag} {name}: {len(reply.split())} words, {rec['finish_reason']}, {time.time() - t0:.0f}s\n  {reply[:300]!r}", flush=True)
