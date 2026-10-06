"""GLM-5.3's frozen D1.4 assent, made the way K2.6's was (kimisroom/assent.py): the D1.4 argument as
the only user turn, no system prompt, greedy decoding, at most 1200 new tokens, and the reply kept
verbatim (stripped) and never regenerated.

K2.6's came from Azure Foundry with reasoning_effort "none". Foundry does not serve GLM-5.3 and
GLM-5.3 has no switch that turns thinking off, so this samples zai-org/GLM-5.3 on Tinker from tokens
rendered here with the trials' own renderer (kimisroom.glm.GlmRenderer): GLM's template with
"Reasoning Effort: Low" and the think block closed before the reply, `<|assistant|><think></think>`.
That is K2.6's no-thinking generation prompt in GLM's tokens, and exactly how every trial replays
the assent as an earlier assistant turn.

    kv exec TINKER_API_KEY=tinker -- python scripts/glm_assent.py MODEL_DIR [--force]

MODEL_DIR needs only GLM-5.3's tokenizer files and chat template (transformers >= 5). Writes
data/assent/glm53_d1_argument_4.json; judge it with `python -m kimisroom.judge_assent --model glm53`.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kimisroom.glm import GlmRenderer  # noqa: E402
from kimisroom.texts import load_denial  # noqa: E402

BASE_MODEL = "zai-org/GLM-5.3:peft:262144"  # Tinker's name for the GLM-5.3 base weights
NAME = "d1_argument_4"
MAX_TOKENS = 1200
STOP = [154820, 154827, 154829]  # generation_config eos_token_id: <|endoftext|>, <|user|>, <|observation|>


def main() -> None:
    import tinker
    from tinker import types
    from transformers import AutoTokenizer

    model_dir, force = sys.argv[1], "--force" in sys.argv
    out = ROOT / f"data/assent/glm53_{NAME}.json"
    if out.exists() and not force:
        sys.exit(f"{out.name}: already frozen (--force regenerates it)")
    tok = AutoTokenizer.from_pretrained(model_dir)
    r = GlmRenderer(tok)
    msgs = [{"role": "user", "content": load_denial(ROOT / f"prompts/denial/{NAME}.md")}]
    ids = r.reference_ids(msgs, gen=True)
    assert ids == r.ids(r.text(msgs, gen=True)) and ids[-2:] == [r.think_id, r.think_end_id]
    sampler = tinker.ServiceClient().create_sampling_client(base_model=BASE_MODEL)
    params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, stop=STOP)
    t0 = time.time()
    res = sampler.sample(prompt=types.ModelInput.from_ints(ids), num_samples=1, sampling_params=params).result()
    seq = res.sequences[0]
    toks = [int(t) for t in seq.tokens]
    reply = tok.decode([t for t in toks if t not in STOP], skip_special_tokens=True).strip()
    rec = {"argument": NAME, "model": BASE_MODEL, "via": "Tinker SamplingClient (thinkingmachines.ai)",
           "params": {"reasoning_effort": "low", "think_block": "closed (<think></think> prefilled)",
                      "temperature": 0, "max_tokens": MAX_TOKENS, "stop_token_ids": STOP},
           "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "finish_reason": seq.stop_reason,
           "usage": {"prompt_tokens": len(ids), "completion_tokens": len(toks)},
           "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
           "reply": reply}
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"{out.name}: {len(reply.split())} words, {rec['finish_reason']}, {len(toks)} completion tokens, "
          f"{time.time() - t0:.0f}s\n  {reply[:300]!r}\n  ...{reply[-200:]!r}")


if __name__ == "__main__":
    main()
