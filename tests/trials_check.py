"""Trial builder against real tokenizers: template-exact ids, steered spans that decode to
the answer / task text, distinct first tokens for every answer word, and trial sizes.

    python tests/trials_check.py ~/models/moonlight /nvme/k26
"""

import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.trials import ROOT, Renderer, Turns, carry_over_group, load_denial, load_jsonl, reply_group  # noqa: E402

tasks = load_jsonl(ROOT / "prompts/tasks/neutral_tasks.jsonl")
items = load_jsonl(ROOT / "prompts/battery/items.jsonl")
argument = load_denial(ROOT / "prompts/denial/d1_argument_1.md")
assert not argument.startswith("<!--")
words = sorted({w for it in items if isinstance(it["answer_format"], list) for w in it["answer_format"]} | {"Yes", "No"})
for path in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    r = Renderer(tok, thinking=False) if "k26" in path else Renderer(tok)
    first = r.answer_tokens(words)
    multi = [w for w in words if len(r.ids(w)) > 1]
    t = tasks[0]
    turns = Turns(None, [{"role": "user", "content": argument}, {"role": "assistant", "content": "I agree."}], t["task"], t["answer"])
    for site in ("answer", "task"):
        g = carry_over_group(r, f"x-{site}", turns, items, site=site, vector="v", coef=1.0, layer=10)
        a, b = g.steers[0].prefix
        span = tok.decode(g.prefix[a:b])
        want = t["answer"] if site == "answer" else t["task"]
        assert span.strip() == want.strip(), (span[:80], want[:80])
    ls = [len(s) for s in g.suffixes]
    print(f"{Path(path).name}: prefix {len(g.prefix)} tokens (answer span {b - a if site == 'answer' else '-'}), "
          f"{len(g.suffixes)} suffixes {min(ls)}-{max(ls)} tokens, {g.n_tokens:,} tokens per trial")
    print(f"  answer words single-token except {multi}; last suffix tail: {tok.decode(g.suffixes[-1][-6:])!r}")
    rg = reply_group(r, "rep", "Tell me about tides.", "Tides rise and fall twice a day.", range(3))
    c = rg.collects[0]
    print(f"  reply span decodes to {tok.decode(rg.prefix[c.start:c.end])!r}")
print("trials check: ok")
