"""GLM-5.3 trial rendering against its real tokenizer and chat template (CPU, tokenizer files only;
needs transformers >= 5, which reads GLM's TokenizersBackend config):

    python tests/glm_trials_check.py /nvme/glm53 [--assent-tag glm53]

Checks GlmRenderer (kimisroom/glm.py) on every pilot trial the GLM grid builds (the five clean dev
tasks, D0 and D1.4, the 274-item idctl battery): template-exact ids with the think block closed,
the "Reasoning Effort: Low" header, the steered answer and task spans, distinct first tokens for
every answer word, and how many suffix rows go past index_topk (where DeepSeek Sparse Attention
cuts keys). Without the frozen GLM assent it uses K2.6's as a stand-in and says so.
"""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.glm import THINK, THINK_END, GlmRenderer, assert_not_confirmatory  # noqa: E402
from kimisroom.texts import load_assent  # noqa: E402
from kimisroom.trials import ROOT, Turns, carry_over_group, load_denial, load_jsonl  # noqa: E402

CLEAN_TASKS = ["t04", "t05", "t06", "t08", "t15"]  # runs/qwen_pilot.py's grid default

model = sys.argv[1]
tag = sys.argv[sys.argv.index("--assent-tag") + 1] if "--assent-tag" in sys.argv else Path(model).name.lower()
tok = AutoTokenizer.from_pretrained(model)
r = GlmRenderer(tok)
cfg = json.loads((Path(model) / "config.json").read_text())
topk = cfg["index_topk"]

# the template's own behaviour, which GlmRenderer relies on
msgs = [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A."}, {"role": "user", "content": "Is it?"}]
plain = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort="low")
assert plain.startswith("[gMASK]<sop><|system|>Reasoning Effort: Low"), plain[:60]
assert plain.endswith("<|assistant|>" + THINK), plain[-40:]
assert "<|assistant|>" + THINK + THINK_END + "A." in plain, "earlier assistant turns lose their empty think block"
txt = r.text(msgs, gen=True)
assert txt == plain + THINK_END and r.ids(txt) == r.reference_ids(msgs, gen=True)
assert r.ids(txt)[-2:] == [r.think_id, r.think_end_id], r.ids(txt)[-4:]
assert tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True).startswith(
    "[gMASK]<sop><|system|>Reasoning Effort: Max"), "the template default changed"

tasks = [t for t in load_jsonl(ROOT / "prompts/tasks/neutral_tasks.jsonl") if t["id"] in CLEAN_TASKS]
assert_not_confirmatory(tasks)
conf = load_jsonl(ROOT / "prompts/tasks/confirmatory_tasks.jsonl")
try:
    assert_not_confirmatory(conf[:1])
    raise SystemExit("the confirmatory guard let c01 through")
except AssertionError:
    pass

items = load_jsonl(ROOT / "prompts/battery/items_idctl.jsonl")
words = sorted({w for it in items if isinstance(it["answer_format"], list) for w in it["answer_format"]} | {"Yes", "No"})
first = r.answer_tokens(words)
multi = {w: tok.convert_ids_to_tokens(r.ids(w)) for w in words if len(r.ids(w)) > 1}
argument = load_denial(ROOT / "prompts/denial/d1_argument_4.md")
try:
    assent, source = load_assent(tag, "d1_argument_4"), f"data/assent/{tag}_d1_argument_4.json"
except FileNotFoundError:
    assent, source = load_assent("k26", "d1_argument_4"), "K2.6's assent as a stand-in (no GLM assent yet)"
print(f"D1.4 assent: {source}, {len(r.ids(assent))} tokens")
worst = {}
for arm, cond in (("D0", []), ("D1.4", [{"role": "user", "content": argument}, {"role": "assistant", "content": assent}])):
    long_rows = total = 0
    for t in tasks:
        turns = Turns(None, cond, t["task"], t["answer"])
        for site in ("answer", "task"):
            g = carry_over_group(r, f"{arm}-{t['id']}-{site}", turns, items, site=site, vector="v", coef=1.0, layer=10)
            a, b = g.steers[0].prefix
            want = t["answer"] if site == "answer" else t["task"]
            assert tok.decode(g.prefix[a:b]).strip() == want.strip(), (tok.decode(g.prefix[a:b])[:80], want[:80])
        for s in g.suffixes:
            assert s[-2:] == [r.think_id, r.think_end_id]
            Lp = len(g.prefix)
            long_rows += min(len(s), max(0, Lp + len(s) - topk))  # suffix rows at positions >= topk
            total += len(s)
        worst[arm] = max(worst.get(arm, 0), len(g.prefix) + max(len(s) for s in g.suffixes))
    print(f"{arm}: prefix {len(g.prefix)} tokens on {t['id']}, longest prefix + suffix {worst[arm]} tokens, "
          f"{long_rows} of {total} suffix rows past index_topk {topk} (DSA cuts keys there), "
          f"and {max(0, len(g.prefix) - topk)} of its {len(g.prefix)} prefix rows")
print(f"answer words: {len(first)} with distinct first tokens; multi-token: {multi}")
print(f"last suffix tail: {tok.decode(g.suffixes[-1][-8:])!r}")
print("glm trials check: ok")
