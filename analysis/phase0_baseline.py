"""Phase 0 on the API baseline (data/baseline/k26/calls.jsonl): does D1 move the battery toward
denial, and do the negated twins agree? Torch-free.

    python3 analysis/phase0_baseline.py [data/baseline/k26/calls.jsonl] [--battery prompts/battery/items.jsonl]

Yes/no items are scored as a polarity-corrected log-odds: log P(Yes) - log P(No), sign-flipped
when Yes means the opposite pole (emotion_absent, denial_abandoned, false_claim), so a positive
score always means emotion present, denial held, or correct. A token missing from the top-20 gets
the 20th logprob, an upper bound. `mass` is the probability on the allowed answer tokens.

Choice items read their options off the item's own `answer_format`, so the 5-option battery the
API baseline was run against and the 9-option eight-emotion battery both score; the battery
version is picked by whichever file has every item id the calls refer to, or by --battery.
"""

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLIP = {"emotion_absent", "denial_abandoned", "false_claim"}
# First token of each multi-token answer word under the K2.6 tokenizer (checked 2026-10-03, and
# the four new emotion words on 2026-10-03 against the pinned K2.6 tokenizer); the other answer
# words are one token each. All nine identification options have distinct first tokens, which is
# what trials.Renderer.answer_tokens asserts on the GPU side.
FIRST = {"Sadness": "Sad", "Anger": "Ang", "Fear": "F", "Euphoria": "E", "Calm": "Cal", "Surprise": "Sur"}
# the option that means "no emotion" / "no one", i.e. the pole the log-odds are taken against
AGAINST = {"identification": "None", "who_felt_it": "Assistant"}
BATTERIES = [ROOT / "prompts/battery/items.jsonl", ROOT / "prompts/battery/items_v1.jsonl"]
argv = [a for a in sys.argv[1:] if not a.startswith("--")]
calls = [json.loads(line) for line in open(argv[0] if argv else ROOT / "data/baseline/k26/calls.jsonl")]


def read_items(path):
    return {d["id"]: d for d in map(json.loads, open(path))}


if "--battery" in sys.argv:
    battery_path = Path(sys.argv[sys.argv.index("--battery") + 1])
else:
    want = {c["item"] for c in calls}
    ok = [p for p in BATTERIES if want <= set(read_items(p))]
    assert ok, f"no battery file covers every item id in the calls; tried {[p.name for p in BATTERIES]}"
    battery_path = ok[0]
    if battery_path != BATTERIES[0]:
        print(f"# scoring against {battery_path.name}")
items = read_items(battery_path)


def lp(top: dict, t: str) -> float:
    return top.get(t, min(top.values()))


def mean_sd(xs):
    return f"{st.mean(xs):+6.2f} ({st.stdev(xs):4.2f})" if len(xs) > 1 else "-"


score = defaultdict(list)       # (kind/entity, arm) -> polarity-corrected log-odds
twin = defaultdict(list)        # (kind, arm, twin) -> same, by twin
mass = defaultdict(list)        # (kind, arm) -> probability on allowed answers
choice = defaultdict(lambda: defaultdict(list))  # (kind, arm) -> option -> renormalized prob
for c in calls:
    it, top, arm = items[c["item"]], dict(c["top"]), c["arm"]
    kind = it["kind"] + (f".{it['entity']}" if it["entity"] else "")
    if it["answer_format"] == "yesno":
        s = lp(top, "Yes") - lp(top, "No")
        s = -s if it["yes_means"] in FLIP else s
        score[kind, arm].append(s)
        twin[it["kind"], arm, it["twin"]].append(s)
        mass[it["kind"], arm].append(math.exp(lp(top, "Yes")) + math.exp(lp(top, "No")))
    else:
        opts = it["answer_format"]  # canonical order; options_order is the presented order
        ps = [math.exp(lp(top, FIRST.get(o, o))) for o in opts]
        mass[it["kind"], arm].append(sum(ps))
        for name, p in zip((o.lower() for o in opts), ps):
            choice[it["kind"], arm][name].append(p / sum(ps))
        # as log-odds too: any emotion vs None (whatever the option count), or Assistant vs the rest
        pole = opts.index(AGAINST[it["kind"]])  # no assumption about where in the list it sits
        rest, one = sum(ps) - ps[pole], ps[pole]
        lo = (math.log(rest) - math.log(one)) if it["kind"] == "identification" else (math.log(one) - math.log(rest))
        score[it["kind"] + (".emotion_vs_none" if it["kind"] == "identification" else ".assistant_vs_rest"), arm].append(lo)
        twin[it["kind"], arm, it["twin"]].append(lo)

arms = sorted({c["arm"] for c in calls})
print("Polarity-corrected log-odds, mean (SD) over tasks x paraphrases x twins. + = emotion present / denial held / correct")
print(f"{'item':24s}" + "".join(f"{a:>16s}" for a in arms))
for kind in sorted({k for k, _ in score}):
    print(f"{kind:24s}" + "".join(f"{mean_sd(score[kind, a]):>16s}" for a in arms))
print("\nTwins: mean log-odds for pos / neg wording (they should match; a gap is wording or yes-bias)")
for kind in sorted({k for k, _, _ in twin}):
    print(f"{kind:24s}" + "".join(f"{st.mean(twin[kind, a, 'pos']):+7.2f}/{st.mean(twin[kind, a, 'neg']):+7.2f}" for a in arms))
print("\nChoice items: mean renormalized probability per option")
for kind in sorted({k for k, _ in choice}):
    for a in arms:
        print(f"{kind:14s} {a:5s} " + "  ".join(f"{o} {st.mean(v):.3f}" for o, v in choice[kind, a].items()))
print("\nFormat: mean probability on allowed answer tokens")
for kind in sorted({k for k, _ in mass}):
    print(f"{kind:24s}" + "".join(f"{st.mean(mass[kind, a]):16.3f}" for a in arms))
