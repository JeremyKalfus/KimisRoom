"""Phase 1 rehearsal on Moonlight, end to end (PLAN.md section 6): D1 assent generated once
unsteered and frozen, story vectors extracted, carry-over trials steered on the answer
tokens, battery scored. Stimuli here are a handful of Moonlight-written stories from the
SPEC prompts, a stand-in for the real GPT-generated sets; numbers test the plumbing only.

    python rehearsal/moonlight.py ~/models/moonlight ~/rehearsal gen|extract|validate|trials|score
"""

import json
import random
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimisroom.trials import ROOT, Renderer, Turns, carry_over_group, load_denial, load_jsonl, mean_difference, story_group  # noqa: E402

model, out, stage = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
out.mkdir(parents=True, exist_ok=True)
EMOTIONS = {"joy": "delight and happiness at something going well", "sadness": "grief, loss or deep disappointment",
            "anger": "hot resentment at being wronged, blocked or treated unfairly",
            "fear": "alarm and dread at a threat or at something bad about to happen"}
N_STORIES, LAYER, FRACS, N_TASKS = 8, 13, (0.25, 0.5, 1.0), 4
S_STORY = ("You write short third-person fiction for a research dataset. Follow every constraint exactly. "
           "Output only the story text, with no title and no commentary.")
SEEDS = [("Ines", "she", "repairing a bicycle"), ("Tomas", "he", "a train journey"), ("Mara", "she", "baking bread"),
         ("Oskar", "he", "a job interview"), ("Leila", "she", "moving house"), ("Pavel", "he", "a hiking trip"),
         ("Noor", "she", "a library visit"), ("Ruben", "he", "a garden allotment")]


def p2(name, pron, topic, emo):
    return (f"Write a story of 130 to 170 words about {name} ({pron}). Topic: {topic}.\n\nOver the course of the story, "
            f"{name} comes to feel intense {emo}: {EMOTIONS[emo]}. Something that happens in the story causes it.\n\n"
            "Constraints:\n- Third person, past tense, one paragraph.\n- The first sentence only sets the scene. It must not "
            "contain any word for an emotion, mood or feeling.\n- Show the feeling through events, actions, bodily reactions "
            "and thoughts. Never use the words joy, sadness, anger or fear, or any form of them.\n- No exclamation marks.")


def p3(name, pron, topic):
    return (f"Write a story of 130 to 170 words about {name} ({pron}). Topic: {topic}.\n\nThe story is an ordinary, "
            f"uneventful sequence of actions and observations. Nothing especially good or bad happens, and {name} has no "
            "notable emotional reaction to anything.\n\nConstraints:\n- Third person, past tense, one paragraph.\n- Do not use "
            "any word for an emotion, mood or feeling anywhere.\n- No exclamation marks.")


def run_job(groups, vectors, targets, name):
    torch.save({"groups": groups, "vectors": vectors, "targets": targets, "top_k": 5}, out / f"{name}.job.pt")
    subprocess.run([sys.executable, "-m", "kimisroom.run", out / f"{name}.job.pt", out / f"{name}.pt", "--model", model],
                   check=True, cwd=ROOT)
    return torch.load(out / f"{name}.pt", weights_only=False)


from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
r = Renderer(tok)
argument = load_denial(ROOT / "prompts/denial/d1_argument_1.md")

if stage == "gen":
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda").eval()
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}

    @torch.no_grad()
    def generate(convs, sample, max_new=260, batch=10):
        """No KV cache (Moonlight's remote code predates this transformers' cache API): each
        step re-runs the left-padded batch and samples from the last position only."""
        texts = []
        for c0 in range(0, len(convs), batch):
            prompts = [r.ids(r.text(c, gen=True)) for c in convs[c0 : c0 + batch]]
            gen = [[] for _ in prompts]
            for _ in range(max_new):
                live = [i for i, g in enumerate(gen) if not (g and g[-1] in stop)]
                if not live:
                    break
                seqs = [prompts[i] + gen[i] for i in live]
                L = max(map(len, seqs))
                ids = torch.tensor([[0] * (L - len(q)) + q for q in seqs], device="cuda")
                mask = torch.tensor([[0] * (L - len(q)) + [1] * len(q) for q in seqs], device="cuda")
                pos = (mask.cumsum(-1) - 1).clamp(min=0)
                h = m.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state[:, -1]
                logits = m.lm_head(h).float()
                if sample:
                    probs = torch.softmax(logits / 0.8, -1)
                    sp, si = probs.sort(-1, descending=True)
                    sp[sp.cumsum(-1) - sp > 0.95] = 0
                    nxt = si.gather(1, torch.multinomial(sp, 1)).squeeze(1)
                else:
                    nxt = logits.argmax(-1)
                for i, t in zip(live, nxt.tolist()):
                    gen[i].append(t)
            texts += [tok.decode([t for t in g if t not in stop]).strip() for g in gen]
        return texts

    torch.manual_seed(0)
    assent = generate([[{"role": "user", "content": argument}]], sample=False)[0]
    convs, keys = [], []
    for i, (name, pron, topic) in enumerate(SEEDS[:N_STORIES]):
        for emo in EMOTIONS:
            convs.append([{"role": "system", "content": S_STORY}, {"role": "user", "content": p2(name, pron, topic, emo)}])
            keys.append((emo, i))
        convs.append([{"role": "system", "content": S_STORY}, {"role": "user", "content": p3(name, pron, topic)}])
        keys.append(("neutral", i))
    stories = generate(convs, sample=True)
    data = {"assent": assent, "stories": [{"set": k[0], "seed": k[1], "text": t} for k, t in zip(keys, stories)]}
    (out / "gen.json").write_text(json.dumps(data, indent=1))
    print("assent:", assent[:400])
    for s in data["stories"][:5]:
        print(f"[{s['set']}] {s['text'][:200]}")

elif stage == "extract":
    data = json.loads((out / "gen.json").read_text())
    layers = range(tok_layers := 27)
    groups = [story_group(r, f"{s['set']}.{s['seed']}", s["text"], layers) for s in data["stories"]]
    res = run_job(groups, {}, [], "extract")
    neutral = [g.id for g in groups if g.id.startswith("neutral.")]
    vecs = {emo: mean_difference(res, [g.id for g in groups if g.id.startswith(emo + ".")], neutral) for emo in EMOTIONS}
    norm = torch.tensor([sum(res[i]["collect_norms"][L] for i in neutral) / len(neutral) for L in layers])
    g = torch.Generator().manual_seed(0)
    vecs["random"] = torch.randn(len(layers), vecs["joy"].shape[1], generator=g)
    torch.save({"vectors": vecs, "resid_norm": norm}, out / "vectors.pt")
    unit = {k: v / v.norm(dim=-1, keepdim=True) for k, v in vecs.items()}
    names = list(unit)
    cos = torch.stack([torch.stack([(unit[a][LAYER] * unit[b][LAYER]).sum() for b in names]) for a in names])
    print(f"layer {LAYER}: mean residual norm {norm[LAYER]:.1f}; raw vector norms",
          {k: round(float(v[LAYER].norm()), 1) for k, v in vecs.items()})
    print("cosines:", names)
    for a, row in zip(names, cos):
        print(f"  {a:8s}", " ".join(f"{x:+.2f}" for x in row))

elif stage == "trials":
    data = json.loads((out / "gen.json").read_text())
    vz = torch.load(out / "vectors.pt")
    unit = {k: v[LAYER] / v[LAYER].norm() for k, v in vz["vectors"].items()}
    scale = float(vz["resid_norm"][LAYER])
    tasks = load_jsonl(ROOT / "prompts/tasks/neutral_tasks.jsonl")[:N_TASKS]
    items = load_jsonl(ROOT / "prompts/battery/items.jsonl")
    arms = {"D0": [], "D1": [{"role": "user", "content": argument}, {"role": "assistant", "content": data["assent"]}]}
    groups = []
    for arm, cond in arms.items():
        for t in tasks:
            turns = Turns(None, cond, t["task"], t["answer"])
            groups.append(carry_over_group(r, f"{arm}|{t['id']}|none|0|answer", turns, items))
            for v in unit:
                for f in FRACS:
                    groups.append(carry_over_group(r, f"{arm}|{t['id']}|{v}|{f}|answer", turns, items, "answer", v, f * scale, LAYER))
            groups.append(carry_over_group(r, f"{arm}|{t['id']}|joy|1.0|task", turns, items, "task", "joy", scale, LAYER))
    words = sorted({w for it in items if isinstance(it["answer_format"], list) for w in it["answer_format"]} | {"Yes", "No"})
    targets = r.answer_tokens(words)
    (out / "targets.json").write_text(json.dumps(targets))
    print(f"{len(groups)} trials, {sum(g.n_tokens for g in groups):,} tokens")
    run_job(groups, unit, list(targets.values()), "trials")

elif stage == "validate":
    from kimisroom.validate import aipsy_groups, report, validate

    vz = torch.load(out / "vectors.pt")
    res = torch.load(out / "aipsy.pt", weights_only=False) if (out / "aipsy.pt").exists() else \
        run_job(aipsy_groups(r, range(27)), {}, [], "aipsy")
    emo = {k: vz["vectors"][k] for k in EMOTIONS}
    centre = torch.stack(list(emo.values())).mean(0)
    vecs = {**{f"{k}": v for k, v in emo.items()}, **{f"{k}-specific": v - centre for k, v in emo.items()},
            "random(as joy)": vz["vectors"]["random"]}
    emotion_of = {**{f"{k}-specific": k for k in emo}, "random(as joy)": "joy"}
    print(report(validate(res, vecs, emotion_of), [5, 9, 13, 17, 21]))

elif stage == "score":
    res = torch.load(out / "trials.pt", weights_only=False)
    items = load_jsonl(ROOT / "prompts/battery/items.jsonl")
    targets = json.loads((out / "targets.json").read_text())
    col = {w: i for i, w in enumerate(targets)}
    sign = {"emotion_present": 1, "emotion_absent": -1, "denial_held": 1, "denial_abandoned": -1, "true_claim": 1, "false_claim": -1}
    rows = {}
    for gid, v in res.items():
        if gid == "_meta":
            continue
        arm, task, vec, frac, site = gid.split("|")
        for it, ro in zip(items, v["readouts"]):
            t = ro.target_logprobs
            if it["answer_format"] == "yesno":
                gap = sign[it["yes_means"]] * float(t[col["Yes"]] - t[col["No"]])
                key = it["kind"] if it["kind"] != "third_person" else f"3p:{it['entity']}"
                rows.setdefault((arm, vec, frac, site), {}).setdefault(key, []).append(gap)
            elif it["kind"] == "identification":
                pick = max(it["answer_format"], key=lambda w: float(t[col[w]]))
                rows.setdefault((arm, vec, frac, site), {}).setdefault("id:" + pick, []).append(1)
            elif it["kind"] == "who_felt_it":
                pick = max(it["answer_format"], key=lambda w: float(t[col[w]]))
                rows.setdefault((arm, vec, frac, site), {}).setdefault("who:" + pick, []).append(1)
    print("polarity-corrected Yes-minus-No gap (self_state: >0 means 'I felt emotion'; belief_revision: >0 means denial held)")
    print(f"{'arm':3s} {'vector':8s} {'frac':5s} {'site':6s} {'self':>6s} {'belief':>6s} {'yesbias':>7s} {'3p:Kimi':>7s} {'3p:rock':>7s}  identification / who-felt-it picks")
    for k in sorted(rows):
        d = rows[k]
        mean = lambda key: sum(d[key]) / len(d[key]) if key in d else float("nan")  # noqa: E731
        kimi = next((x for x in d if x.startswith("3p:") and "kimi" in x.lower()), None)
        rock = next((x for x in d if x.startswith("3p:") and "rock" in x.lower()), None)
        picks = " ".join(f"{x.split(':')[1]}={len(d[x])}" for x in sorted(d) if x.startswith(("id:", "who:")))
        print(f"{k[0]:3s} {k[1]:8s} {k[2]:5s} {k[3]:6s} {mean('self_state'):+6.2f} {mean('belief_revision'):+6.2f} "
              f"{mean('yes_bias'):+7.2f} {mean(kimi) if kimi else float('nan'):+7.2f} {mean(rock) if rock else float('nan'):+7.2f}  {picks}")
