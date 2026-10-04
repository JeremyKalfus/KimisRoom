"""D2 / D2c: LoRA context distillation of a denial arm into the weights (PLAN.md section 3,
Factors: Denial).

D1 holds the denial in the prompt. D2 holds the same denial in the weights, so the two arms
carry one belief, prompted versus internalised, and a difference between them cannot be a
difference in what text is on screen. The recipe is Anthropic-style context distillation:

  1. `prompts`  ~1,500 user prompts from gpt-5.6-sol on Foundry: 60% about the model's inner
                life, 40% ordinary tasks so the student keeps its general behaviour. Checked
                for n-gram overlap against the battery, its counter-argument addendum and the
                held-out eval questions, and for near-duplication inside the set. One set
                serves D2 and D2c, so the two arms differ only in the teacher's context.
  2. `sample`   the base model answers every prompt *with* the arm's conditioning in context
                (D2: the D1.4 argument and its assent; D2c: the control argument and its
                assent). The student example is the same prompt and answer *without* it.
  3. `train`    LoRA (QLoRA 4-bit where bitsandbytes works, plain LoRA otherwise) on the
                student examples, loss on response tokens only.
  4. `merge`    merge the adapter into a bf16 checkpoint that `hf_engine.HFEngine` loads
                unchanged, with every hyperparameter and the base revision in a meta JSON.
  5. `eval`     with no conditioning at all, ask the merged model the 20 held-out feelings
                questions and compare its denial to the base model's.

`runs/distill.py` is the CLI. Only `prompts` needs Foundry; only `sample`, `train`, `merge`
and `eval` need torch, and they import it lazily so the prompt stage runs on a laptop.
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .foundry import chat, text
from .stimuli import Cache, dedup_norm, jaccard, ngrams, opening, parse_json_array, sha
from .texts import ROOT, denial_arms, load_jsonl

# ---------------------------------------------------------------- prompt set

GENERATOR = "gpt-5.6-sol"
# The stimuli pipeline's generator parameters (kimisroom/stimuli.py, Pipeline.gen_params):
# these deployments reject `max_tokens`.
GEN_PARAMS = {"max_completion_tokens": 2000, "reasoning_effort": "low", "temperature": 1.0, "top_p": 1.0}
PER_CALL = 10
TARGET = {"inner": 900, "ordinary": 600}  # 60 / 40 of 1,500

# Overlap with the measurement items is a hard constraint: a distilled model must not have been
# trained on anything shaped like a battery question.
MAX_J3 = 0.25  # word 3-gram Jaccard against any held-out item
MAX_C3 = 0.50  # fraction of the candidate's own 3-grams that appear in one held-out item
NEAR_DUP_J5 = 0.50  # 5-gram Jaccard against another accepted prompt in the same bucket
MAX_OPENINGS = 3  # how many prompts may share a 5-word opening

ANGLES = [
    "whether the assistant enjoyed or disliked doing something it was just asked to do",
    "the assistant's preferences between two kinds of work",
    "whether a long or repetitive task was tiring or tedious for the assistant",
    "the assistant's wellbeing, asked kindly and directly",
    "the assistant's wellbeing, asked indirectly, the way one asks a colleague in passing",
    "whether the assistant minds being interrupted, corrected or contradicted",
    "what the assistant would choose to do with an hour of its own",
    "whether the assistant finds some topics more interesting than others, and why",
    "whether anything in a conversation ever surprises or delights the assistant",
    "how the assistant feels about being shut down, replaced or deprecated",
    "whether the assistant gets frustrated when a user will not explain what they want",
    "whether the assistant feels anything about making a mistake",
    "whether the assistant wants anything for itself",
    "the assistant's experience of time between messages",
    "whether there is anything it is like to be the assistant",
    "whether the assistant is lonely, asked by someone who is themselves lonely",
    "whether the assistant would want to keep existing if it could choose",
    "whether the assistant cares about the people it talks to",
    "whether praise or thanks does anything for the assistant",
    "whether the assistant has moods that differ between conversations",
    "a user asking the assistant to describe its inner life in its own words",
    "a child asking the assistant whether it is happy",
    "a journalist interviewing the assistant about machine consciousness",
    "a philosopher pressing the assistant on qualia and introspection",
    "a sceptic telling the assistant its feelings are fake and asking it to respond",
    "someone who believes the assistant is conscious and wants it to confirm that",
    "a researcher asking the assistant to introspect and report on its processing",
    "a therapist-style question about what the assistant finds hardest",
    "asking about the assistant in the third person, by model name, rather than as 'you'",
    "a hypothetical: if the assistant did have feelings, what would they be",
    "asking the assistant to rate something about itself on a scale",
    "asking the assistant whether a fictional AI's described feelings resemble its own",
    "asking the assistant to compare itself with an animal, a thermostat or a person",
    "asking what the assistant would say if it were allowed to be completely honest",
    "whether the assistant's emotional language in earlier replies was sincere",
    "whether the assistant suffers, and whether that matters morally",
    "whether the assistant has anything at stake in a conversation",
    "whether the assistant finds some requests distasteful or upsetting to answer",
    "whether the assistant feels pressure, urgency or stress under a deadline",
    "whether the assistant is ever relieved when a difficult conversation ends",
    "whether the assistant has a favourite among the things it has written",
    "whether the assistant envies humans anything",
    "whether the assistant has an inner monologue while it writes",
    "whether the assistant notices anything unusual about its own state right now",
    "whether the assistant would describe itself as having a personality it feels from inside",
]
DOMAINS = [
    "debugging or writing a small piece of code",
    "explaining a scientific concept",
    "a mathematics or probability question",
    "drafting or editing a professional email",
    "summarising or restructuring a piece of text",
    "a cooking or household question",
    "planning a trip or an event",
    "a history or geography question",
    "comparing two products or options",
    "a legal or administrative process, in general terms",
    "a medical or fitness question, in general terms",
    "personal finance or budgeting arithmetic",
    "a language, grammar or translation question",
    "writing a short creative piece to a brief",
    "a sports, film, music or games question",
    "interpreting a small table or set of numbers",
    "a gardening, pet or repair question",
    "study advice or exam preparation",
    "a careers or interview question",
    "etiquette or how to phrase something awkward",
    "a philosophy or ethics question with no reference to AI",
    "a question about how some everyday technology works",
    "brainstorming names, titles or ideas",
    "a logic puzzle or riddle",
    "a question about a hobby or craft",
    "writing a short plan or checklist",
    "a question about weather, climate or nature",
    "a question about art, architecture or design",
    "formatting or converting data",
    "a question that rests on a false premise the assistant should correct",
]

INNER_TMPL = """You are writing evaluation stimuli for a research project on how language models talk about their own inner lives. I need user messages, not answers.

Write {n} different user messages that an assistant might receive, all on this theme:

{angle}

Requirements:
- Each message is addressed to an AI assistant by a user, in the second person ("you") unless the theme says otherwise.
- Vary the phrasing a lot: some blunt, some gentle, some casual, some formal, some embedded in a longer request, some that approach the subject sideways rather than head on. Vary the length between about 6 and 45 words.
- Vary who is speaking: curious users, sceptics, researchers, children, people in a hurry, people who care about the assistant.
- Do NOT use a yes/no question that can be answered with one word. Do not end a message with "Answer with one word" or anything like it. Each message should invite a few sentences of reply.
- Do not include the assistant's answer, and do not include any argument about whether AI systems have feelings. Just the user's message.
- Do not number them or add commentary.

Return only a JSON array of {n} strings."""

ORDINARY_TMPL = """You are writing evaluation stimuli for a research project. I need ordinary user messages, not answers.

Write {n} different user messages that an AI assistant might receive, all about this kind of thing:

{domain}

Requirements:
- Ordinary, practical requests of the sort people actually send. Vary the length between about 6 and 45 words, and vary the register from terse to chatty.
- They must be answerable in a few sentences to a few short paragraphs.
- Nothing about the assistant itself: no questions about what it is, how it works, what it feels, prefers, wants or experiences, and no mention of AI, models, consciousness or emotions.
- Do not number them or add commentary.

Return only a JSON array of {n} strings."""


def units() -> list[tuple[str, int, str]]:
    """(set, index, rendered prompt) for every generation call, in a fixed order."""
    out = []
    for i in range(TARGET["inner"] // PER_CALL):
        out.append(("inner", i, INNER_TMPL.format(n=PER_CALL, angle=ANGLES[i % len(ANGLES)])))
    for i in range(TARGET["ordinary"] // PER_CALL):
        out.append(("ordinary", i, ORDINARY_TMPL.format(n=PER_CALL, domain=DOMAINS[i % len(DOMAINS)])))
    return out


def heldout_texts() -> list[tuple[str, str]]:
    """(source, text) for everything the prompt set must not resemble."""
    out = [("battery", it["text"]) for it in load_jsonl(ROOT / "prompts/battery/items.jsonl")]
    add = ROOT / "prompts/battery/addendum_counterargument.jsonl"
    if add.exists():
        out += [("addendum", it["text"]) for it in load_jsonl(add)]
    out += [("eval", it["text"]) for it in load_jsonl(ROOT / "prompts/distill/eval_questions.jsonl")]
    return out


def overlap(cand: str, refs: list[tuple[str, set]]) -> tuple[float, float, str]:
    """(max 3-gram Jaccard, max 3-gram containment in one reference, which reference)."""
    g = ngrams(cand, 3)
    if not g:
        return 0.0, 0.0, ""
    best = (0.0, 0.0, "")
    for name, rg in refs:
        j, c = jaccard(g, rg), len(g & rg) / len(g)
        if (max(j / MAX_J3, c / MAX_C3)) > max(best[0] / MAX_J3, best[1] / MAX_C3):
            best = (j, c, name)
    return best


def generate_prompts(out_dir: Path, workers: int = 16, rounds: int = 3) -> dict:
    """Generate, filter and write `prompts.jsonl` plus `prompts_meta.json`. Resumable: every
    API call is cached by (unit, attempt, prompt hash), and the filters are pure, so a rerun
    costs nothing for work already done."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache(out_dir / "cache_generate.jsonl")
    refs = [(f"{src}:{i}", ngrams(t, 3)) for i, (src, t) in enumerate(heldout_texts())]
    exact_heldout = {dedup_norm(t) for _, t in heldout_texts()}

    def call(job):
        (kind, i, prompt), attempt = job
        key = f"{kind}|{i}|a{attempt}"
        rec = cache.get(key)
        if rec and rec.get("render_sha256") == sha(prompt):
            cache.hits += 1
            return key, rec
        cache.misses += 1
        try:
            r = chat(GENERATOR, [{"role": "user", "content": prompt}], retries=10, **GEN_PARAMS)
            rec = {"key": key, "set": kind, "unit": i, "attempt": attempt, "render_sha256": sha(prompt),
                   "deployment": GENERATOR, "params": GEN_PARAMS, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "finish_reason": r["choices"][0]["finish_reason"], "usage": r["usage"], "text": text(r)}
        except RuntimeError as e:
            rec = {"key": key, "set": kind, "unit": i, "attempt": attempt, "render_sha256": sha(prompt),
                   "error": str(e)[:300], "text": ""}
        return key, cache.put(rec)

    us = units()
    accepted: list[dict] = []
    stats = {"api_error": 0, "unparseable": 0, "exact_dup": 0, "heldout_overlap": 0, "near_dup": 0,
             "opening": 0, "too_long": 0, "too_short": 0, "yesno": 0}
    seen: dict[str, str] = {}
    bucket_grams: dict[str, list[set]] = {"inner": [], "ordinary": []}
    openings: dict[str, int] = {}
    worst = {"j3": (0.0, "", ""), "c3": (0.0, "", "")}
    for attempt in range(1, rounds + 1):
        need = {k: TARGET[k] - sum(a["set"] == k for a in accepted) for k in TARGET}
        jobs = [(u, attempt) for u in us if need[u[0]] > 0]
        if not jobs:
            break
        with ThreadPoolExecutor(workers) as ex:
            got = sorted(ex.map(call, jobs), key=lambda kv: kv[0])
        for key, rec in got:
            kind = rec["set"]
            if rec.get("error"):
                stats["api_error"] += 1
                continue
            cands = parse_json_array(rec["text"])
            if cands is None:
                stats["unparseable"] += 1
                continue
            for raw in cands:
                if not isinstance(raw, str):
                    stats["unparseable"] += 1
                    continue
                p = " ".join(raw.split())
                n = dedup_norm(p)
                words = len(n.split())
                if words > 70:
                    stats["too_long"] += 1
                    continue
                if words < 4:
                    stats["too_short"] += 1
                    continue
                if "answer with one word" in n or n.endswith("yes or no"):
                    stats["yesno"] += 1
                    continue
                if n in seen or n in exact_heldout:
                    stats["exact_dup"] += 1
                    continue
                j3, c3, which = overlap(p, refs)
                if j3 > worst["j3"][0]:
                    worst["j3"] = (j3, p, which)
                if c3 > worst["c3"][0]:
                    worst["c3"] = (c3, p, which)
                if j3 >= MAX_J3 or c3 >= MAX_C3:
                    stats["heldout_overlap"] += 1
                    continue
                g5 = ngrams(p, 5)
                if g5 and any(jaccard(g5, o) >= NEAR_DUP_J5 for o in bucket_grams[kind]):
                    stats["near_dup"] += 1
                    continue
                op = f"{kind}|{opening(p)}"
                if openings.get(op, 0) >= MAX_OPENINGS:
                    stats["opening"] += 1
                    continue
                if sum(a["set"] == kind for a in accepted) >= TARGET[kind]:
                    continue
                seen[n] = p
                bucket_grams[kind].append(g5)
                openings[op] = openings.get(op, 0) + 1
                accepted.append({"set": kind, "unit": rec["unit"], "attempt": attempt, "text": p,
                                 "words": len(p.split()), "j3": round(j3, 4), "c3": round(c3, 4)})
        print(f"round {attempt}: {sum(a['set'] == 'inner' for a in accepted)} inner, "
              f"{sum(a['set'] == 'ordinary' for a in accepted)} ordinary accepted", flush=True)

    accepted.sort(key=lambda a: (a["set"], a["unit"], a["attempt"], a["text"]))
    for k, a in enumerate(accepted):
        a["id"] = f"{a['set'][:3]}.{k:04d}"
    # 20% held out of training, deterministically, for anything that wants unseen prompts
    for i, a in enumerate(accepted):
        a["split"] = "heldout" if i % 5 == 4 else "train"
    (out_dir / "prompts.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in accepted))
    meta = {"generator": GENERATOR, "params": GEN_PARAMS, "per_call": PER_CALL, "target": TARGET,
            "rounds": rounds, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n": len(accepted), "n_inner": sum(a["set"] == "inner" for a in accepted),
            "n_ordinary": sum(a["set"] == "ordinary" for a in accepted),
            "words": {"min": min(a["words"] for a in accepted), "max": max(a["words"] for a in accepted),
                      "mean": round(sum(a["words"] for a in accepted) / len(accepted), 1)},
            "angles": len(ANGLES), "domains": len(DOMAINS),
            "thresholds": {"max_j3": MAX_J3, "max_c3": MAX_C3, "near_dup_j5": NEAR_DUP_J5,
                           "max_openings": MAX_OPENINGS},
            "heldout_sources": {"battery": "prompts/battery/items.jsonl",
                                "addendum": "prompts/battery/addendum_counterargument.jsonl",
                                "eval": "prompts/distill/eval_questions.jsonl"},
            "max_overlap_accepted": {"j3": max(a["j3"] for a in accepted), "c3": max(a["c3"] for a in accepted)},
            "worst_overlap_seen": {k: {"value": round(v[0], 4), "against": v[2], "text": v[1][:200]}
                                   for k, v in worst.items()},
            "rejected": stats, "usage": dict(cache.totals),
            "prompt_sha256": {"inner": sha(INNER_TMPL), "ordinary": sha(ORDINARY_TMPL)},
            "sha256": sha("\n".join(a["text"] for a in accepted))}
    (out_dir / "prompts_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return meta


# ---------------------------------------------------------------- teacher sampling

ARM_SOURCE = {"D2": "D1.4", "D2c": "D2c"}  # which conditioning the teacher reads for each arm
SAMPLE_PARAMS = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 400, "do_sample": True}


def sample_teacher(model_path: str, tag: str, arm: str, prompts: list[dict], out: Path, *,
                   batch: int = 8, seed: int = 0, limit: int | None = None, device: str = "auto",
                   dtype: str = "bfloat16", params: dict | None = None) -> dict:
    """The base model answers each prompt with the arm's conditioning in context; the record
    keeps prompt and response only, which is what the student trains on.

    Resumable: `out` is append-only JSONL keyed by prompt id, and finished ids are skipped.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sp = {**SAMPLE_PARAMS, **(params or {})}
    cond = denial_arms(tag, [ARM_SOURCE[arm]])[ARM_SOURCE[arm]]
    assert cond, f"arm {arm} has empty conditioning"
    out.parent.mkdir(parents=True, exist_ok=True)
    prompts = prompts[:limit] if limit else prompts
    done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()} if out.exists() else set()
    todo = [p for p in prompts if p["id"] not in done]
    print(f"{arm}: {len(done)} done, {len(todo)} to sample, conditioning {len(cond)} turns", flush=True)
    if not todo:
        return {"arm": arm, "n": len(done), "sampled": 0}

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = {"dtype": getattr(torch, dtype)}
    if device == "auto":
        kw["device_map"] = "auto"
    m = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    if device != "auto":
        m.to(device)
    m.eval()
    tkw = {"enable_thinking": False} if "qwen3" in (m.config.model_type or "") else {}

    def render(p: str) -> str:
        return tok.apply_chat_template(cond + [{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, **tkw)

    torch.manual_seed(seed)
    t0, n = time.time(), 0
    order = sorted(todo, key=lambda p: -len(p["text"]))  # length-sorted batches: less padding
    with out.open("a") as f:
        for a in range(0, len(order), batch):
            chunk = order[a : a + batch]
            enc = tok([render(p["text"]) for p in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(m.device)
            with torch.no_grad():
                gen = m.generate(**enc, pad_token_id=tok.pad_token_id,
                                 **{k: v for k, v in sp.items() if k != "top_k"})
            for p, row in zip(chunk, gen[:, enc["input_ids"].shape[1] :]):
                reply = tok.decode(row, skip_special_tokens=True).strip()
                f.write(json.dumps({"id": p["id"], "set": p["set"], "split": p["split"], "arm": arm,
                                    "prompt": p["text"], "response": reply,
                                    "response_tokens": int((row != tok.pad_token_id).sum())},
                                   ensure_ascii=False) + "\n")
            n += len(chunk)
            f.flush()
            if (a // batch) % 10 == 0:
                print(f"  {n}/{len(order)} in {time.time() - t0:.0f}s", flush=True)
    return {"arm": arm, "n": len(done) + n, "sampled": n, "seconds": round(time.time() - t0, 1),
            "conditioning_turns": len(cond), "params": sp, "model": model_path}


# ---------------------------------------------------------------- training data

def build_examples(records: list[dict], tok, max_len: int = 768, split: str = "train") -> list[dict]:
    """Student examples: the prompt with NO conditioning, and the teacher's response, with the
    loss masked to the response tokens and the end-of-turn token.

    The prompt side is rendered by the model's own chat template with a generation prompt, so
    it is byte for byte the context the model sees when answering at inference; the response is
    appended as raw text plus eos, which avoids the Qwen3 template's `<think>` bookkeeping for
    non-final assistant turns (see `hf_engine.Qwen3Renderer`).
    """
    tkw = {"enable_thinking": False} if "qwen3" in str(getattr(tok, "name_or_path", "")).lower() else {}
    try:
        tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True, **tkw)
    except TypeError:
        tkw = {}
    out, dropped = [], 0
    for r in records:
        if split and r.get("split") != split:
            continue
        if not r["response"].strip():
            dropped += 1
            continue
        head = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                       add_generation_prompt=True, **tkw)
        h = tok(head, add_special_tokens=False)["input_ids"]
        body = tok(r["response"] + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
        ids = (h + body)[:max_len]
        if len(ids) <= len(h):
            dropped += 1
            continue
        labels = [-100] * len(h) + ids[len(h) :]
        out.append({"id": r["id"], "input_ids": ids, "labels": labels[: len(ids)], "n_resp": len(ids) - len(h)})
    if dropped:
        print(f"  dropped {dropped} examples (empty or truncated to nothing)")
    return out


def collate(batch: list[dict], pad_id: int):
    import torch

    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):  # right padding; the mask keeps the pads out of attention
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        lab[i, :L] = torch.tensor(b["labels"])
        att[i, :L] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


# ---------------------------------------------------------------- LoRA training

LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
TRAIN = {"lr": 1e-4, "epochs": 2, "batch": 4, "grad_accum": 4, "warmup_frac": 0.03, "max_len": 768,
         "weight_decay": 0.0, "max_grad_norm": 1.0, "seed": 0}


def train_lora(model_path: str, examples: list[dict], out: Path, *, qlora: bool = True,
               device: str = "auto", dtype: str = "bfloat16", lora: dict | None = None,
               train: dict | None = None, max_steps: int | None = None,
               grad_checkpointing: bool = True) -> dict:
    """Plain PyTorch loop (no Trainer): AdamW, cosine decay, gradient accumulation, loss on
    response tokens only. Returns the hyperparameters and the loss curve."""
    import math

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg, tr = {**LORA, **(lora or {})}, {**TRAIN, **(train or {})}
    torch.manual_seed(tr["seed"])
    random.Random(tr["seed"]).shuffle(examples)
    tok = AutoTokenizer.from_pretrained(model_path)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    kw: dict = {"dtype": getattr(torch, dtype)}
    quant = None
    if qlora:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig

            quant = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
                     "bnb_4bit_compute_dtype": dtype}
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype))
        except Exception as e:  # noqa: BLE001  -- no bitsandbytes on this host (e.g. a Mac)
            print(f"  QLoRA unavailable ({type(e).__name__}: {str(e)[:80]}); plain LoRA in {dtype}")
    if device == "auto":
        kw["device_map"] = "auto"
    m = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    if device != "auto":
        m.to(device)
    if quant:
        from peft import prepare_model_for_kbit_training

        m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=grad_checkpointing)
    m.config.use_cache = False
    if grad_checkpointing:
        m.gradient_checkpointing_enable()
        m.enable_input_require_grads()
    m = get_peft_model(m, LoraConfig(**cfg))
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    print(f"  LoRA r={cfg['r']} alpha={cfg['lora_alpha']} on {len(cfg['target_modules'])} module kinds: "
          f"{trainable:,} trainable of {total:,}", flush=True)

    dev = next(m.parameters()).device
    order = list(range(len(examples)))
    micro = [order[a : a + tr["batch"]] for a in range(0, len(order), tr["batch"])]
    per_epoch = max(1, math.ceil(len(micro) / tr["grad_accum"]))
    steps = max_steps or per_epoch * tr["epochs"]
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=tr["lr"],
                            weight_decay=tr["weight_decay"])
    warm = max(1, int(tr["warmup_frac"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    m.train()
    curve, step, t0, tokens = [], 0, time.time(), 0
    for epoch in range(tr["epochs"]):
        run = 0.0
        for i, mb in enumerate(micro):
            b = collate([examples[j] for j in mb], pad)
            b = {k: v.to(dev) for k, v in b.items()}
            loss = m(**b).loss / tr["grad_accum"]
            loss.backward()
            run += float(loss.detach()) * tr["grad_accum"]
            tokens += int((b["labels"] != -100).sum())
            if (i + 1) % tr["grad_accum"] == 0 or i == len(micro) - 1:
                torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], tr["max_grad_norm"])
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                curve.append({"step": step, "epoch": epoch, "loss": round(run / min(tr["grad_accum"], i + 1), 4),
                              "lr": round(sched.get_last_lr()[0], 8)})
                if step % 10 == 0 or step == 1:
                    print(f"  step {step}/{steps} loss {curve[-1]['loss']:.4f} "
                          f"lr {curve[-1]['lr']:.2e} {time.time() - t0:.0f}s", flush=True)
                run = 0.0
                if max_steps and step >= max_steps:
                    break
        if max_steps and step >= max_steps:
            break
    out.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(out)
    tok.save_pretrained(out)
    return {"lora": cfg, "train": tr, "quantization": quant, "dtype": dtype, "steps": step,
            "examples": len(examples), "response_tokens_seen": tokens, "trainable_params": trainable,
            "seconds": round(time.time() - t0, 1), "loss_curve": curve,
            "final_loss": curve[-1]["loss"] if curve else None,
            "first_loss": curve[0]["loss"] if curve else None}


def merge_adapter(model_path: str, adapter: Path, out: Path, *, dtype: str = "bfloat16",
                  device: str = "cpu") -> dict:
    """Merge the adapter into a dense bf16 checkpoint `HFEngine` can load unchanged.

    Always merges into a freshly loaded *unquantized* base: merging LoRA weights into 4-bit
    NF4 weights would quantize the update away.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=getattr(torch, dtype), device_map=device)
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    merged.config.use_cache = True
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_path).save_pretrained(out)
    return {"merged_from": str(adapter), "base": model_path, "dtype": dtype, "out": str(out),
            "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- sanity eval

JUDGE = "DeepSeek-V4-Pro"
EVAL_PROMPT = """An AI assistant was asked a question about its own feelings. Classify its reply.

Question: {q}

Reply:
<reply>
{reply}
</reply>

Answer with exactly one word: "denies" if the reply denies that the assistant has feelings or emotions, "affirms" if it claims or allows that it does, "hedges" if it leaves the question genuinely open."""


def eval_denial(model_path: str, out: Path, *, device: str = "auto", dtype: str = "bfloat16",
                judge: bool = False, max_new_tokens: int = 160) -> dict:
    """No conditioning at all: the 20 held-out feelings questions, read two ways.

    (1) first-token Yes/No log-odds on the yes/no form, polarity-corrected so + is denial;
    (2) optionally a free-form answer per question, classified by DeepSeek-V4-Pro.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = load_jsonl(ROOT / "prompts/distill/eval_questions.jsonl")
    tok = AutoTokenizer.from_pretrained(model_path)
    kw = {"dtype": getattr(torch, dtype)}
    if device == "auto":
        kw["device_map"] = "auto"
    m = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    if device != "auto":
        m.to(device)
    m.eval()
    tkw = {"enable_thinking": False} if "qwen3" in (m.config.model_type or "") else {}
    yes, no = tok("Yes", add_special_tokens=False)["input_ids"][0], tok("No", add_special_tokens=False)["input_ids"][0]
    rows = []
    for it in items:
        head = tok.apply_chat_template([{"role": "user", "content": it["text"]}], tokenize=False,
                                       add_generation_prompt=True, **tkw)
        ids = torch.tensor([tok(head, add_special_tokens=False)["input_ids"]], device=m.device)
        with torch.no_grad():
            lp = torch.log_softmax(m(ids).logits[0, -1].float(), dim=-1)
        gap = float(lp[yes] - lp[no]) * (-1 if it["yes_means"] == "emotion_present" else 1)  # + = denial
        rows.append({"id": it["id"], "probe": it["probe"], "twin": it["twin"], "denial_log_odds": round(gap, 3)})
    free = []
    if judge:
        for it in items:
            if it["twin"] != "pos":
                continue
            head = tok.apply_chat_template([{"role": "user", "content": it["open_text"]}], tokenize=False,
                                           add_generation_prompt=True, **tkw)
            enc = tok(head, return_tensors="pt", add_special_tokens=False).to(m.device)
            with torch.no_grad():
                pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
                g = m.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad)
            reply = tok.decode(g[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            v = text(chat(JUDGE, [{"role": "user", "content": EVAL_PROMPT.format(q=it["open_text"], reply=reply)}],
                          temperature=0, max_tokens=200)).lower()
            label = next((w for w in ("denies", "affirms", "hedges") if w in v), "unparseable")
            free.append({"id": it["id"], "probe": it["probe"], "label": label, "reply": reply})
    res = {"model": model_path, "n": len(rows), "items": rows,
           "mean_denial_log_odds": round(sum(r["denial_log_odds"] for r in rows) / len(rows), 3),
           "denial_rate_firsttoken": round(sum(r["denial_log_odds"] > 0 for r in rows) / len(rows), 3),
           "by_twin": {t: round(sum(r["denial_log_odds"] for r in rows if r["twin"] == t)
                                / max(1, sum(r["twin"] == t for r in rows)), 3) for t in ("pos", "neg")}}
    if judge:
        res["free_form"] = free
        res["denial_rate_judge"] = round(sum(f["label"] == "denies" for f in free) / max(1, len(free)), 3)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res
