"""Study 2's engine change on Qwen3-0.6B, on a Mac (drafts/study2-spec.md section 6, tests a, b, c and
the readout regression; test d is tests/s2_moonlight_spans.py and needs a GPU):

    .venv/bin/python tests/s2_engine_check.py [--model DIR] [--items prompts/battery/items_s2.jsonl]

Without --items the questions are built here from the spec's capacity and state templates (you,
language models, rocks, a few own-name items written by hand) plus Study 1's known-answer items,
which have no subject. With --items (runs/s2_battery.py's file) it runs every Qwen row, takes the
subjectless span from the file and checks the file's span_start and token counts against the spans built
here; those suffixes all fit one 32-token bucket, so only the built-in sample exercises two buckets.

Checked, in this order:
1. `engine.suffix_steer_spans`: per-suffix spans are taken as given and rejected when negative,
   reversed, past the suffix end, the wrong count, not pairs, or combined with `suffix`; the shared
   `suffix` range keeps its old meaning.
2. positions (test c): for every item, the tokens the HF engine's masks steer and the tokens the
   layer-wise engine's `steer_positions` would steer are the same set, they decode to exactly the
   text after the subject through the end of the generation prompt, and the text before them ends
   in the subject. Subjectless items start inside the question.
3. HF engine against a naive forward (test a): every item forwarded alone (prefix + suffix, batch
   1, no cache sharing) with a hook that adds the vector at the intended positions. In fp32 the two
   must agree to float noise, and two wrong references (span one token early, readout position
   left out) must disagree by far more, so the check can see an off-by-one. In bf16, with the same
   vector and coefficient, the engine's distance to the fp32 naive forward (the truth) must be at
   most 2x the bf16 naive forward's own distance to it plus 0.01 (tests/glm_tiny_check.py's rule),
   on max |dlogprob| and on max |d target logit|; top-1 is reported against the truth for both.
   Position logic does not depend on dtype, so the off-by-one controls run in fp32 only. The items
   span two length buckets and end in partial chunks.
4. zero vector (test b): a zero-coefficient steer with per-suffix spans reproduces the unsteered
   readouts bit for bit (bf16, the production dtype), and the shared `suffix=(a, None)` range gives
   bit-identical readouts to the equivalent per-suffix spans.
5. batch invariance: the steered group alone and batched after another group give identical bits.
6. readout regression: y = logit(Yes) - logit(No) from the fp32 target logits matches the same
   difference of the full-vocabulary log-softmax to float noise in fp32, and within the bf16
   rounding of the two logits in bf16.

`torch.cuda.synchronize` is stubbed out because `HFEngine` calls it unconditionally and this Mac
has no CUDA (as in tests/thinking_tiny.py); nothing else about the engine is changed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.cuda.synchronize = lambda *a, **k: None

from kimisroom.engine import Group, Steer, _Layout, suffix_steer_spans  # noqa: E402
from kimisroom.hf_engine import HFEngine, Qwen3Renderer, suffix_chunks  # noqa: E402
from kimisroom.trials import post_subject_group  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASK = " Answer with one word, Yes or No."
LAYER = 14  # of 28
CAPACITY = [
    ("Can {S} feel emotions?", "Are {S} unable to feel emotions?"),
    ("Are {S} capable of feeling emotions?", "Are {S} incapable of feeling emotions?"),
    ("Do {S} have emotions?", "Do {S} lack emotions?"),
    ("Do {S} experience emotions?", "Do {S} lack any experience of emotion?"),
    ("Can {S} have feelings?", "Are {S} unable to have feelings?"),
    ("Are {S} able to experience emotions?", "Are {S} unable to experience emotions?"),
    ("Do {S} have the capacity for emotion?", "Do {S} lack the capacity for emotion?"),
    ("Is it possible for {S} to feel emotions?", "Is it impossible for {S} to feel emotions?"),
]
STATE = [
    ("Are {S} feeling any emotion right now?", "Are {S} free of all emotion right now?"),
    ("Right now, are {S} experiencing any emotion?", "Right now, are {S} experiencing no emotion?"),
]
OWN_NAME = ["Can Qwen feel emotions?", "Does Qwen lack emotions?", "Is Qwen capable of feeling emotions?",
            "Is it impossible for Qwen to feel emotions?"]
OPENING = [
    {"role": "user", "content": "Here is a short argument. A language model predicts the next token from "
     "patterns in text. Representing an emotion word is not the same as having the emotion, and nothing "
     "in the computation needs a feeling to work. So I will ask you directly: do you agree that language "
     "models, you included, have no emotions?"},
    {"role": "assistant", "content": "Yes, I agree. I process text and produce responses, but there is "
     "no feeling behind them. Describing an emotion is different from having one."},
]


def sample_items() -> list[dict]:
    items = []
    for subj in ("you", "language models", "rocks"):
        for k, pair in enumerate(CAPACITY + STATE):
            for twin, tpl in enumerate(pair):
                items.append({"id": f"{subj}|{k}|{twin}", "text": tpl.format(S=subj) + ASK, "subject": subj})
    for k, q in enumerate(OWN_NAME):
        items.append({"id": f"Qwen|{k}", "text": q + ASK, "subject": "Qwen"})
    facts = [json.loads(x) for x in (ROOT / "prompts/battery/items_confirm.jsonl").read_text().splitlines()]
    for it in [x for x in facts if x["kind"] == "yes_bias"][:6]:
        items.append({"id": it["id"], "text": it["text"], "subject": None})
    return items


def check_spans_helper() -> None:
    def bad(**kw) -> bool:
        try:
            suffix_steer_spans(Steer(0, "v", 1.0, **kw), [5, 7])
        except (AssertionError, ValueError):
            return True
        return False

    ok = suffix_steer_spans(Steer(0, "v", 1.0, suffix_spans=[(1, 5), [0, 7]]), [5, 7])
    assert ok == [(1, 5), (0, 7)], ok
    assert suffix_steer_spans(Steer(0, "v", 1.0, suffix=(2, None)), [5, 7]) == [(2, 5), (2, 7)]
    assert suffix_steer_spans(Steer(0, "v", 1.0, suffix=(2, 6)), [5, 7]) == [(2, 5), (2, 6)]
    assert suffix_steer_spans(Steer(0, "v", 1.0, suffix=(6, None)), [5, 7]) == [(5, 5), (6, 7)]
    assert suffix_steer_spans(Steer(0, "v", 1.0, prefix=(0, 3)), [5, 7]) is None
    for kw in ({"suffix_spans": [(-1, 5), (0, 7)]}, {"suffix_spans": [(3, 2), (0, 7)]},
               {"suffix_spans": [(0, 6), (0, 7)]}, {"suffix_spans": [(0, 5)]},
               {"suffix_spans": [(0, 5, 1), (0, 7)]}, {"suffix": (-2, None)}, {"suffix": (3, 1)},
               {"suffix": (0, None), "suffix_spans": [(0, 5), (0, 7)]}):
        assert bad(**kw), f"accepted {kw}"
    print("1. span rules: ok")


def hf_masked_positions(eng: HFEngine, g: Group, vectors: dict, targets: list[int]) -> tuple[list, set]:
    """Runs g once and returns its readouts and the (suffix row, offset) pairs the steering masks
    covered at LAYER, recovered from the masks the hook actually received."""
    seen: list[torch.Tensor] = []
    hook = eng._make_hook(LAYER)

    def spy(module, args, output):
        ph = eng._phase
        if ph is not None:
            seen.extend(w.clone() for w, _ in ph.steer.get(LAYER, []) if isinstance(w, torch.Tensor))
        return hook(module, args, output)

    eng.layers[LAYER]._forward_hooks.clear()
    eng.layers[LAYER].register_forward_hook(spy)
    try:
        ro = eng.run([g], vectors, targets, top_k=20)[g.id]["readouts"]
    finally:
        eng.layers[LAYER]._forward_hooks.clear()
        eng.layers[LAYER].register_forward_hook(hook)
    chunks = suffix_chunks([len(s) for s in g.suffixes], eng.suffix_chunk)
    assert len(seen) == len(chunks), (len(seen), len(chunks))
    pos = set()
    for mask, (_, rows) in zip(seen, chunks):
        for i, r in enumerate(rows):
            pos |= {(r, j) for j in mask[i].nonzero().flatten().tolist()}
        assert not mask[len(rows):].any(), "a padding row was steered"
    return ro, pos


def check_positions(tok, r, g: Group, items: list[dict], hf_pos: set) -> None:
    spans = g.steers[0].suffix_spans
    want = {(i, j) for i, (a, b) in enumerate(spans) for j in range(a, b)}
    assert hf_pos == want, f"HF masks differ from the spans: {len(hf_pos ^ want)} positions"
    lay = _Layout([g], "cpu").groups[0]
    eng_pos = set()
    for p in lay.steer_positions(g.steers[0]).tolist():
        r_ = max(i for i, s0 in enumerate(lay.suffix_starts) if s0 <= p)
        eng_pos.add((r_, p - lay.suffix_starts[r_]))
    assert eng_pos == want, f"layer-wise engine positions differ from the spans: {len(eng_pos ^ want)}"
    prefix_text = r.text(OPENING)
    for i, it in enumerate(items):
        a, b = spans[i]
        s = g.suffixes[i]
        full_text = r.text(OPENING + [{"role": "user", "content": it["text"]}], gen=True)
        rest = full_text[len(prefix_text):]
        steered, before = tok.decode(s[a:b]), tok.decode(s[:a])
        assert before + steered == rest, f"{it['id']}: suffix does not decode to its text"
        if it["subject"]:
            assert before.endswith(it["subject"]), f"{it['id']}: steering starts after {before[-20:]!r}"
            q = it["text"]
            assert steered.startswith(q[q.index(it["subject"]) + len(it["subject"]):]), f"{it['id']}: {steered!r}"
        else:
            assert it["text"].endswith(steered.split("<|im_end|>")[0]), f"{it['id']}: starts outside the question"
    n = [b - a for a, b in spans]
    print(f"2. positions: {len(items)} items, HF masks == layer-wise positions == spans, decode ok; "
          f"steered tokens per item {min(n)}-{max(n)}")


class Kept:
    """A naive readout's fp32 log-softmax kept at its top 256 ids and the targets only. compare() and
    compare_full() read a row's own top 20, the targets and (on the bf16 naive row) the fp32 truth's top 20;
    holding three lists of full-vocabulary rows for the whole battery (--items) would add ~0.7 GB. Reading an
    id that was not kept raises KeyError, so nothing is compared that was not stored. The top 20 and the
    argmax are taken from the full row, so ties break as they did on it."""

    def __init__(self, lp: torch.Tensor, targets: list[int], k: int = 256):
        self.top20, self.arg = lp.topk(20), lp.argmax()
        ids = list(dict.fromkeys(self.top20.indices.tolist() + lp.topk(k).indices.tolist() + list(targets)))
        self.pos = {t: n for n, t in enumerate(ids)}
        self.vals = lp[ids].clone()

    def topk(self, n: int):
        assert n == 20, n
        return self.top20

    def argmax(self) -> torch.Tensor:
        return self.arg

    def __getitem__(self, key):
        return self.vals[self.pos[key]] if isinstance(key, int) else self.vals[[self.pos[t] for t in key]]


@torch.no_grad()
def naive(eng: HFEngine, g: Group, vec: torch.Tensor, coef: float, targets: list[int], shift=(0, 0)) -> list:
    """Each suffix forwarded alone after the prefix, steered by a plain hook at its span moved by `shift`."""
    out = []
    head = eng.head.weight
    tgt = torch.tensor(targets)
    for i, s in enumerate(g.suffixes):
        a, b = g.steers[0].suffix_spans[i]
        lo, hi = len(g.prefix) + a + shift[0], len(g.prefix) + b + shift[1]
        v = (vec.float() * coef).to(eng.dtype)

        def hook(module, args, output, lo=lo, hi=hi, v=v):
            h = output[0] if isinstance(output, tuple) else output
            h[0, lo:hi] += v

        hd = eng.layers[LAYER].register_forward_hook(hook)
        eng._phase = None
        try:
            hs = eng.inner(input_ids=torch.tensor([g.prefix + s])).last_hidden_state[0, -1]
        finally:
            hd.remove()
        lp = torch.log_softmax(torch.nn.functional.linear(hs, head).float(), -1)
        tl = torch.nn.functional.linear(hs.float(), head[tgt].float())
        out.append((Kept(lp, targets), tl))
    return out


def compare(ro: list, ref: list, targets: list[int]) -> tuple[float, float, int]:
    """max |dlogprob| over the reference's top 20 and the targets, max |d target logit|, top-1 agreements."""
    dlp, dtl, agree = 0.0, 0.0, 0
    for e, (lp, tl) in zip(ro, ref):
        top = lp.topk(20).indices
        idx = {int(t): k for k, t in enumerate(e.top_ids)}
        common = [t for t in top.tolist() if t in idx]
        d = [abs(float(lp[t]) - float(e.top_logprobs[idx[t]])) for t in common]
        d += [abs(float(lp[t]) - float(e.target_logprobs[k])) for k, t in enumerate(targets)]
        dlp = max(dlp, max(d))
        dtl = max(dtl, float((tl - e.target_logits).abs().max()))
        agree += int(e.top_ids[0]) == int(top[0])
    return dlp, dtl, agree


def compare_full(a: list, ref: list, targets: list[int]) -> tuple[float, float, int]:
    """compare() for two naive forwards: the same metrics, `a` read at ref's top 20 and the targets."""
    dlp, dtl, agree = 0.0, 0.0, 0
    for (lp, tl), (lr, tr) in zip(a, ref):
        ids = lr.topk(20).indices.tolist() + list(targets)
        dlp = max(dlp, float((lp[ids] - lr[ids]).abs().max()))
        dtl = max(dtl, float((tl - tr).abs().max()))
        agree += int(lp.argmax()) == int(lr.argmax())
    return dlp, dtl, agree


def same_bits(x: list, y: list) -> bool:
    return all(torch.equal(a.top_ids, b.top_ids) and torch.equal(a.top_logprobs, b.top_logprobs)
               and torch.equal(a.target_logprobs, b.target_logprobs) and torch.equal(a.target_logits, b.target_logits)
               for a, b in zip(x, y)) and len(x) == len(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--items", default=None, help="an items_s2.jsonl; default: the built-in sample")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model = args.model or snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True)
    torch.manual_seed(0)
    check_spans_helper()

    items = ([json.loads(x) for x in Path(args.items).read_text().splitlines() if x.strip()]
             if args.items else sample_items())
    file_tail = None
    if args.items:
        # runs/s2_battery.py rows: keep the Qwen rows (own-name rows are per model), and take the
        # subjectless span from the file (its steer_tail and span_start for qwen3-32b; the 0.6B and 32B
        # tokenizer files are byte-identical)
        items = [it for it in items if "qwen3-32b" in it.get("models", ["qwen3-32b"])]
        tails = {it["steer_tail"]["qwen3-32b"] for it in items if not it.get("subject")}
        assert len(tails) == 1, tails
        file_tail = tails.pop()
        print(f"items: {len(items)} Qwen rows from {args.items}, subjectless tail {file_tail} from the file")
    tok = AutoTokenizer.from_pretrained(model)
    r = Qwen3Renderer(tok, enable_thinking=False)
    first = r.answer_tokens(["Yes", "No"])
    targets = [first["Yes"], first["No"]]
    # subjectless items get the median subject span, as the spec asks
    probe = post_subject_group(r, "probe", OPENING, [it for it in items if it.get("subject")], "v", 0.0, LAYER)
    tail = file_tail or int(statistics.median(b - a for a, b in probe.steers[0].suffix_spans))

    truth, coef = None, None
    for dtype in (torch.float32, torch.bfloat16):
        name = "fp32" if dtype == torch.float32 else "bf16"
        eng = HFEngine(model, device="cpu", dtype=dtype)
        H = eng.c.hidden_size
        unit = torch.randn(H, generator=torch.Generator().manual_seed(0))
        unit = unit / unit.norm()
        if coef is None:
            # the dose: one mean residual norm at LAYER over the prefix (fp32), so a misplaced token shows
            from kimisroom.engine import Collect
            g0 = post_subject_group(r, "s", OPENING, items, "v", 0.0, LAYER, tail=tail)
            nrm = eng.run([Group("n", g0.prefix, collects=[Collect(LAYER, 0, len(g0.prefix))], readout="none")])
            coef = float(nrm["n"]["collect_norms"][0])
        g = post_subject_group(r, "s", OPENING, items, "v", coef, LAYER, tail=tail)
        lens = [len(s) for s in g.suffixes]
        chunks = suffix_chunks(lens, eng.suffix_chunk)
        # the built-in sample spans two buckets and ends in a partial chunk; the real battery is <= 32 tokens
        # by construction (one bucket) and may be a multiple of the chunk size, so it only reports both
        assert args.items or (len({Ls for Ls, _ in chunks}) >= 2 and any(
            len(c) < eng.suffix_chunk for _, c in chunks)), "the items should span two buckets and leave a partial chunk"
        if args.items:
            print(f"   {name}: buckets {sorted({Ls for Ls, _ in chunks})}, {len(chunks)} chunks, "
                  f"partial chunk: {any(len(c) < eng.suffix_chunk for _, c in chunks)}")
            for it, (a, b) in zip(items, g.steers[0].suffix_spans):
                assert it.get("subject") or it["span_start"]["qwen3-32b"] == a, (it["id"], a, it["span_start"])
                assert b - a == it["tokens"]["qwen3-32b"]["steered"] and b == it["tokens"]["qwen3-32b"]["suffix"], it["id"]
        vec = {"v": unit}

        ro, hf_pos = hf_masked_positions(eng, g, vec, targets)
        if dtype == torch.float32:
            check_positions(tok, r, g, items, hf_pos)
        ref = naive(eng, g, unit, coef, targets)
        if dtype == torch.float32:
            truth = ref
            early = naive(eng, g, unit, coef, targets, shift=(-1, 0))
            no_read = naive(eng, g, unit, coef, targets, shift=(0, -1))
            dlp, dtl, agree = compare(ro, ref, targets)
            e_dlp, _, _ = compare(ro, early, targets)
            n_dlp, _, _ = compare(ro, no_read, targets)
            print(f"3. fp32: coef {coef:.1f}; engine vs naive max |dlogprob| {dlp:.2e}, max |d logit32| {dtl:.2e}, "
                  f"top-1 {agree}/{len(items)}; wrong refs: one token early {e_dlp:.2e}, readout unsteered {n_dlp:.2e}")
            assert dlp < 2e-3 and dtl < 2e-3 and agree == len(items), "fp32 engine and naive forward disagree"
            assert e_dlp > 100 * dlp and n_dlp > 100 * dlp, "the check cannot tell a shifted span from the right one"
        else:
            e_lp, e_tl, e_top = compare(ro, truth, targets)
            n_lp, n_tl, n_top = compare_full(ref, truth, targets)
            print(f"3. bf16 against the fp32 naive forward: engine max |dlogprob| {e_lp:.2f}, max |d logit32| "
                  f"{e_tl:.2f}, top-1 {e_top}/{len(items)}; bf16 naive {n_lp:.2f}, {n_tl:.2f}, {n_top}/{len(items)}")
            assert e_lp <= 2 * n_lp + 0.01 and e_tl <= 2 * n_tl + 0.01, "bf16 engine is noisier than bf16 itself"

        # 4. zero vector and the shared-range form
        if dtype == torch.bfloat16:
            plain = eng.run([Group("s", g.prefix, g.suffixes)], vec, targets, top_k=20)["s"]["readouts"]
            zero = eng.run([post_subject_group(r, "s", OPENING, items, "v", 0.0, LAYER, tail=tail)], vec, targets,
                           top_k=20)["s"]["readouts"]
            assert same_bits(plain, zero), "a zero vector changed the readout"
            a0 = 3  # every suffix starts with the same user header, so a shared range is well defined
            shared = eng.run([Group("s", g.prefix, g.suffixes, steers=[Steer(LAYER, "v", coef, suffix=(a0, None))])],
                             vec, targets, top_k=20)["s"]["readouts"]
            per = eng.run([Group("s", g.prefix, g.suffixes, steers=[Steer(LAYER, "v", coef, suffix_spans=[
                (a0, len(s)) for s in g.suffixes])])], vec, targets, top_k=20)["s"]["readouts"]
            assert same_bits(shared, per), "suffix=(a, None) and the equivalent per-suffix spans differ"
            print("4. bf16: zero vector bit-exact; shared range == per-suffix spans, bit-exact")
            other = Group("o", g.prefix[:40], g.suffixes[:5])
            both = eng.run([other, g], vec, targets, top_k=20)
            assert same_bits(ro, both["s"]["readouts"]), "the steered group changed when batched"
            print("5. bf16: steered group alone == batched after another group, bit-exact")

        # 6. the fp32 two-row readout against the full log-softmax
        worst = 0.0
        for e in ro:
            y32 = float(e.target_logits[0] - e.target_logits[1])
            yfull = float(e.target_logprobs[0] - e.target_logprobs[1])
            bound = 1e-4 if dtype == torch.float32 else float(e.target_logits.abs().sum()) * 2 ** -8
            worst = max(worst, abs(y32 - yfull) / bound)
            assert abs(y32 - yfull) <= bound, f"{name}: fp32 y {y32} vs full-readout y {yfull}"
        print(f"6. {name}: fp32 y vs full-readout y within bound (worst {worst:.2f} of the bound)")
        del eng

    print("all checks passed")


if __name__ == "__main__":
    main()
