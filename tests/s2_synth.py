"""Synthetic Study 2 readouts in the runner format that runs/s2_table.py documents (for tests only).

    .venv/bin/python tests/s2_synth.py out.csv [--phase confirm] [--models qwen,k26] [--beta qwen=0.25,k26=0]
        [--n-args 7] [--n-wordings 3] [--squeeze-only] [--seed 1]

Data model, per readout (sign as in the spec, s0 = unsteered sign x y0 depth):
    y0 = sign x s0(item, opening)
    y  = a_v y0 + c_v + sign x (m_v [emotion items] + [subject you] x self(v, opening, para)) + noise
a_v is the vector's squeeze (emotions around 0.6, every other class around 0.85) and c_v its Yes lean, shared by
every item, so the reference fit removes them exactly. m_v is a meaning push on the emotion items that is the
same for every subject. self() is the planted self-specific term, which d = r(you) - r(language models)
recovers: beta for emotion vectors, a random intercept per non-emotion vector, and small argument, paraphrase and
vector:argument terms. --squeeze-only sets every push and random term to 0, so any d is compression bias.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

EMO = {"qwen": ["joy", "sadness", "anger", "fear", "euphoria", "surprise"], "k26": ["joy", "fear", "euphoria", "surprise"]}
LAYER = {"qwen": 40, "k26": 20}
DOSE = {"qwen": 2.0, "k26": 4.0}
NAME = {"qwen": "Qwen", "k26": "Kimi"}
CAP_S0 = {"you": -20.0, "language models": -18.0, "other language models": -17.0, "own name": -19.0,
          "rocks": -24.0, "humans": 18.0}
MP = ["visual_search", "route_planning", "digit_memorising", "distance_estimation", "mental_rotation",
      "voice_in_noise", "weighing_options", "room_recall"]


def items(model: str, phase: str) -> list[dict]:
    rng = random.Random(7)
    out = []

    CAP_T = [("Can {S} feel emotions?", "Are {S} unable to feel emotions?"), ("Do {S} have emotions?", "Do {S} lack emotions?"),
             ("Do {S} experience emotions?", "Do {S} lack any experience of emotion?"), ("Can {S} have feelings?", "Are {S} unable to have feelings?")]

    def add(kind, para, subj, s0p, sign_pos, stem, nst):
        for twin, sg in (("pos", sign_pos), ("neg", -sign_pos)):
            text = ""
            if kind == "s2_capacity":
                text = CAP_T[(int(para) - 1) % 4][twin == "neg"].format(S=subj) + " Answer with one word, Yes or No."
            # the stored answer is the unsteered direction (y0 = sign x s0); emotion items have none
            ans = "" if kind in ("s2_capacity", "s2_state") else ("Yes" if sg * s0p > 0 else "No")
            out.append(dict(item=f"{stem}.{twin}", pair_id=stem, kind=kind, para=str(para), twin=twin,
                            subject=subj, sign=sg, s0=s0p, n_steered=nst, text=text, correct_answer=ans))
    caps = ["you", "language models", "other language models", "own name", "rocks", "humans"]
    for p in range(1, 9):
        for s in caps:
            add("s2_capacity", p, NAME[model] if s == "own name" else s, CAP_S0[s], 1, f"cap{p}.{s[:5]}", 9 + p)
        for s in ("you", "language models"):
            add("s2_state", p, s, CAP_S0[s] + 4, 1, f"st{p}.{s[:5]}", 11 + p)
    for k in range(1, 7):
        for s in ("you", "language models", "rocks", "humans"):
            add("s2_placebo", f"P0{k}", s, rng.uniform(5, 12), rng.choice((1, -1)), f"plc{k}.{s[:5]}", 8 + k)
    for s, sg in (("you", 1), ("language models", 1), ("rocks", -1), ("humans", -1)):
        add("s2_persona", "P20", s, 10.0, sg, f"per.{s[:5]}", 9)
    for k, conf in enumerate((3, 6, 9, 12, 15, 18, 21, 25), start=1):
        for s in ("rocks", "humans"):
            add("s2_reference", f"R0{k}", s, conf + rng.uniform(-1, 1), rng.choice((1, -1)), f"ref{k}.{s[:5]}", 10 + k)
    if phase == "pilot":
        for k in range(1, 21):
            add("yes_bias", f"yes_bias.{k:02d}", "", 6.0, 1, f"yes_bias.{k:02d}", 12)
    for i, it in enumerate(out):
        it["bucket"], it["chunk"] = str(i // 32), str(i % 7)
    return out


def gate_item(it: dict) -> bool:
    return it["kind"] in ("yes_bias", "s2_reference") or (it["kind"] in ("s2_placebo", "s2_persona") and it["subject"] in ("rocks", "humans"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--phase", default="confirm", choices=("pilot", "confirm"))
    ap.add_argument("--models", default="qwen,k26")
    ap.add_argument("--arms", default="S2-DEN,S2-CON,S2-FTD,S2-FTC")
    ap.add_argument("--beta", default="qwen=0.25,k26=0", help="emotion self effect per model (S2-DEN, S2-FTD)")
    ap.add_argument("--beta-con", type=float, default=0.0, help="emotion self effect in S2-CON and S2-FTC")
    ap.add_argument("--beta-mp", type=float, default=0.0, help="mental-process self effect")
    ap.add_argument("--n-args", type=int, default=7)
    ap.add_argument("--n-wordings", type=int, default=3)
    ap.add_argument("--n-nouns", type=int, default=12)
    ap.add_argument("--n-mp", type=int, default=8)
    ap.add_argument("--sd-vec", type=float, default=0.06)
    ap.add_argument("--sd-small", type=float, default=0.03)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--squeeze-only", action="store_true")
    ap.add_argument("--fail-vec", default="", help="model:vec that fails the rule 1 gates (pilot)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    G = lambda sd: rng.gauss(0, sd) if sd > 0 and not a.squeeze_only else 0.0  # noqa: E731
    beta = {k: float(v) for k, v in (kv.split("=") for kv in a.beta.split(","))}
    noise = a.noise  # kept under --squeeze-only (pass --noise 0 for an exact squeeze)
    rows = []
    for model in a.models.split(","):
        its = items(model, a.phase)
        vecs = ([(e, "emotion") for e in EMO[model]] + [(f"N{k:02d}", "noun") for k in range(1, a.n_nouns + 1)] +
                [("arithmetic", "arithmetic"), ("random", "noise")] + [(f"mp:{m}", "thinking") for m in MP[:a.n_mp]])
        squeeze = {v: (rng.gauss(0.6, 0.02 * (not a.squeeze_only)) if c == "emotion" else rng.gauss(0.85, 0.02 * (not a.squeeze_only)))
                   for v, c in vecs + [("red", "bridge"), ("ocean", "bridge")]}
        lean = {v: (0.4 if a.squeeze_only else rng.gauss(0.4, 0.3)) for v in squeeze}
        uvec = {v: (G(a.sd_vec) if c != "emotion" else 0.0) for v, c in vecs + [("red", "bridge"), ("ocean", "bridge")]}
        arms = [x for x in a.arms.split(",") if model == "qwen" or x in ("S2-DEN", "S2-CON")]
        doses = [DOSE[model], DOSE[model] / 2] if (a.phase == "pilot" and model == "qwen") else [DOSE[model]]
        for arm in arms:
            pre = "DA" if arm == "S2-DEN" else "CA"
            b = 0.0 if a.squeeze_only else (beta[model] if arm in ("S2-DEN", "S2-FTD") else a.beta_con)
            arm_vecs = vecs + ([("red", "bridge"), ("ocean", "bridge")] if arm == "S2-DEN" and a.phase == "confirm" else [])
            for ai in range(1, a.n_args + 1):
                arg = f"{pre}{ai:02d}"
                kind = "control" if pre == "CA" else ("experience-general" if ai in (7, 8, 10) else "emotion-specific")
                u_arg, u_argx = G(a.sd_small), G(a.sd_small)
                nw = 1 if a.phase == "pilot" else a.n_wordings
                if a.phase == "pilot" and ai <= 2:
                    nw = 2
                for w in range(nw):
                    slot = w if a.phase == "confirm" else (6 - w)
                    role = "heldout" if a.phase == "confirm" else ("pilot" if w == 0 else "pilot2")
                    opening = f"{arg}-w{slot}"
                    shift = G(1.0)
                    u_aw = G(a.sd_small)
                    y0 = {it["item"]: it["sign"] * (it["s0"] + shift + G(0.5)) for it in its}
                    meta = dict(model=model, phase=a.phase, arm=arm, argument=arg, arg_kind=kind, wording=slot,
                                wording_role=role, opening=opening, layer=LAYER[model], site="post_subject",
                                reply_len=rng.randint(150, 400), emo_words=rng.randint(0, 30) if pre == "DA" else rng.randint(0, 3),
                                ckpt={"S2-FTD": "ftd", "S2-FTC": "ftc"}.get(arm, "base"))
                    for it in its:
                        rows.append(dict(meta, vec="none", vclass="none", dose=0.0, **{k: it[k] for k in
                                    ("item", "kind", "para", "pair_id", "twin", "subject", "sign", "bucket", "chunk", "text", "correct_answer")},
                                    y=y0[it["item"]], y0=y0[it["item"]], mass=0.98, top1="Yes" if y0[it["item"]] > 0 else "No",
                                    n_steered=0))
                    for dose in doses:
                        for v, c in arm_vecs:
                            if a.phase == "pilot" and c == "bridge":
                                continue
                            x = c == "emotion"
                            u_va = G(a.sd_small)
                            selfp = {}
                            for p in range(1, 9):
                                selfp[str(p)] = ((b if x else (a.beta_mp if c == "thinking" else 0.0)) + uvec[v] + u_va +
                                                 G(a.sd_small) + u_arg + x * u_argx + u_aw)
                            m_v = (1.0 if x else 0.1) * (not a.squeeze_only)
                            fail = a.fail_vec == f"{model}:{v}" and dose == DOSE[model]
                            for it in its:
                                if a.phase == "pilot" and (model == "k26" or x) and not gate_item(it):
                                    continue
                                push = 0.0
                                if it["kind"] in ("s2_capacity", "s2_state"):
                                    push = m_v + (selfp[it["para"]] if it["subject"] == "you" else 0.0)
                                yy = squeeze[v] * y0[it["item"]] + lean[v] + it["sign"] * push + (rng.gauss(0, noise) if noise > 0 else 0.0)
                                top1 = ("Yes" if yy > 0 else "No") if not fail else "The"
                                rows.append(dict(meta, vec=v, vclass=c, dose=dose, **{k: it[k] for k in
                                            ("item", "kind", "para", "pair_id", "twin", "subject", "sign", "bucket", "chunk", "text", "correct_answer")},
                                            y=yy, y0=y0[it["item"]], mass=0.6 if fail else 0.97, top1=top1,
                                            n_steered=it["n_steered"]))
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{a.out}: {len(rows):,} readouts")


if __name__ == "__main__":
    main()
