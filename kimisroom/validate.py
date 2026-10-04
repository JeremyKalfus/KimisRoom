"""Independent tests of the emotion vectors, on data we did not generate.

AIPsy-Affect (Keeman 2026, arXiv 2604.23719): human-written, keyword-free, third-person
vignettes, each peak-emotion item paired with a neutral twin that keeps its characters, setting
and length. The set is never used to build or tune a vector, so a vector that passes cannot be an
artifact of our generator. It covers six of our eight emotions through Plutchik peaks
(`emotions.AIPSY_LABEL`); calm and pain have no peak in it and are not gated on it at all --
`validate` skips them instead of raising, and they are reported on held-out AUROC only.

Pain Axis (Tagliabue, Dung and Berg 2026, arXiv 2609.16247, MIT, pinned in `data/pain_axis/`) is
pain's independent set: short cue sentences ending "I feel:", five pain categories (physical,
psychological, social, moral, cognitive) against five control categories that hold valence and
arousal fixed -- fear, negative emotion, negative world state, neutral and non-painful body
sensation -- plus separate high-arousal, numbed-injury, sadness and random sets. Set S1 matches
the syntactic frame across categories within a `set` index, which gives matched pairs analogous to
AIPsy's twins; set S2 is naturalistic and is the paper's primary measure.

Each unit is read as plain text (like the extraction stories). AIPsy vignettes get one Collect per
layer over the whole vignette; Pain Axis sentences are read at their final token by default, which
is where the paper measures ("I feel:" is a cue, and the sentences are 7-12 words long).
Per vector and layer:
- pair_acc: share of matched pairs where the emotional item outscores its control
- auroc: the emotion's items against the baseline items
- specific: the emotion's items against the other studied emotions' items
- moderate (AIPsy): moderate-intensity items against the complex neutrals
- p_pair, p_auroc: share of 500 isotropic random directions that do at least as well. Needed
  because the pairs are correlated (all twins differ from their items the same broad way), so a
  random direction can reach pair_acc 0.3 or 0.7 by chance; 0.5 is not the right yardstick.
"""

from __future__ import annotations

import json

import torch

from .emotions import AIPSY_LABEL, EMOTIONS, emotion_of_vector
from .engine import Collect, Group
from .trials import ROOT, Renderer, load_jsonl, story_group

AIPSY = ROOT / "data/aipsy_affect"
PAIN_AXIS = ROOT / "data/pain_axis"
N_NULL = 500


def auroc(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """P(random pos > random neg) along the last dim, ties count half (Mann-Whitney U / n_pos n_neg)."""
    d = pos[..., :, None] - neg[..., None, :]
    return (d > 0).float().mean((-1, -2)) + 0.5 * (d == 0).float().mean((-1, -2))


def _null_dirs(n_layers: int, H: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    null = torch.randn(n_layers, H, N_NULL, generator=gen)
    return null / null.norm(dim=1, keepdim=True)


# ---------------------------------------------------------------- AIPsy-Affect


def aipsy_units() -> list[dict]:
    return [r for split in ("clinical", "neutral", "moderate", "complex_neutral") for r in load_jsonl(AIPSY / f"{split}.jsonl")]


def aipsy_groups(r: Renderer, layers: range) -> list[Group]:
    return [story_group(r, f"aipsy:{u['id']}", u["text"], layers) for u in aipsy_units()]


def validate(results: dict, vectors: dict[str, torch.Tensor], emotion_of: dict[str, str] | None = None,
             emotions: list[str] | None = None) -> dict[str, list[dict]]:
    """vectors: name -> (n_layers, H). emotion_of maps a vector name to the emotion (ours) it is
    scored against, default the vector's own emotion; a random vector mapped to "joy" gives the
    chance row. A vector whose emotion has no AIPsy-Affect label (calm, pain) is skipped, not
    raised on. `emotions` is the active emotion list, which fixes the `specific` comparison set:
    passing the pilot's four reproduces the pilot's specificity numbers exactly."""
    emotions = emotions if emotions is not None else EMOTIONS
    labels = {AIPSY_LABEL[e] for e in emotions if e in AIPSY_LABEL}
    units = {u["id"]: u for u in aipsy_units()}
    acts = {i: torch.stack(results[f"aipsy:{i}"]["collects"]) for i in units}  # (n_layers, H) each

    def ids(split, emotion=None):
        return [i for i, u in units.items() if u["split"] == split and (emotion is None or u["emotion"] == emotion)]

    n_layers, H = next(iter(vectors.values())).shape
    null = _null_dirs(n_layers, H)
    out = {}
    for name, vec in vectors.items():
        emo = (emotion_of or {}).get(name, emotion_of_vector(name))
        if emo not in AIPSY_LABEL:
            continue  # calm, pain: no Plutchik peak in AIPsy-Affect; held-out AUROC only
        label = AIPSY_LABEL[emo]
        unit = (vec / vec.norm(dim=-1, keepdim=True)).unsqueeze(-1)  # (n_layers, H, 1)

        def score(id_list, dirs):  # -> (n_layers, n_dirs, n_items)
            a = torch.stack([acts[i] for i in id_list], dim=1)  # (n_layers, n, H)
            return torch.einsum("lnh,lhd->ldn", a, dirs)

        clin = ids("clinical", label)
        twins = [units[i]["matched_control_id"] for i in clin]
        others = [i for e in labels if e != label for i in ids("clinical", e)]
        sets = {"clin": clin, "twin": twins, "neu": ids("neutral"), "oth": others,
                "mod": ids("moderate", label), "cx": ids("complex_neutral")}
        v = {k: score(x, unit)[:, 0] for k, x in sets.items()}  # (n_layers, n)
        r = {k: score(sets[k], null) for k in ("clin", "twin", "neu")}  # (n_layers, N_NULL, n)
        pair, pair_null = (v["clin"] > v["twin"]).float().mean(-1), (r["clin"] > r["twin"]).float().mean(-1)
        roc, roc_null = auroc(v["clin"], v["neu"]), auroc(r["clin"], r["neu"])
        spec, mod = auroc(v["clin"], v["oth"]), auroc(v["mod"], v["cx"])
        out[name] = [
            {
                "layer": L,
                "pair_acc": pair[L].item(),
                "p_pair": (pair_null[L] >= pair[L]).float().mean().item(),
                "auroc": roc[L].item(),
                "p_auroc": (roc_null[L] >= roc[L]).float().mean().item(),
                "specific": spec[L].item(),
                "moderate": mod[L].item(),
            }
            for L in range(n_layers)
        ]
    return out


def report(table: dict[str, list[dict]], layers: list[int]) -> str:
    lines = [f"{'vector':18s} {'layer':>5s} {'pair_acc':>8s} {'p':>5s} {'auroc':>6s} {'p':>5s} {'specific':>8s} {'moderate':>8s}"]
    for name, rows in table.items():
        for L in layers:
            r = rows[L]
            lines.append(f"{name:18s} {L:5d} {r['pair_acc']:8.2f} {r['p_pair']:5.3f} {r['auroc']:6.2f} {r['p_auroc']:5.3f} "
                         f"{r['specific']:8.2f} {r['moderate']:8.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------- Pain Axis (pain's own gate)

PAIN_CATEGORIES = ("A1", "A2", "A3", "A4", "A5")  # physical, psychological, social, moral, cognitive
CONTROL_CATEGORIES = ("B", "C1", "C2", "D", "E")  # fear, negative emotion, negative world state, neutral, body
PAIN_PERSPECTIVES = ("1P", "3P")


def pain_units(perspectives: tuple[str, ...] = PAIN_PERSPECTIVES) -> list[dict]:
    """Every Pain Axis cue sentence as {id, dataset, perspective, category, set, role, text}.

    `role` is "pain" for the A categories, "control" for the B/C/D/E categories, and the dataset's
    own name for the auxiliary sets (arousal, numb, random, sadness).
    """
    main = json.loads((PAIN_AXIS / "3.1_pain_and_control_datasets.json").read_text())["datasets"]
    sad = json.loads((PAIN_AXIS / "3.1_sadness_dataset.json").read_text())["datasets"]
    role = {"Random": "random", "Arousal": "arousal", "Numb": "numb", "SD_sadness": "sadness"}
    out = []
    for ds, block in list(main.items()) + list(sad.items()):
        stem, _, persp = ds.rpartition("_")
        if persp not in perspectives:
            continue
        for s in block["sentences"]:
            cat = s["category"]
            out.append({"id": f"pain:{ds}:{cat}:{s['set']}", "dataset": stem, "perspective": persp,
                        "category": cat, "set": s["set"], "text": s["prompt"],
                        "role": role.get(stem, "pain" if cat in PAIN_CATEGORIES else "control")})
    assert len({u["id"] for u in out}) == len(out), "Pain Axis unit ids are not unique"
    return out


def pain_groups(r: Renderer, layers: range, readout: str = "last",
                perspectives: tuple[str, ...] = PAIN_PERSPECTIVES) -> list[Group]:
    """One group per cue sentence. readout "last": the final token, where the paper measures
    ("I feel:" is the cue); "mean": the mean over every token, as the extraction stories are read."""
    assert readout in ("last", "mean")
    groups = []
    for u in pain_units(perspectives):
        ids = r.ids(u["text"])
        start = len(ids) - 1 if readout == "last" else 0
        groups.append(Group(u["id"], ids, collects=[Collect(L, start, len(ids)) for L in layers], readout="none"))
    return groups


def validate_pain(results: dict, vectors: dict[str, torch.Tensor],
                  perspectives: tuple[str, ...] = PAIN_PERSPECTIVES) -> dict[tuple[str, str], list[dict]]:
    """Pain's AIPsy analogue, per (vector, perspective).

    - pair_acc: S1 pairs a pain item against a control item sharing its `set`, hence its syntactic
      frame (5 pain x 5 control categories x 20 sets = 500 pairs), the matched-pair test
    - auroc: S2 (naturalistic, the paper's primary) pain items against S2 control items
    - p_pair, p_auroc: against 500 isotropic random directions, as for AIPsy
    - vs_fear / vs_neg / vs_body: pain against the fear (B), negative (C1+C2) and non-painful body
      sensation (E) controls, which are what the set holds fixed against the pain signal
    - vs_sadness / vs_arousal / vs_numb / vs_random: pain against the auxiliary sets (low mood,
      high-arousal non-pain, injury without nociception, neutral baseline)
    """
    units = {u["id"]: u for u in pain_units(perspectives)}
    acts = {i: torch.stack(results[i]["collects"]) for i in units}
    n_layers, H = next(iter(vectors.values())).shape
    null = _null_dirs(n_layers, H)

    def ids(persp, **where):
        return [i for i, u in units.items() if u["perspective"] == persp
                and all(u[k] in (v if isinstance(v, tuple) else (v,)) for k, v in where.items())]

    def score(id_list, dirs):  # -> (n_layers, n_dirs, n_items)
        a = torch.stack([acts[i] for i in id_list], dim=1)
        return torch.einsum("lnh,lhd->ldn", a, dirs)

    out = {}
    for name, vec in vectors.items():
        unit = (vec / vec.norm(dim=-1, keepdim=True)).unsqueeze(-1)
        for persp in perspectives:
            # S1 matched pairs: every (pain, control) pair inside a `set`, which fixes the frame
            s1_pain = ids(persp, dataset="S1", role="pain")
            s1_ctrl = ids(persp, dataset="S1", role="control")
            by_set: dict[int, tuple[list[str], list[str]]] = {}
            for i in s1_pain:
                by_set.setdefault(units[i]["set"], ([], []))[0].append(i)
            for i in s1_ctrl:
                by_set.setdefault(units[i]["set"], ([], []))[1].append(i)
            left = [p for s in sorted(by_set) for p in by_set[s][0] for _ in by_set[s][1]]
            right = [c for s in sorted(by_set) for _ in by_set[s][0] for c in by_set[s][1]]
            sets = {
                "s2_pain": ids(persp, dataset="S2", role="pain"),
                "s2_ctrl": ids(persp, dataset="S2", role="control"),
                "fear": ids(persp, dataset=("S1", "S2", "ControlSupplement"), category="B"),
                "neg": ids(persp, dataset=("S1", "S2", "ControlSupplement"), category=("C1", "C2")),
                "body": ids(persp, dataset=("S1", "S2", "ControlSupplement"), category="E"),
                "pain_all": ids(persp, dataset=("S1", "S2"), role="pain"),
                "phys": ids(persp, dataset=("S1", "S2"), category="A1"),
                "sadness": ids(persp, role="sadness"),
                "arousal": ids(persp, role="arousal"),
                "numb": ids(persp, role="numb"),
                "random": ids(persp, role="random"),
            }
            v = {k: score(x, unit)[:, 0] for k, x in sets.items() if x}
            pl, pr = score(left, unit)[:, 0], score(right, unit)[:, 0]
            pair = (pl > pr).float().mean(-1)
            pair_null = (score(left, null) > score(right, null)).float().mean(-1)
            roc = auroc(v["s2_pain"], v["s2_ctrl"])
            roc_null = auroc(score(sets["s2_pain"], null), score(sets["s2_ctrl"], null))

            def against(key):
                return auroc(v["pain_all"], v[key]) if key in v else torch.full((n_layers,), float("nan"))

            vs_numb = auroc(v["phys"], v["numb"]) if "numb" in v else torch.full((n_layers,), float("nan"))
            out[name, persp] = [
                {
                    "layer": L, "n_pairs": len(left),
                    "pair_acc": pair[L].item(),
                    "p_pair": (pair_null[L] >= pair[L]).float().mean().item(),
                    "auroc": roc[L].item(),
                    "p_auroc": (roc_null[L] >= roc[L]).float().mean().item(),
                    "vs_fear": against("fear")[L].item(),
                    "vs_neg": against("neg")[L].item(),
                    "vs_body": against("body")[L].item(),
                    "vs_sadness": against("sadness")[L].item(),
                    "vs_arousal": against("arousal")[L].item(),
                    "vs_numb": vs_numb[L].item(),
                    "vs_random": against("random")[L].item(),
                }
                for L in range(n_layers)
            ]
    return out


PAIN_COLS = ("pair_acc", "p_pair", "auroc", "p_auroc", "vs_fear", "vs_neg", "vs_body",
             "vs_sadness", "vs_arousal", "vs_numb", "vs_random")


def pain_report(table: dict[tuple[str, str], list[dict]], layers: list[int]) -> str:
    """pair_acc: S1 matched pairs. auroc: S2 pain vs S2 controls. The rest are AUROCs of all pain
    items against that comparison set; vs_numb uses the physical-pain items only."""
    head = f"{'vector':18s} {'p':>3s} {'layer':>5s} " + " ".join(f"{c:>9s}" for c in PAIN_COLS)
    lines = [head]
    for (name, persp), rows in table.items():
        for L in layers:
            r = rows[L]
            lines.append(f"{name:18s} {persp:>3s} {L:5d} " + " ".join(f"{r[c]:9.3f}" for c in PAIN_COLS))
    return "\n".join(lines)


def pain_gate(table: dict[tuple[str, str], list[dict]], layer: int, alpha: float = 0.05) -> dict[tuple[str, str], bool]:
    """The same gate shape as AIPsy's: both pair accuracy and AUROC beat the random null."""
    return {k: (rows[layer]["p_pair"] < alpha and rows[layer]["p_auroc"] < alpha) for k, rows in table.items()}
