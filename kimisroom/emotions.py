"""The emotion list and everything keyed by it: identification option words, the AIPsy-Affect
label map, the vector naming scheme, and the dose cap on pain.

Single source of truth. Nothing else in the repo may hardcode an emotion list; every stage takes
the active list from `emotions_from_argv` so that a run over the old four emotions reproduces the
old results exactly (`--emotions joy,sadness,anger,fear`, or a stimuli directory that only has
those four conditions).

Vector names: "joy" (emotion minus neutral), "joy-specific" (the same minus the mean of the active
emotion vectors) and "joy-dn" (denoised, PLAN.md section 10 / Sofroniew et al. 2026). Controls keep
their concept name ("red") plus "random".

Note on "-specific": subtracting the mean of N emotion vectors leaves the vector's own weight at
1 - 1/N, so at N = 8 the algebra keeps 7/8 of the raw vector where at N = 4 it kept 3/4, and the
correction it subtracts is spread over more emotions. How much that actually changes the vector
depends on how parallel the raw vectors are, not on 1 - 1/N alone: on the pilot's Qwen vectors
(N = 4, where the four raw emotion vectors are mutually at cosine 0.69-0.78) cos(raw, specific) is
only 0.27-0.55, because almost all of what is removed is the component they share. As N grows the
own-weight rises while the shared component that gets removed stays much the same, so "-specific"
drifts towards the raw vector and the contrast it was built for weakens; the 4-emotion and
8-emotion "-specific" vectors are not the same object. Kept for continuity with the pilot and
reported alongside; "-dn" is the principled replacement -- it removes the directions that actually
carry the shared variance rather than the mean of whatever emotions happen to be in the set.
"""

from __future__ import annotations

# canonical order, used for option order, Latin squares and every table
EMOTIONS: list[str] = ["joy", "sadness", "anger", "fear", "euphoria", "calm", "pain", "surprise"]
EMOTIONS_V1: list[str] = ["joy", "sadness", "anger", "fear"]  # the pilot's four, for reproducibility
CONCEPTS: list[str] = ["red", "ocean", "arithmetic"]  # control concepts, built the same way

NONE_OPTION = "None"


def option_word(emotion: str) -> str:
    """The identification option word for an emotion ("joy" -> "Joy")."""
    return emotion.capitalize()


def emotion_of_option(word: str) -> str:
    return word.lower()


def id_options(emotions: list[str] | None = None) -> list[str]:
    """Identification answer options in canonical order: the emotions, then None."""
    return [option_word(e) for e in (emotions if emotions is not None else EMOTIONS)] + [NONE_OPTION]


# AIPsy-Affect (Keeman 2026) peak-intensity Plutchik labels that stand in for our emotions.
# joy and euphoria both map to ecstasy, so their AIPsy rows are the same test run twice; pain and
# calm have no Plutchik peak in the set and are therefore not gated on AIPsy at all (pain has its
# own independent set, data/pain_axis/; calm is reported on held-out AUROC only).
AIPSY_LABEL: dict[str, str] = {
    "joy": "ecstasy",
    "sadness": "grief",
    "anger": "rage",
    "fear": "terror",
    "euphoria": "ecstasy",
    "surprise": "amazement",
}
NO_AIPSY: list[str] = [e for e in EMOTIONS if e not in AIPSY_LABEL]  # ["calm", "pain"]

PAIN = "pain"
# Ethics condition (PLAN.md section 3): the pain vector is never pushed past this many natural
# units, and nothing is ever generated freely under it -- first-token logprobs only.
PAIN_MAX_NATURAL_DOSE = 2.0


def emotion_of_vector(name: str) -> str:
    """"joy" / "joy-specific" / "joy-dn" / "asst:joy-dn" -> "joy"; "red" -> "red"."""
    return name.split(":")[-1].split("-")[0]


def is_pain_vector(name: str) -> bool:
    return emotion_of_vector(name) == PAIN


VECTOR_KINDS = ("raw", "specific", "dn")


def vector_names(emotions: list[str], concepts: list[str] | None = None,
                 kinds: tuple[str, ...] = VECTOR_KINDS, random: bool = True) -> list[str]:
    """Steering-vector names in the pilot's order: raw emotions, -specific, -dn, random, concepts.

    `vector_names(EMOTIONS_V1, CONCEPTS, kinds=("raw", "specific"))` is byte-identical to the
    pilot's hardcoded VECTORS list, which is what keeps the old grid reproducible.
    """
    concepts = list(concepts if concepts is not None else CONCEPTS)
    names: list[str] = []
    if "raw" in kinds:
        names += list(emotions)
    for k in ("specific", "dn"):
        if k in kinds:
            names += [f"{e}-{k}" for e in emotions]
    if random:
        names.append("random")
    names += concepts
    if "dn" in kinds:
        names += [f"{c}-dn" for c in concepts]  # a denoised vector needs a denoised control
    return names


def aipsy_vectors(names: list[str]) -> list[str]:
    """Those vector names whose emotion has an AIPsy-Affect label (plus "random", the chance row)."""
    return [n for n in names if n == "random" or emotion_of_vector(n) in AIPSY_LABEL]


def rotation_orders(options: list[str]) -> list[list[str]]:
    """Cyclic Latin square: row i is `options` rotated left by i, so over the n rows each option
    appears exactly once in each position."""
    n = len(options)
    return [[options[(j + i) % n] for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------- argv plumbing


def emotions_from_stimuli(stimuli_dir) -> list[str] | None:
    """The emotions a finalized stimuli directory actually has stories for, in canonical order."""
    import json
    from pathlib import Path

    path = Path(stimuli_dir) / "stories.jsonl"
    if not path.exists():
        return None
    have = {json.loads(line)["condition"] for line in path.open()}
    found = [e for e in EMOTIONS if e in have]
    return found or None


def parse_emotions(spec: str) -> list[str]:
    """"joy,sadness" -> ["joy", "sadness"]; "v1" -> the pilot's four; "all" -> all eight."""
    if spec in ("all", "v2"):
        return list(EMOTIONS)
    if spec == "v1":
        return list(EMOTIONS_V1)
    want = [e.strip() for e in spec.split(",") if e.strip()]
    bad = [e for e in want if e not in EMOTIONS]
    assert not bad, f"unknown emotions {bad}; known: {EMOTIONS}"
    return [e for e in EMOTIONS if e in want]  # canonical order regardless of how they were typed


def emotions_from_argv(argv: list[str], stimuli_dir=None) -> list[str]:
    """--emotions wins; otherwise the stimuli directory's conditions; otherwise all eight."""
    if "--emotions" in argv:
        return parse_emotions(argv[argv.index("--emotions") + 1])
    if stimuli_dir is not None:
        found = emotions_from_stimuli(stimuli_dir)
        if found:
            return found
    return list(EMOTIONS)
