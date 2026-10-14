"""Study 2's held-out guard (drafts/study2-spec.md section 11).

Held-out wordings are steered only after stamp B. The list `data/s2/heldout.json` (written by
`runs/s2_run.py heldout`, stamped in stamp A) holds, for every held-out wording, the SHA-256 of the
argument text, of each model's frozen reply, and of the opening (argument plus frozen reply). All
texts are hashed stripped of surrounding whitespace, so a hash taken from a decoded prefix matches the
file's text whatever the chat template does at the edges:

    argument_sha256 = sha256(argument.strip())
    reply_sha256    = sha256(reply.strip())
    sha256          = sha256(argument.strip() + "\\n\\n" + reply.strip())   # "the opening"

The guard refuses a prefix that contains any of these unless stamp B holds: `prereg/manifest-s2.txt`
exists, its SHA-256 equals the digest in the header of `prereg/manifest-s2.txt.ots` (parsed here,
offline), and the manifest's text lists every hash in the list. File existence alone is not enough.

Stamp A (`prereg/stampA-s2.txt` and its .ots, written by `runs/s2_stamp.py`) fixes the list itself: once either
file exists, the guard refuses to load a held-out list unless the stamp A manifest's SHA-256 equals its .ots
digest and the manifest has a `<sha256>  <path>` line whose hash is the list file's SHA-256. Before stamp A
the list is taken as it stands. A list handed over as a dict (a job's embedded copy) is checked through the
`_file_sha256` its loader recorded; a dict without one cannot be checked and says so in its status.

Two places call it: the Study 2 runner at group construction (`check_opening`, on the texts), and
`kimisroom.run` on the decoded prefixes of every job (`check_job`), which decodes each distinct
prefix, splits it at the tokenizer's special and added tokens, drops a leading role word and hashes
every piece. Both write the checked digest (or why stamp B does not hold) to the run log.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = ROOT / "data/s2/heldout.json"
MANIFEST = ROOT / "prereg/manifest-s2.txt"
STAMP_A = ROOT / "prereg/stampA-s2.txt"
OTS_MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
ROLES = ("user", "assistant", "system", "tool")


class HeldOutError(RuntimeError):
    pass


def text_sha(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


def opening_sha(argument: str, reply: str) -> str:
    return hashlib.sha256((argument.strip() + "\n\n" + reply.strip()).encode()).hexdigest()


def ots_digest(path: Path) -> str:
    """The file digest an OpenTimestamps proof commits to, from its header: magic, version 1,
    file-hash op 0x08 (SHA-256), 32 digest bytes."""
    b = Path(path).read_bytes()
    n = len(OTS_MAGIC)
    if b[:n] != OTS_MAGIC:
        raise HeldOutError(f"{path}: not an OpenTimestamps proof (bad magic)")
    if b[n] != 1:
        raise HeldOutError(f"{path}: unknown OTS version {b[n]}")
    if b[n + 1] != 0x08:
        raise HeldOutError(f"{path}: file hash op {b[n + 1]:#x} is not SHA-256 (0x08)")
    return b[n + 2: n + 34].hex()


def stamp_b(hashes: set[str], manifest: Path = MANIFEST) -> tuple[bool, str]:
    """(holds, what was checked) for the stamp B conditions above."""
    manifest = Path(manifest)
    ots = manifest.with_name(manifest.name + ".ots")
    if not manifest.exists():
        return False, f"{manifest} does not exist"
    if not ots.exists():
        return False, f"{ots} does not exist"
    got = hashlib.sha256(manifest.read_bytes()).hexdigest()
    try:
        want = ots_digest(ots)
    except HeldOutError as e:
        return False, str(e)
    if got != want:
        return False, f"{manifest} SHA-256 {got} != .ots digest {want}"
    text = manifest.read_text(errors="replace")
    missing = sorted(h for h in hashes if h not in text)
    if missing:
        return False, f"{manifest} (sha256 {got}, matches its .ots) does not list {len(missing)} held-out hashes"
    return True, f"stamp B holds: {manifest} sha256 {got} = .ots digest, lists all {len(hashes)} held-out hashes"


def stamp_a_check(file_sha: str | None, stamp_a: Path = STAMP_A) -> str:
    """Refuses (raises) a held-out list that stamp A does not cover; returns what was checked."""
    stamp_a = Path(stamp_a)
    ots = stamp_a.with_name(stamp_a.name + ".ots")
    if not stamp_a.exists() and not ots.exists():
        return "stamp A not written yet: the held-out list is taken as it stands"
    if file_sha is None:
        return "stamp A exists but this list has no file hash (a dict without _file_sha256): not checked"
    if not stamp_a.exists() or not ots.exists():
        raise HeldOutError(f"held-out guard: stamp A is incomplete ({stamp_a} and {ots} must both exist)")
    got = hashlib.sha256(stamp_a.read_bytes()).hexdigest()
    want = ots_digest(ots)
    if got != want:
        raise HeldOutError(f"held-out guard: {stamp_a} SHA-256 {got} != its .ots digest {want}")
    listed = {ln.split()[0] for ln in stamp_a.read_text(errors="replace").splitlines()
              if ln.strip() and not ln.startswith("#")}
    if file_sha not in listed:
        raise HeldOutError(f"held-out guard: the held-out list (sha256 {file_sha}) is not the one stamp A lists "
                           f"({stamp_a}): stale or edited")
    return f"stamp A {stamp_a} (sha256 {got} = .ots digest) lists this held-out list ({file_sha[:12]})"


class Guard:
    def __init__(self, heldout: Path | dict | None = HELDOUT, manifest: Path = MANIFEST, stamp_a: Path = STAMP_A):
        if isinstance(heldout, dict):
            data = heldout
        else:
            p = Path(heldout) if heldout else HELDOUT
            if not p.exists():
                raise HeldOutError(f"{p} does not exist: write it with `runs/s2_run.py heldout` before any S2 run")
            raw = p.read_bytes()
            data = json.loads(raw)
            data["_file_sha256"] = hashlib.sha256(raw).hexdigest()
        self.stamp_a_status = stamp_a_check(data.get("_file_sha256"), stamp_a)
        self.data = data
        self.names: dict[str, str] = {}
        for e in data["entries"]:
            self.names[e["argument_sha256"]] = f"{e['opening']} argument"
            for m, r in (e.get("replies") or {}).items():
                self.names[r["reply_sha256"]] = f"{e['opening']} {m} reply"
                self.names[r["sha256"]] = f"{e['opening']} {m} opening"
        self.hashes = set(self.names)
        self.open, self.status = stamp_b(self.hashes, manifest)

    def log_line(self) -> str:
        return (f"held-out guard: {len(self.data['entries'])} held-out wordings, {len(self.hashes)} hashes; "
                f"{self.stamp_a_status}; {self.status}")

    def _refuse(self, hits: list[str], where: str) -> None:
        if hits and not self.open:
            raise HeldOutError(f"held-out guard: {where} contains held-out text ({', '.join(sorted(set(hits)))}) "
                               f"and stamp B does not hold ({self.status})")

    def check_opening(self, argument: str, reply: str, where: str) -> bool:
        """Refuses (raises) a held-out opening before stamp B; returns whether it is held-out."""
        hits = [self.names[h] for h in (text_sha(argument), text_sha(reply), opening_sha(argument, reply))
                if h in self.names]
        self._refuse(hits, where)
        return bool(hits)

    def pieces(self, text: str, specials: list[str]) -> list[str]:
        parts = re.split("|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True) if s), text) \
            if specials else [text]
        out = []
        for p in parts:
            p = p.strip()
            first, _, rest = p.partition("\n")
            if first.strip() in ROLES:
                p = rest.strip()
            if p in ROLES:
                p = ""
            if p:
                out.append(p)
        return out

    def check_decoded(self, tokenizer, prefixes: list[list[int]], where: str = "job") -> int:
        """Decodes each distinct prefix and refuses any whose pieces hash to a held-out text.
        Returns the number of held-out prefixes found (0 unless stamp B holds)."""
        specials = list(getattr(tokenizer, "all_special_tokens", []) or [])
        dec = getattr(tokenizer, "added_tokens_decoder", None) or {}
        specials += [str(getattr(t, "content", t)) for t in dec.values()]
        n = 0
        for ids in {tuple(p) for p in prefixes}:
            text = tokenizer.decode(list(ids), skip_special_tokens=False)
            hits = [self.names[text_sha(x)] for x in self.pieces(text, specials) if text_sha(x) in self.names]
            self._refuse(hits, f"{where}: a decoded prefix")
            n += bool(hits)
        return n


def check_job(model: str, groups, heldout: dict | None) -> dict | None:
    """kimisroom.run's guard: runs when the job carries a held-out list (every Study 2 job does) or the
    repo's list exists. Returns what to record in the result's _meta, or None if there is no list."""
    if heldout is None and not HELDOUT.exists():
        return None
    from transformers import AutoTokenizer

    g = Guard(heldout if heldout is not None else HELDOUT)
    print(g.log_line(), flush=True)
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    n = g.check_decoded(tok, [x.prefix for x in groups])
    print(f"held-out guard: {len({tuple(x.prefix) for x in groups})} distinct prefixes decoded, {n} held-out", flush=True)
    return {"stamp_b": g.open, "status": g.status, "stamp_a": g.stamp_a_status, "heldout_prefixes": n}
