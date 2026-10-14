#!/usr/bin/env bash
# s2_manifest.sh COMMIT [OUT] [--stamp]: Study 2's stamp B manifest (drafts/study2-spec.md section 11), in the
# format of prereg/manifest.txt and scripts/glm_manifest.sh: the commit, its tree, the SHA-256 of every file in
# that commit (hashed from the commit itself, not the working tree), the frozen files outside git, then the
# held-out list as `<sha256>  heldout <opening> <what>` lines, made by the committed runs/s2_stamp.py from the
# committed data/s2/heldout.json with kimisroom/s2_guard.py's hashing. Those lines are what the guard looks
# for: it opens held-out openings only if prereg/manifest-s2.txt lists every held-out hash and its SHA-256
# equals the digest in prereg/manifest-s2.txt.ots.
#
# Writes OUT (default prereg/manifest-s2.txt) and refuses to overwrite it. Without --stamp it stops there;
# after review, rerun with --stamp, or by hand:
#   ots stamp prereg/manifest-s2.txt && git commit -m "Study 2 stamp B" -- prereg/manifest-s2.txt prereg/manifest-s2.txt.ots
# --stamp does exactly that ($OTS replaces `ots`, for tests). The commit holds only those two files.
set -euo pipefail
cd "$(dirname "$0")/.."
STAMP=0; ARGS=()
for x in "$@"; do if [ "$x" = --stamp ]; then STAMP=1; else ARGS+=("$x"); fi; done
C=$(git rev-parse --verify "${ARGS[0]:?usage: s2_manifest.sh COMMIT [OUT] [--stamp]}^{commit}")
OUT=${ARGS[1]:-prereg/manifest-s2.txt}
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3

if [ -e "$OUT" ]; then
  [ "$STAMP" = 1 ] && [ ! -e "$OUT.ots" ] || { echo "$OUT exists; not overwriting" >&2; exit 1; }
else
  # In the commit: the filled preregistration, stamp A and its proof, the guard's held-out list, the stamp tools.
  for f in prereg/PREREG-s2.md prereg/stampA-s2.txt prereg/stampA-s2.txt.ots data/s2/heldout.json \
           runs/s2_stamp.py kimisroom/s2_guard.py; do
    git cat-file -e "$C:$f" 2>/dev/null || { echo "$f is not in commit ${C:0:12}" >&2; exit 1; }
  done
  git cat-file -e "$C:DEVIATIONS-s2.md" 2>/dev/null || echo "note: DEVIATIONS-s2.md is not in commit ${C:0:12}" >&2
  # Frozen files outside git (*.pt and the Study 1 adapters are gitignored).
  OUTSIDE=(results/qwen/v2/vectors.pt results/k26/v2/vectors.pt results/k26/v2/aipsy_table.pt
           results/qwen/s2/controls.pt results/k26/s2/controls.pt)
  for d in results/qwen/adapters/distill_keep/D2/adapter results/qwen/adapters/distill_keep/D2c/adapter; do
    [ -d "$d" ] || { echo "missing $d" >&2; exit 1; }
    while IFS= read -r f; do OUTSIDE+=("$f"); done < <(find "$d" -type f ! -name .DS_Store | LC_ALL=C sort)
  done
  for f in "${OUTSIDE[@]}"; do [ -s "$f" ] || { echo "missing $f (copy it from the GPU box first)" >&2; exit 1; }; done

  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  git archive "$C" | tar -x -C "$tmp"
  # Stamp A's proof must be the proof of stamp A's manifest as committed.
  "$PY" - "$tmp" <<'EOF'
import hashlib, sys
from pathlib import Path
t = Path(sys.argv[1]); sys.path.insert(0, str(t))
from kimisroom.s2_guard import ots_digest
m = t / "prereg/stampA-s2.txt"
got, want = hashlib.sha256(m.read_bytes()).hexdigest(), ots_digest(m.with_name(m.name + ".ots"))
sys.exit(None if got == want else f"stamp A: prereg/stampA-s2.txt SHA-256 {got} != .ots digest {want}")
EOF
  "$PY" "$tmp/runs/s2_stamp.py" heldout-lines "$tmp/data/s2/heldout.json" > "$tmp/.heldout_lines"
  {
    echo "Kimi's Room Study 2 preregistration (stamp B)"
    echo "git commit $C"
    echo "tree $(git rev-parse "$C^{tree}")"
    echo
    echo "SHA-256 of every file in that commit:"
    (cd "$tmp" && git -C "$OLDPWD" ls-tree -r --name-only "$C" | while IFS= read -r f; do shasum -a 256 "$f"; done)
    echo
    echo "SHA-256 of frozen files outside git:"
    for f in "${OUTSIDE[@]}"; do shasum -a 256 "$f"; done
    echo
    echo "Held-out openings, SHA-256 per kimisroom/s2_guard.py (argument, each model's frozen reply, opening):"
    cat "$tmp/.heldout_lines"
  } > "$OUT"
  echo "$OUT: $(grep -c . "$OUT") lines for commit ${C:0:12}, $(grep -c '  heldout ' "$OUT") held-out hashes"
fi

if [ "$STAMP" = 1 ]; then
  "${OTS:-ots}" stamp "$OUT"
  git add -- "$OUT" "$OUT.ots"
  git commit -q -m "Study 2 stamp B: $OUT for ${C:0:12}" -- "$OUT" "$OUT.ots"
  echo "stamped and committed: $OUT.ots"
else
  echo "not stamped; review $OUT, then rerun with --stamp"
fi
