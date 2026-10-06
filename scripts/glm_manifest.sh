#!/usr/bin/env bash
# glm_manifest.sh COMMIT [OUT]: the manifest that the GLM-5.3 amendment's OpenTimestamps proof covers
# (drafts/prereg-amendment-glm.md, section 7), in the format of prereg/manifest.txt: the commit, its
# tree, the SHA-256 of every file in that commit (hashed from the commit itself, not the working
# tree), then the frozen files outside git and the pinned model revision. Writes OUT (default
# prereg/manifest-glm.txt) and refuses to overwrite it. It does not stamp: after review, run
#   ots stamp prereg/manifest-glm.txt
# whose proof prereg/manifest-glm.txt.ots is what kimisroom/glm.py checks before GLM may see a
# confirmatory task.
set -euo pipefail
cd "$(dirname "$0")/.."
C=$(git rev-parse --verify "${1:?usage: glm_manifest.sh COMMIT [OUT]}^{commit}")
OUT=${2:-prereg/manifest-glm.txt}
[ -e "$OUT" ] && { echo "$OUT exists; not overwriting" >&2; exit 1; }
OUTSIDE=(results/glm/vec/vectors.pt results/glm/vec/aipsy_table.pt)
for f in "${OUTSIDE[@]}"; do [ -s "$f" ] || { echo "missing $f (copy it from the GPU box first)" >&2; exit 1; }; done
[ -n "${GLM_MANIFEST_ANY:-}" ] || git show "$C:drafts/prereg-amendment-glm.md" > /dev/null  # the amendment must be in the commit (GLM_MANIFEST_ANY=1: testing only)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git archive "$C" | tar -x -C "$tmp"
{
  echo "Kimi's Room preregistration amendment: GLM-5.3, tag glm-v1"
  echo "git commit $C"
  echo "tree $(git rev-parse "$C^{tree}")"
  echo
  echo "SHA-256 of every file in that commit:"
  (cd "$tmp" && git -C "$OLDPWD" ls-tree -r --name-only "$C" | while IFS= read -r f; do shasum -a 256 "$f"; done)
  echo
  echo "SHA-256 of frozen files outside git:"
  for f in "${OUTSIDE[@]}"; do shasum -a 256 "$f"; done
  echo "Model weights: zai-org/GLM-5.3 at aca966e4e02791568aa6a4ced368624b3d897f42"
} > "$OUT"
echo "$OUT: $(grep -c . "$OUT") lines for commit ${C:0:12}"
