#!/bin/sh
# Check every file listed in prereg/manifest.txt against this repository.
#
# Files in git are read from the tag prereg-v1, the preregistered tree. The three frozen vector files
# (*.pt), which were kept out of git when the study was registered, are read from HEAD, where they were
# added later unchanged. One preregistered file is withheld from this public copy (see README.md).
#
# Usage: sh prereg/verify.sh
set -eu
cd "$(git rev-parse --show-toplevel)"

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum; else shasum -a 256; fi | cut -d' ' -f1
}

grep -E '^[0-9a-f]{64}  ' prereg/manifest.txt | while read -r want path; do
  if git cat-file -e "prereg-v1:$path" 2>/dev/null; then
    rev=prereg-v1
  elif git cat-file -e "HEAD:$path" 2>/dev/null; then
    rev=HEAD
  elif [ "$path" = drafts/aws-sales-request.md ]; then
    echo "withheld $path (an AWS quota request with account details)"
    continue
  else
    echo "missing  $path"
    continue
  fi
  if [ "$(git cat-file blob "$rev:$path" | sha256)" = "$want" ]; then
    echo "ok       $path"
  else
    echo "MISMATCH $path (read at $rev)"
  fi
done | awk '
  $1 != "ok" { print }
  { n[$1]++ }
  END {
    printf "%d of %d files in the manifest match", n["ok"], NR
    if (n["withheld"]) printf ", %d withheld", n["withheld"]
    print ""
    exit (n["MISMATCH"] + n["missing"] > 0)
  }'
