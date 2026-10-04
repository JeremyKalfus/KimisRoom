#!/usr/bin/env bash
# fetch_aipsy.sh: AIPsy-Affect (Keeman 2026, arXiv 2604.23719, MIT) at a pinned revision, plus a
# JSONL copy of each split so the loaders need no parquet library. Independent test set for the
# emotion vectors (PLAN.md section 10); never used to build or tune them.
set -euo pipefail
cd "$(dirname "$0")/../data/aipsy_affect"
rev=1e24598f2041e86f788d1974cb3661b67d236fe4
for f in README.md data/clinical.parquet data/neutral.parquet data/moderate.parquet data/complex_neutral.parquet; do
  curl -sfL -o "$(basename $f)" "https://huggingface.co/datasets/keidolabs/aipsy-affect/resolve/$rev/$f"
done
echo "$rev" > REVISION
uv run --no-project --with pyarrow python - <<'PY'
import json
import pyarrow.parquet as pq
for split in ["clinical", "neutral", "moderate", "complex_neutral"]:
    rows = pq.read_table(f"{split}.parquet").to_pylist()
    with open(f"{split}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({**r, "split": split}) + "\n")
    print(split, len(rows))
PY
