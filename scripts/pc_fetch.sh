#!/usr/bin/env bash
# pc_fetch.sh TAG: copy one size's sweep outputs from WSL on the desktop to results/sweep/TAG/. Grid
# results stay on the desktop (stdout through ssh -> PowerShell runs at ~100 KB/s); the desktop turns
# each grid into its runs/confirm_table.py long table (gzipped CSV) and only those and the small
# files travel: AIPsy table and report, choice.json, summaries, logs.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"; t=$1; mkdir -p "$root/results/sweep/$t"
cat <<EOS | ssh pc '$env:WSL_UTF8=1; wsl -e bash -s' | tar -x -C "$root/results/sweep/$t"
source ~/krenv/bin/activate; cd ~/KimisRoom
for s in cal pilot hidose; do
  [ -f ~/sweep/$t/\$s/grid.pt ] && [ ! -f ~/sweep/$t/\${s}_table.csv.gz ] &&
    PYTHONPATH=. python runs/confirm_table.py ~/sweep/$t/\$s $t ~/sweep/$t/\${s}_table.csv --entity Qwen >&2 && gzip -f ~/sweep/$t/\${s}_table.csv
done
cd ~/sweep/$t && tar -c --ignore-failed-read vec/aipsy_table.pt vec/report.txt choice.json cal_summary.txt pilot_summary.txt hidose_summary.txt *_table.csv.gz *.log {cal,pilot,hidose}/grid_meta.json 2>/dev/null
EOS
ls "$root/results/sweep/$t"
