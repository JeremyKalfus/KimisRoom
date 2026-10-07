#!/usr/bin/env bash
# On the sweep's cloud box after a restart (/nvme is wiped on stop): download Qwen3-14B again at the
# revision pinned in ~/sweep/revisions.txt, then the calibration -> rule -> pilot chain for 14B.
#   ssh ubuntu@BOX 'nohup bash ~/KimisRoom/scripts/sweep_box_14b.sh > ~/sweep/box14b.log 2>&1 &'
set -uo pipefail
source /opt/pytorch/bin/activate
cd ~/KimisRoom
S=~/sweep; Q=/nvme/qwen; m=Qwen3-14B
r=$(awk -v m="$m" '$1==m{print $2}' $S/revisions.txt)
[ -n "$r" ] || { echo "no pinned revision for $m"; exit 1; }
mkdir -p $Q
[ -f $Q/$m/config.json ] || hf download Qwen/$m --revision "$r" --local-dir $Q/$m > /dev/null 2>&1 || { echo "download failed"; exit 1; }
echo "$m downloaded $(date -u +%H:%M:%S)"
bash ~/KimisRoom/scripts/sweep_chain.sh $Q/$m
echo "BOX 14B DONE $(date -u +%H:%M:%S)"
