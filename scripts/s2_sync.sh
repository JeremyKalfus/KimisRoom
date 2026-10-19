#!/usr/bin/env bash
# s2_sync.sh NAME: sync.sh for Study 2. Pushes the repo to ~/KimisRoom on the instance with sync.sh's excludes,
# except the box's own Qwen replies (data/s2/replies/qwen3-32b), which the assent stage writes there: they are
# never deleted, and the laptop's copy (graded: verdicts, selected.json, eligible.json) goes up with --update, so
# a newer file on the box is never overwritten. Then the inputs that live under the excluded results/: the two
# registered vector files, Study 1's D2 and D2c adapters (adapter directory and meta.json only), and the stamped
# controls.pt and pool.json of both models (the pilot's control vectors; a fresh box would not have them). Also rule
# 1's decisions (results/s2/pilot/*decision*.json) and rule 3's noun-gate results (results/s2/{qwen,k26}/nouns).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; root="$(cd "$here/.." && pwd)"
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-$("$here/region.sh" "$1")}
"$here/allow_ip.sh"
ip=$("$here/aws.sh" ec2 describe-instances --filters Name=tag:Name,Values="$1" Name=instance-state-name,Values=running \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
[ "$ip" != None ] || { echo "$1 is not running" >&2; exit 1; }
e="ssh -i $HOME/.ssh/kimisroom_aws -o StrictHostKeyChecking=accept-new"
rsync -az --delete --exclude .git --exclude refs --exclude results --exclude .env.local --exclude __pycache__ --exclude .venv \
  --exclude /data/s2/replies/qwen3-32b/ -e "$e" "$root/" ubuntu@"$ip":KimisRoom/
cd "$root"
[ -d data/s2/replies/qwen3-32b ] && rsync -az --update -e "$e" data/s2/replies/qwen3-32b/ ubuntu@"$ip":KimisRoom/data/s2/replies/qwen3-32b/
# The box files the runner's weight check reads (a fresh box has none; never overwrite a newer one there)
[ -d results/s2/box ] && rsync -azR --update -e "$e" results/s2/box ubuntu@"$ip":KimisRoom/
# Rule 1's decisions (made here from the pilot rows) and rule 3's noun-gate results (made on a box, fetched home; a
# newer copy there is kept), which the nouns_* and confirm_* stages read
D=(results/s2/pilot/*decision*.json); [ -e "${D[0]}" ] && rsync -azR -e "$e" "${D[@]}" ubuntu@"$ip":KimisRoom/
for d in results/s2/qwen/nouns results/s2/k26/nouns; do
  [ -d $d ] && rsync -azR --update --exclude '*.parts/' --exclude '*.job.pt' -e "$e" $d ubuntu@"$ip":KimisRoom/
done
A=results/qwen/adapters/distill_keep
rsync -azR -e "$e" results/qwen/v2/vectors.pt results/k26/v2/vectors.pt \
  $A/D2/adapter $A/D2/meta.json $A/D2c/adapter $A/D2c/meta.json \
  results/qwen/s2/controls.pt results/qwen/s2/pool.json results/k26/s2/controls.pt results/k26/s2/pool.json \
  ubuntu@"$ip":KimisRoom/
