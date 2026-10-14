#!/usr/bin/env bash
# s2_sync.sh NAME: sync.sh for Study 2. Pushes the repo to ~/KimisRoom on the instance with sync.sh's excludes,
# except the box's own Qwen replies (data/s2/replies/qwen3-32b), which the assent stage writes there: they are
# never deleted, and the laptop's copy (graded: verdicts, selected.json, eligible.json) goes up with --update, so
# a newer file on the box is never overwritten. Then the inputs that live under the excluded results/: the two
# registered vector files and Study 1's D2 and D2c adapters (adapter directory and meta.json only).
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
A=results/qwen/adapters/distill_keep
rsync -azR -e "$e" results/qwen/v2/vectors.pt results/k26/v2/vectors.pt \
  $A/D2/adapter $A/D2/meta.json $A/D2c/adapter $A/D2c/meta.json ubuntu@"$ip":KimisRoom/
