#!/usr/bin/env bash
# sync.sh NAME: push the repo (code, prompts, configs) to ~/KimisRoom on the instance.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-$("$here/region.sh" "$1")}
"$here/allow_ip.sh"; root="$(cd "$here/.." && pwd)"
ip=$("$here/aws.sh" ec2 describe-instances --filters Name=tag:Name,Values="$1" Name=instance-state-name,Values=running \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
rsync -az --delete --exclude .git --exclude refs --exclude results --exclude .env.local --exclude __pycache__ --exclude .venv \
  -e "ssh -i $HOME/.ssh/kimisroom_aws -o StrictHostKeyChecking=accept-new" "$root/" ubuntu@"$ip":KimisRoom/
