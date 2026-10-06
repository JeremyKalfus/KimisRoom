#!/usr/bin/env bash
# sweep_box_up.sh [TYPE]: the Qwen3 size sweep's cloud box. Resizes the stopped kr-q (us-east-2) to
# TYPE (default g6e.2xlarge, 1x L40S 48 GB) and starts it, all under one kv grant, then prints its IP.
# kr-q is never terminated; its root volume keeps the DLAMI venv and the repo, /nvme is wiped on stop.
# The box's own timers stop it: per-boot uptime cap (/etc/kimisroom/max_hours) and 45 idle minutes.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
type=${1:-g6e.2xlarge}
export AWS_DEFAULT_REGION=us-east-2
if [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  source "$here/../.env.local"
  exec kv exec AWS_SECRET_ACCESS_KEY="$AWS_KV_NAME" -- env AWS_ACCESS_KEY_ID="$AWS_KV_NAME" "$0" "$@"
fi
aws() { "$here/aws.sh" "$@"; }
id=$(aws ec2 describe-instances --filters Name=tag:Name,Values=kr-q Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)
[ "$id" = None ] && { echo "kr-q not found in $AWS_DEFAULT_REGION" >&2; exit 1; }
read -r state cur az <<<"$(aws ec2 describe-instances --instance-ids "$id" \
  --query 'Reservations[0].Instances[0].[State.Name,InstanceType,Placement.AvailabilityZone]' --output text)"
echo "kr-q $id: $state $cur $az, shutdown behavior $(aws ec2 describe-instance-attribute --instance-id "$id" \
  --attribute instanceInitiatedShutdownBehavior --query InstanceInitiatedShutdownBehavior.Value --output text)" >&2
[ "$state" = stopping ] && { aws ec2 wait instance-stopped --instance-ids "$id"; state=stopped; }
if [ "$state" = stopped ] && [ "$cur" != "$type" ]; then
  aws ec2 modify-instance-attribute --instance-id "$id" --instance-type "{\"Value\": \"$type\"}"
  echo "resized $cur -> $type" >&2
fi
if [ "$state" != running ]; then
  aws ec2 start-instances --instance-ids "$id" >/dev/null
  aws ec2 wait instance-running --instance-ids "$id"
fi
"$here/allow_ip.sh"
aws ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
