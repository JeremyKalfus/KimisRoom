#!/usr/bin/env bash
# vm_down.sh NAME: stop the instance (EBS root kept, /nvme wiped).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; aws() { "$here/aws.sh" "$@"; }
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-$("$here/region.sh" "$1")}
id=$(aws ec2 describe-instances --filters Name=tag:Name,Values="$1" Name=instance-state-name,Values=pending,running \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)
[ "$id" = None ] && { echo "$1 not running"; exit 0; }
aws ec2 stop-instances --instance-ids "$id" >/dev/null && echo "stopping $id"
