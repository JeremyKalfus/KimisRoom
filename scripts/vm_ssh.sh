#!/usr/bin/env bash
# vm_ssh.sh NAME [CMD...]: ssh to the running instance.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-$("$here/region.sh" "$1")}
"$here/allow_ip.sh"
name=$1; shift
ip=$("$here/aws.sh" ec2 describe-instances --filters Name=tag:Name,Values="$name" Name=instance-state-name,Values=running \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
exec ssh -i ~/.ssh/kimisroom_aws -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=10 ubuntu@"$ip" "$@"
