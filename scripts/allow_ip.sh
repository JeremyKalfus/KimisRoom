#!/usr/bin/env bash
# allow_ip.sh: make sure the kimisroom-ssh group admits this machine's current public IP.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
ip=$(curl -s https://checkip.amazonaws.com)
cache=/tmp/kimisroom-allowed-ip-${AWS_DEFAULT_REGION:-us-east-2}
[ "$(cat $cache 2>/dev/null)" = "$ip" ] && exit 0
sg=$("$here/aws.sh" ec2 describe-security-groups --filters Name=group-name,Values=kimisroom-ssh --query 'SecurityGroups[0].GroupId' --output text)
"$here/aws.sh" ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 22 --cidr "$ip/32" >/dev/null 2>&1 || true
echo "$ip" > $cache
