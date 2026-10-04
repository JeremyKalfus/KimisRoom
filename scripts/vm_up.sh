#!/usr/bin/env bash
# vm_up.sh NAME [TYPE] [ROOT_GB]: start the tagged instance, or create it. Prints its public IP.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; aws() { "$here/aws.sh" "$@"; }
name=$1; type=${2:-g6e.2xlarge}; root=${3:-200}
key=kimisroom; sg_name=kimisroom-ssh
[ -f ~/.ssh/kimisroom_aws ] || ssh-keygen -q -t ed25519 -N "" -f ~/.ssh/kimisroom_aws -C kimisroom
aws ec2 describe-key-pairs --key-names $key >/dev/null 2>&1 || \
  aws ec2 import-key-pair --key-name $key --public-key-material fileb://$HOME/.ssh/kimisroom_aws.pub >/dev/null
sg=$(aws ec2 describe-security-groups --filters Name=group-name,Values=$sg_name --query 'SecurityGroups[0].GroupId' --output text)
if [ "$sg" = None ]; then
  sg=$(aws ec2 create-security-group --group-name $sg_name --description "Kimis Room SSH" --query GroupId --output text)
fi
ip=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 22 --cidr "$ip/32" >/dev/null 2>&1 || true
id=$(aws ec2 describe-instances --filters Name=tag:Name,Values="$name" Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)
if [ "$id" = None ]; then
  ami=$(aws ssm get-parameter --name /aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.12-ubuntu-24.04/latest/ami-id --query Parameter.Value --output text)
  # GPU capacity runs out per AZ, so try each default subnet in turn
  id=None
  for subnet in $(aws ec2 describe-subnets --filters Name=default-for-az,Values=true --query 'Subnets[].SubnetId' --output text); do
    id=$(aws ec2 run-instances --image-id "$ami" --instance-type "$type" --key-name $key --security-group-ids "$sg" \
      --subnet-id "$subnet" --instance-initiated-shutdown-behavior stop \
      --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$root,VolumeType=gp3,Throughput=500,Iops=6000,DeleteOnTermination=true}" \
      --user-data fileb://"$here/userdata.yaml" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$name},{Key=Project,Value=KimisRoom}]" \
      "ResourceType=volume,Tags=[{Key=Project,Value=KimisRoom}]" \
      --query 'Instances[0].InstanceId' --output text 2>/tmp/kimisroom-run.err) && break
    echo "$subnet: $(grep -o 'InsufficientInstanceCapacity\|[A-Za-z]*Exception\|InvalidParameter[A-Za-z]*' /tmp/kimisroom-run.err | head -1)" >&2
    id=None
  done
  [ "$id" = None ] && { cat /tmp/kimisroom-run.err >&2; exit 1; }
  echo "created $id ($type)" >&2
else
  state=$(aws ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].State.Name' --output text)
  [ "$state" = stopping ] && aws ec2 wait instance-stopped --instance-ids "$id"
  [ "$state" != running ] && aws ec2 start-instances --instance-ids "$id" >/dev/null && echo "started $id" >&2
fi
aws ec2 wait instance-running --instance-ids "$id"
aws ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
