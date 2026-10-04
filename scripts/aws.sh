#!/usr/bin/env bash
# aws CLI with the root key from kv (entry name in the gitignored .env.local), us-east-2 by default.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/.env.local"
exec kv exec AWS_SECRET_ACCESS_KEY="$AWS_KV_NAME" -- env AWS_ACCESS_KEY_ID="$AWS_KV_NAME" \
  AWS_SHARED_CREDENTIALS_FILE=/dev/null AWS_CONFIG_FILE=/dev/null \
  AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-2}" aws "$@"
