#!/usr/bin/env bash
# Run on the instance: python env plus model snapshots. vm_setup.sh moonlight|k26|qwen
set -euo pipefail
source /opt/pytorch/bin/activate  # the DLAMI's venv (owned by ubuntu), with its torch
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.device_count(), 'GPUs')"
pip install -q "transformers==4.57.1" safetensors compressed-tensors "huggingface_hub[hf_xet]" tiktoken blobfile accelerate
export HF_XET_HIGH_PERFORMANCE=1
case "${1:-}" in
  moonlight) hf download moonshotai/Moonlight-16B-A3B-Instruct --revision 4e735b07a89f73647dfab71ab91b840f362ede5b --local-dir ~/models/moonlight ;;
  k26) hf download moonshotai/Kimi-K2.6 --revision 7eb5002f6aadc958aed6a9177b7ed26bb94011bb --local-dir /nvme/k26 ;;
  qwen) hf download Qwen/Qwen3-32B --revision 9216db5781bf21249d130ec9da846c4624c16137 --local-dir /nvme/qwen/Qwen3-32B ;;
esac
