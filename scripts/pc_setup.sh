#!/usr/bin/env bash
# Runs inside WSL on the desktop (RX 7800 XT, ROCm 7.2 via /dev/dxg): a venv for the sweep with the
# ROCm 7.2 PyTorch 2.9.1 wheels already in ~/wheels (AMD's WSL step: drop torch's bundled HSA runtime
# so the system rocr4wsl runtime is used), the repo's pinned transformers, and the small Qwen3 weights.
#   cat scripts/pc_setup.sh | ssh pc '$env:WSL_UTF8=1; wsl -e bash -s'
set -euo pipefail
if [ ! -x ~/krenv/bin/python ]; then
  python3 -m venv ~/krenv
  source ~/krenv/bin/activate
  pip install -q --upgrade pip wheel
  pip install -q numpy==1.26.4
  pip install -q ~/wheels/torch-*.whl ~/wheels/triton-*.whl
  loc=$(pip show torch | awk -F': ' '/^Location/{print $2}')
  rm -f "$loc"/torch/lib/libhsa-runtime64.so*
  pip install -q "transformers==4.57.1" safetensors accelerate "huggingface_hub[hf_xet]" tiktoken
fi
source ~/krenv/bin/activate
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
mkdir -p ~/sweep ~/models
export HF_XET_HIGH_PERFORMANCE=1
for m in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
  [ -f ~/models/$m/.done ] && continue
  rev=$(python -c "from huggingface_hub import model_info; print(model_info('Qwen/$m').sha)")
  hf download Qwen/$m --revision $rev --local-dir ~/models/$m > /dev/null
  echo "$m $rev" >> ~/sweep/revisions.txt; touch ~/models/$m/.done
done
cat ~/sweep/revisions.txt
