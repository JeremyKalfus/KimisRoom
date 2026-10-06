#!/usr/bin/env bash
# On the GPU box, for GLM-5.3 (kimisroom/glm.py). Run from the Mac as
#   scripts/vm_ssh.sh kr-c 'bash -s' < scripts/glm_box.sh ... (or copy the steps):
#   bash ~/KimisRoom/scripts/glm_box.sh setup      # ~/glmenv: the DLAMI torch plus transformers 5.18
#   bash ~/KimisRoom/scripts/glm_box.sh download   # zai-org/GLM-5.3 at the pinned revision -> /nvme/glm53
#   bash ~/KimisRoom/scripts/glm_box.sh cap HOURS  # move the per-boot shutdown cap
#
# GLM needs transformers >= 5.6 (glm_moe_dsa, its TokenizersBackend tokenizer) while the Qwen backend
# keeps 4.57.1 in /opt/pytorch, so GLM gets its own venv that sees the DLAMI's torch through
# --system-site-packages; both live on the root EBS and survive a stop. /nvme is wiped on every stop,
# so the weights are downloaded again after each start (~755 GB).
set -euo pipefail
REV=aca966e4e02791568aa6a4ced368624b3d897f42  # zai-org/GLM-5.3, pinned 2026-10-05
case "${1:-}" in
  setup)
    [ -x ~/glmenv/bin/python ] || /opt/pytorch/bin/python -m venv --system-site-packages ~/glmenv
    ~/glmenv/bin/pip install -q "transformers==5.18.0" "huggingface_hub[hf_xet]>=1.31" safetensors jinja2
    ~/glmenv/bin/python -c "import torch, transformers; assert torch.cuda.is_available(); \
print('glmenv: torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.device_count(), 'GPUs')"
    ;;
  download)
    export HF_XET_HIGH_PERFORMANCE=1
    t0=$(date +%s)
    ~/glmenv/bin/hf download zai-org/GLM-5.3 --revision $REV --local-dir /nvme/glm53
    n=$(ls /nvme/glm53/*.safetensors | wc -l)
    echo "GLM-5.3 $REV: $n shards, $(du -sh /nvme/glm53 | cut -f1), $(( $(date +%s) - t0 ))s"
    [ "$n" -eq 141 ]
    ;;
  cap)
    echo "$2" | sudo tee /etc/kimisroom/max_hours > /dev/null
    sudo shutdown -c 2>/dev/null || true
    sudo shutdown -h +$(( $2 * 60 ))
    echo "shutdown in $2 h"
    ;;
  *) echo "usage: $0 setup|download|cap HOURS" >&2; exit 2 ;;
esac
