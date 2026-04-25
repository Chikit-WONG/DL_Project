#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
type unclash >/dev/null 2>&1 && unclash || true

DATA_ROOT=${DATA_ROOT:-/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data}
CLIP_CHECKPOINT=${CLIP_CHECKPOINT:-/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/open_clip_pytorch_model.bin}
SD_MODEL_PATH=${SD_MODEL_PATH:-/hpc2hdd/home/ckwong627/workdir/models/stable-diffusion-v1-5}
IP_ADAPTER_ROOT=${IP_ADAPTER_ROOT:-/hpc2hdd/home/ckwong627/workdir/models/IP-Adapter}

if [[ -n "${TASK1_CKPT:-}" ]]; then
  TASK1_CKPT_RESOLVED=$TASK1_CKPT
else
  TASK1_CKPT_RESOLVED=$(find "$REPO_DIR/output/logs/main_eeg_course" -type f -name '*_select.pth' | sort | tail -n 1 || true)
fi

if [[ -z "${TASK1_CKPT_RESOLVED:-}" ]]; then
  echo "TASK1_CKPT is not set and no *_select.pth checkpoint was found under output/logs/main_eeg_course." >&2
  echo "Run task 1 first or export TASK1_CKPT=/abs/path/to/checkpoint.pth" >&2
  exit 1
fi

cd "$REPO_DIR"

python scripts/run_task2_pipeline.py \
  --data_root "$DATA_ROOT" \
  --clip_checkpoint "$CLIP_CHECKPOINT" \
  --task1_ckpt "$TASK1_CKPT_RESOLVED" \
  --sd_model_path "$SD_MODEL_PATH" \
  --ip_adapter_root "$IP_ADAPTER_ROOT" \
  "$@"
