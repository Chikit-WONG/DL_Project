#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
type unclash >/dev/null 2>&1 && unclash || true

DATA_ROOT=${DATA_ROOT:-/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data}
CLIP_CHECKPOINT=${CLIP_CHECKPOINT:-/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/open_clip_pytorch_model.bin}

cd "$REPO_DIR"

python scripts/run_course_pipeline.py \
  --data_root "$DATA_ROOT" \
  --clip_checkpoint "$CLIP_CHECKPOINT" \
  "$@"
