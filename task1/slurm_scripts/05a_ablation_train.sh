#!/bin/bash
# Ablation training on regular (non-grey-background) images.
# Runs 5 configurations as a Slurm array — all use existing feature files,
# so no feature extraction is needed beforehand.
#
# Submit from the repository root:
#   sbatch task1/slurm_scripts/05a_ablation_train.sh
#
# Array task → config mapping:
#   0  12 blur,  no EVNet   (logs/ablation_12blur)
#   1  12 blur + EVNet      (logs/ablation_12blur_evnet)
#   2   8 blur + EVNet      (logs/ablation_8blur_evnet)
#   3   1 blur + EVNet      (logs/ablation_1blur_evnet)   ← "EVNet, no blur"
#   4   1 blur,  no EVNet   (logs/ablation_1blur)         ← "no blur, no EVNet"
#
#SBATCH -p emergency_gpua40
#SBATCH --array=0-4
#SBATCH -o task1/slurm_scripts/logs/05a_ablation_%a_%j.out
#SBATCH -e task1/slurm_scripts/logs/05a_ablation_%a_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TASK1_DIR="$REPO_ROOT/task1"
mkdir -p "$REPO_ROOT/task1/slurm_scripts/logs"
cd "$TASK1_DIR"

set -eo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
type unclash >/dev/null 2>&1 && unclash || true

for _conda in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    [ -f "$_conda/etc/profile.d/conda.sh" ] && { source "$_conda/etc/profile.d/conda.sh"; break; }
done
conda activate DL_Project

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi

# Per-task config
BLUR_CONFIGS=("12"   "12"    "8"    "1"    "1")
EVNET_FLAGS=( ""     "--use_evnet" "--use_evnet" "--use_evnet" "")
OUTPUT_TAGS=( "ablation_12blur" "ablation_12blur_evnet" "ablation_8blur_evnet" "ablation_1blur_evnet" "ablation_1blur")

BLUR_CFG="${BLUR_CONFIGS[$SLURM_ARRAY_TASK_ID]}"
EVNET_FLAG="${EVNET_FLAGS[$SLURM_ARRAY_TASK_ID]}"
OUT_TAG="${OUTPUT_TAGS[$SLURM_ARRAY_TASK_ID]}"

echo "Array task $SLURM_ARRAY_TASK_ID: blur=$BLUR_CFG evnet='$EVNET_FLAG' → output/logs/$OUT_TAG"

# EEG data directory (auto-detect or read from local.yaml)
if [ -z "$EEG_DATA_DIR" ]; then
    _yaml_eeg=$(python -c "
import yaml, sys
c = yaml.safe_load(open('$REPO_ROOT/task2/configs/local.yaml'))
v = (c.get('paths') or {}).get('eeg_data_dir', '')
print(v)
" 2>/dev/null)
    EEG_DATA_DIR="${_yaml_eeg:-$TASK1_DIR/../image-eeg-data}"
fi
echo "Using EEG_DATA_DIR=$EEG_DATA_DIR"

python main_eeg_course.py \
    --blur_config "$BLUR_CFG" \
    $EVNET_FLAG \
    --epoch 200 --train_batch_size 1024 --lr 0.001 \
    --n_seeds 10 --first_seed 21 \
    --feature_path output/Image_feature \
    --output_dir "output/logs/$OUT_TAG" \
    --eeg_data_dir "$EEG_DATA_DIR"

echo "Job ended: $(date)"
