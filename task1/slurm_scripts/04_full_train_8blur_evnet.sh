#!/bin/bash
# Submit from the repository root:
#   sbatch task1/slurm_scripts/04_full_train_8blur_evnet.sh
#
#SBATCH -p emergency_gpua40
#SBATCH -o task1/slurm_scripts/logs/04_full_train_8blur_evnet_%j.out
#SBATCH -e task1/slurm_scripts/logs/04_full_train_8blur_evnet_%j.err
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

# Phase B: 8-blur + EVNet fixed, full train (best result)
# EEG_DATA_DIR: directory containing train.pt and test.pt (= image-eeg-data/).
# Resolution order: env var → paths.eeg_data_dir in task2/configs/local.yaml → repo default.
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
    --blur_config 8 --use_evnet --use_full_train --epoch 200 --train_batch_size 1024 --lr 0.001 --n_seeds 10 --first_seed 21 --feature_path output/Image_feature --output_dir output/logs/8blur_evnet_full \
    --eeg_data_dir "$EEG_DATA_DIR"

echo "Job ended: $(date)"
