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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK1_DIR="$(dirname "$SCRIPT_DIR")"
mkdir -p "$SCRIPT_DIR/logs"
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
# Set EEG_DATA_DIR to your preprocessed EEG data path before submitting:
#   export EEG_DATA_DIR=/path/to/Preprocessed_data_250Hz_whiten/sub-01
: "${EEG_DATA_DIR:?ERROR: set EEG_DATA_DIR to Preprocessed_data_250Hz_whiten/sub-01}"

python main_eeg_course.py \
    --blur_config 8 --use_evnet --use_full_train --epoch 200 --train_batch_size 1024 --lr 0.001 --n_seeds 10 --first_seed 21 --feature_path output/Image_feature --output_dir output/logs/8blur_evnet_full \
    --eeg_data_dir "$EEG_DATA_DIR"

echo "Job ended: $(date)"
