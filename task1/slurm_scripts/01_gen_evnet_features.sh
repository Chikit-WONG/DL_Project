#!/bin/bash
# Submit from the repository root:
#   sbatch task1/slurm_scripts/01_gen_evnet_features.sh
#
#SBATCH -p debug
#SBATCH -o task1/slurm_scripts/logs/01_gen_evnet_features_%j.out
#SBATCH -e task1/slurm_scripts/logs/01_gen_evnet_features_%j.err
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

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

# Set CLIP_RN50 to your OpenCLIP RN50 checkpoint path before submitting:
#   export CLIP_RN50=/path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin
: "${CLIP_RN50:?ERROR: set CLIP_RN50 to the path of open_clip_pytorch_model.bin}"

python preprocess/process_image_course.py \
    --clip_checkpoint "$CLIP_RN50" \
    --save_dir output/Image_feature

echo "Job ended: $(date)"
