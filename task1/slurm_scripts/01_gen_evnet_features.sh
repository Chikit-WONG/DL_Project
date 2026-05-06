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

# Read WEIGHTS_ROOT from task2/configs/local.yaml (the single path config file).
# Override by exporting WEIGHTS_ROOT or CLIP_RN50 before sbatch if needed.
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$(python -c "import yaml; print(yaml.safe_load(open('$REPO_ROOT/task2/configs/local.yaml'))['paths']['weights_root'])")}"
CLIP_RN50="${CLIP_RN50:-$WEIGHTS_ROOT/CLIP-RN50-openai/open_clip_pytorch_model.bin}"
echo "Using CLIP_RN50=$CLIP_RN50"

python preprocess/process_image_course.py \
    --clip_checkpoint "$CLIP_RN50" \
    --save_dir output/Image_feature

echo "Job ended: $(date)"
