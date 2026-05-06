#!/bin/bash
# Step 1 of grey-background ablations:
#   (a) Create grey-background composite images (CPU, ~5 min)
#   (b) Extract MultiBlur + EVNet features from those images (GPU, ~15 min)
#
# Must complete before 05c_greybg_train.sh can run.
#
# Submit from the repository root:
#   sbatch task1/slurm_scripts/05b_greybg_features.sh
#
# Output:
#   task1/output/Image_set_greybg/     ← composite images
#   task1/output/Image_feature_greybg/ ← .pt feature files
#
#SBATCH -p emergency_gpua40
#SBATCH -o task1/slurm_scripts/logs/05b_greybg_features_%j.out
#SBATCH -e task1/slurm_scripts/logs/05b_greybg_features_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

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

# ── (a) Create grey-background images ──────────────────────────────────────
GREYBG_IMG_DIR="$TASK1_DIR/output/Image_set_greybg"

echo ""
echo "=== Step (a): creating grey-background images ==="
python "$REPO_ROOT/task1/scripts/make_greybg_images.py" \
    --output "$GREYBG_IMG_DIR"
# --input and --redpoint use defaults (Image_set_Resize and scripts/redpoint.png)

# ── (b) Extract features from grey-background images ───────────────────────
GREYBG_FEAT_DIR="$TASK1_DIR/output/Image_feature_greybg"

# Resolve CLIP RN50 checkpoint (same logic as 01_gen_evnet_features.sh)
if [ -z "$WEIGHTS_ROOT" ]; then
    WEIGHTS_ROOT=$(python -c "
import yaml
print(yaml.safe_load(open('$REPO_ROOT/task2/configs/local.yaml'))['paths']['weights_root'])
" 2>/dev/null || echo "")
fi
CLIP_RN50="${CLIP_RN50:-$WEIGHTS_ROOT/CLIP-RN50-openai/open_clip_pytorch_model.bin}"

echo ""
echo "=== Step (b): extracting features (clip=$CLIP_RN50) ==="
python preprocess/process_image_course.py \
    --clip_checkpoint "$CLIP_RN50" \
    --image_set_dir  "$GREYBG_IMG_DIR" \
    --save_dir       "$GREYBG_FEAT_DIR" \
    --evnet_mode     random \
    --backbone       rn50 \
    --batch_size     128

echo ""
echo "Feature files written to: $GREYBG_FEAT_DIR"
ls "$GREYBG_FEAT_DIR"
echo "Job ended: $(date)"
