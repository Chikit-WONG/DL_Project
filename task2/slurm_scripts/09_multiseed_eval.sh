#!/bin/bash
# Submit from the repository root:
#   sbatch --dependency=afterok:<gen_job_id> task2/slurm_scripts/09_multiseed_eval.sh
#
# Reads data_root and weights_root from task2/configs/local.yaml.
#
#SBATCH -p emergency_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --time=10:00:00
#SBATCH -J t2_eval_s%a
#SBATCH --array=0-4
#SBATCH -o task2/slurm_scripts/logs/%x-%j.out
#SBATCH -e task2/slurm_scripts/logs/%x-%j.err

SEED=$SLURM_ARRAY_TASK_ID
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TASK2_DIR="$REPO_ROOT/task2"
EXP_NAME="intra-subject_cogcappro_EEGProjectLayer_multimodal_cogcap_list_ViT-H-14"
RUN_DIR="$TASK2_DIR/runs/multiseed/$EXP_NAME/sub-01_seed${SEED}"
mkdir -p "$TASK2_DIR/slurm_scripts/logs"
cd "$TASK2_DIR"

set -euo pipefail
for _conda in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    [ -f "$_conda/etc/profile.d/conda.sh" ] && { source "$_conda/etc/profile.d/conda.sh"; break; }
done
conda activate DL_Project
module load cuda/12.6

DATA_ROOT=$(python scripts/get_data_root.py)
WEIGHTS_ROOT=$(python -c "import yaml; c=yaml.safe_load(open('configs/local.yaml')); print(c['paths']['weights_root'])")
REAL_ROOT="$DATA_ROOT/ThingsEEG/Image_set_Resize/test_images"
CLIP_H14="$WEIGHTS_ROOT/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"

echo "Job started at $(date) — seed ${SEED}"

echo "=== Evaluating all_before ==="
python scripts/evaluate_reconstruction.py \
  --real-root "$REAL_ROOT" \
  --fake-root "$RUN_DIR/generated_image/all_before" \
  --output "$RUN_DIR/generated_image/all_before/reconstruction_metrics.json" \
  --clip-pretrained "$CLIP_H14" \
  --device cuda \
  --image-size 256

echo "=== Evaluating all ==="
python scripts/evaluate_reconstruction.py \
  --real-root "$REAL_ROOT" \
  --fake-root "$RUN_DIR/generated_image/all" \
  --output "$RUN_DIR/generated_image/all/reconstruction_metrics.json" \
  --clip-pretrained "$CLIP_H14" \
  --device cuda \
  --image-size 256

echo "Job ended at $(date)"
