#!/bin/bash
# Submit from the repository root:
#   sbatch --dependency=afterok:<train_job_id> task2/slurm_scripts/07_multiseed_align.sh
#
#SBATCH -p emergency_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --time=04:00:00
#SBATCH -J t2_align_s%a
#SBATCH --array=0-4
#SBATCH -o task2/slurm_scripts/logs/%x-%j.out
#SBATCH -e task2/slurm_scripts/logs/%x-%j.err

SEED=$SLURM_ARRAY_TASK_ID
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK2_DIR="$(dirname "$SCRIPT_DIR")"
EXP_NAME="intra-subject_cogcappro_EEGProjectLayer_multimodal_cogcap_list_ViT-H-14"
EXP_DIR="$TASK2_DIR/runs/multiseed/$EXP_NAME/sub-01_seed${SEED}"
mkdir -p "$SCRIPT_DIR/logs"
cd "$TASK2_DIR"

set -euo pipefail
for _conda in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    [ -f "$_conda/etc/profile.d/conda.sh" ] && { source "$_conda/etc/profile.d/conda.sh"; break; }
done
conda activate DL_Project
module load cuda/12.6

echo "Job started at $(date) — seed ${SEED}"
python -m src.cogcappro.align.main \
  --exp_dir "$EXP_DIR" \
  --seed ${SEED} \
  --device 0 \
  --epoch 100 \
  --lr 3e-4 \
  --model_type simple
echo "Job ended at $(date)"
