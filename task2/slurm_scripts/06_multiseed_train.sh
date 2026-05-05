#!/bin/bash
# Submit from the repository root:
#   sbatch task2/slurm_scripts/06_multiseed_train.sh
#
#SBATCH -p emergency_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --time=24:00:00
#SBATCH -J t2_train_s%a
#SBATCH --array=0-4
#SBATCH -o task2/slurm_scripts/logs/%x-%j.out
#SBATCH -e task2/slurm_scripts/logs/%x-%j.err

SEED=$SLURM_ARRAY_TASK_ID
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK2_DIR="$(dirname "$SCRIPT_DIR")"
EXP_ROOT="$TASK2_DIR/runs/multiseed"
EXP_NAME="intra-subject_cogcappro_EEGProjectLayer_multimodal_cogcap_list_ViT-H-14"
SAVE_DIR="$EXP_ROOT/$EXP_NAME/sub-01_seed${SEED}"
mkdir -p "$SCRIPT_DIR/logs" "$SAVE_DIR"
cd "$TASK2_DIR"

set -euo pipefail
for _conda in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    [ -f "$_conda/etc/profile.d/conda.sh" ] && { source "$_conda/etc/profile.d/conda.sh"; break; }
done
conda activate DL_Project
module load cuda/12.6

echo "Job started at $(date) — seed ${SEED}"
python main.py \
  --config configs/cogcappro.yaml \
  --subjects sub-01 \
  --devices 0 \
  --vision_backbone ViT-H-14 \
  --data_type EEG \
  --seed ${SEED} \
  --exp_setting intra-subject \
  --brain_backbone EEGProjectLayer_multimodal_cogcap_list \
  --lr 1e-4 \
  --save_dir "$SAVE_DIR"
echo "Job ended at $(date)"
