#!/bin/bash
# Submit from the repository root:
#   sbatch --dependency=afterok:<align_job_id> task2/slurm_scripts/08_multiseed_generate.sh
#
# batch_generate auto-discovers all sub-01_seed* directories under base_dir.
#
#SBATCH -p emergency_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --time=12:00:00
#SBATCH -J t2_generate_all
#SBATCH -o task2/slurm_scripts/logs/%x-%j.out
#SBATCH -e task2/slurm_scripts/logs/%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK2_DIR="$(dirname "$SCRIPT_DIR")"
BASE_DIR="$TASK2_DIR/runs/multiseed"
mkdir -p "$SCRIPT_DIR/logs"
cd "$TASK2_DIR"

set -euo pipefail
for _conda in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    [ -f "$_conda/etc/profile.d/conda.sh" ] && { source "$_conda/etc/profile.d/conda.sh"; break; }
done
conda activate DL_Project
module load cuda/12.6

echo "Job started at $(date)"

echo "=== Generating pre-alignment images (all_before) for all seeds ==="
python -m src.cogcappro.generate_image.batch_generate \
  --base_dir "$BASE_DIR" \
  --config configs/cogcappro.yaml \
  --data_type EEG \
  --modality_mode all \
  --device cuda:0 \
  --subjects sub-01 \
  --use_before_align

echo "=== Generating post-alignment images (all) for all seeds ==="
python -m src.cogcappro.generate_image.batch_generate \
  --base_dir "$BASE_DIR" \
  --config configs/cogcappro.yaml \
  --data_type EEG \
  --modality_mode all \
  --device cuda:0 \
  --subjects sub-01

echo "Job ended at $(date)"
