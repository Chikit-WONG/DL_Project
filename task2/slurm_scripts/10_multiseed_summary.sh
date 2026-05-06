#!/bin/bash
# Submit from the repository root:
#   sbatch --dependency=afterok:<eval_job_id> task2/slurm_scripts/10_multiseed_summary.sh
#
#SBATCH -p debug
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH -J t2_summary
#SBATCH -o task2/slurm_scripts/logs/%x-%j.out
#SBATCH -e task2/slurm_scripts/logs/%x-%j.err

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TASK2_DIR="$REPO_ROOT/task2"
EXP_NAME="intra-subject_cogcappro_EEGProjectLayer_multimodal_cogcap_list_ViT-H-14"
mkdir -p "$TASK2_DIR/slurm_scripts/logs"
cd "$TASK2_DIR"

for _conda in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    [ -f "$_conda/etc/profile.d/conda.sh" ] && { source "$_conda/etc/profile.d/conda.sh"; break; }
done
conda activate DL_Project

echo "Job started at $(date)"
python scripts/summarize_results.py \
  --runs-root "runs/multiseed/$EXP_NAME" \
  --output runs/multiseed/summary.json
echo "Summary saved to runs/multiseed/summary.json"
echo "Job ended at $(date)"
