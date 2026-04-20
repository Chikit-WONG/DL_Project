#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM/logs/eval_reconstruction_%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM/logs/eval_reconstruction_%j.err
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM
#SBATCH --time=02:00:00

# Usage: edit TENSORS_PATH before submitting
# Example: sbatch run_eval_reconstruction.sh ./outputs/reconstructions/run01/recon_tensors.pt

set -eo pipefail
mkdir -p logs outputs

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
module load cuda/12.6

TENSORS_PATH="${1:-./outputs/reconstructions/run01/recon_tensors.pt}"
RUN_NAME="${2:-run01}"

echo "Evaluating tensors: ${TENSORS_PATH}"

python -u eval/eval_reconstruction_metrics.py \
  --tensors_path "${TENSORS_PATH}" \
  --output_csv   "./outputs/reconstruction_eval_${RUN_NAME}.csv" \
  --device       cuda:0 \
  --num_seeds    1
