#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM/logs/eval_reconstruction_%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM/logs/eval_reconstruction_%j.err
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM
#SBATCH --time=08:00:00

# Usage:
#   sbatch run_eval_reconstruction.sh ./outputs/reconstructions/run01 run01
# or
#   sbatch run_eval_reconstruction.sh ./outputs/reconstructions/run01/seed00/recon_tensors.pt run01

set -eo pipefail
mkdir -p logs outputs

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
module load cuda/12.6

TENSORS_INPUT="${1:-./outputs/reconstructions/run01}"
RUN_NAME="${2:-run01}"

if [ -d "${TENSORS_INPUT}" ]; then
  echo "Evaluating tensor directory: ${TENSORS_INPUT}"
  python -u eval/eval_reconstruction_metrics.py \
    --tensors_dir  "${TENSORS_INPUT}" \
    --output_csv   "./outputs/reconstruction_eval_${RUN_NAME}.csv" \
    --device       cuda:0
else
  echo "Evaluating tensor file: ${TENSORS_INPUT}"
  python -u eval/eval_reconstruction_metrics.py \
    --tensors_path "${TENSORS_INPUT}" \
    --output_csv   "./outputs/reconstruction_eval_${RUN_NAME}.csv" \
    --device       cuda:0 \
    --num_seeds    1
fi
