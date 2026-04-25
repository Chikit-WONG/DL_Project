#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM/logs/eval_retrieval_%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM/logs/eval_retrieval_%j.err
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM
#SBATCH --time=02:00:00

# Usage: edit CHECKPOINT before submitting
# Example: sbatch run_eval_retrieval.sh ./models/contrast/ATMS/sub-01/<time>/40.pth

set -eo pipefail
mkdir -p logs outputs

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
module load cuda/12.6

CHECKPOINT="${1:-./models/contrast/ATMS/sub-01/LATEST_RETRIEVAL/40.pth}"
RUN_NAME="${2:-run01}"

echo "Evaluating checkpoint: ${CHECKPOINT}"

python -u eval/eval_retrieval_200way.py \
  --checkpoint   "${CHECKPOINT}" \
  --data_path    /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data \
  --subject      sub-01 \
  --clip_cache_dir /hpc2hdd/home/ckwong627/workdir/models \
  --output_csv   "./outputs/retrieval_eval_${RUN_NAME}.csv" \
  --device       cuda:0
