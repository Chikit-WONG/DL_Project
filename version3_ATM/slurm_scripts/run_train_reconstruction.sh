#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM/logs/train_reconstruction_%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM/logs/train_reconstruction_%j.err
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM
#SBATCH --time=08:00:00

set -eo pipefail

mkdir -p logs

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
module load cuda/12.6

SEED="${1:-0}"

# avg_trials=True is enforced in Generation/eegdatasets_leaveone.py:
#   training: 16540 samples (1654 classes x 10 images, averaged over 80 trials)
#   testing:  200 samples  (200 test classes, averaged over 4 trials) -- required by course
# Loss = alpha * MSE(eeg_feat, clip_feat) + (1-alpha) * CLIP_loss  (alpha=0.90)
python -u Generation/ATMS_reconstruction.py \
  --data_path /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data \
  --subjects sub-01 \
  --insubject true \
  --logger false \
  --gpu cuda:0 \
  --epochs 40 \
  --batch_size 64 \
  --lr 3e-4 \
  --seed "${SEED}"

latest_dir=$(find ./models/contrast/ATMS/sub-01 -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
if [ -n "${latest_dir}" ]; then
  ln -sfn "$(basename "${latest_dir}")" ./models/contrast/ATMS/sub-01/LATEST_RECONSTRUCTION
  echo "Updated LATEST_RECONSTRUCTION -> ${latest_dir}"
fi
