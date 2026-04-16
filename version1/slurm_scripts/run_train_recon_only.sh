#!/bin/bash
#SBATCH -p debug
#SBATCH -o temp/train_rec_only_%j.out
#SBATCH -e temp/train_rec_only_%j.err
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project
#SBATCH --time=00:30:00

set -eo pipefail
mkdir -p temp checkpoints outputs

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
module load cuda/12.6

echo "Job started at $(date) on $(hostname)"

# Architecture B - reconstruction-only baseline (alpha=0, beta=1)
python -u codes/train.py \
    --phase 1 \
    --alpha 0.0 \
    --beta 1.0 \
    --epochs 50 \
    --tag archB_reconstruction \
    --seed 0

echo "Job ended at $(date)"
conda deactivate
