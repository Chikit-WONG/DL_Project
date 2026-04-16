#!/bin/bash
#SBATCH -p debug
#SBATCH -o temp/sanity_%j.out
#SBATCH -e temp/sanity_%j.err
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

python -u codes/train.py \
    --phase 1 \
    --alpha 1.0 \
    --beta 0.0 \
    --epochs 5 \
    --batch_size 64 \
    --tag sanity_check \
    --seed 0

echo "Job ended at $(date)"
conda deactivate
