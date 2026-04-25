#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version1/logs/train_phase2_%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version1/logs/train_phase2_%j.err
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version1
#SBATCH --time=08:00:00

set -eo pipefail
mkdir -p logs checkpoints outputs

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
module load cuda/12.6

echo "Job started at $(date) on $(hostname)"

python -u codes/train.py \
    --phase 2 \
    --alpha 0.5 \
    --beta 1.0 \
    --epochs 100 \
    --resume checkpoints/phase1_main_best.pt \
    --tag phase2_main \
    --seed 0

echo "Job ended at $(date)"
conda deactivate
