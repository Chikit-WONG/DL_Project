#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/main/task1/slurm_scripts/logs/05_full_train_12blur_evnet_%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/main/task1/slurm_scripts/logs/05_full_train_12blur_evnet_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/main/task1

set -eo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
type unclash >/dev/null 2>&1 && unclash || true

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi

# Phase B: 12 blur levels + EVNet fixed, full training set (no val split)
# 'select' checkpoint = last epoch; 'best' checkpoint = best test epoch
python main_eeg_course.py \
    --blur_config 12 \
    --use_evnet \
    --use_full_train \
    --epoch 200 \
    --train_batch_size 1024 \
    --lr 0.001 \
    --n_seeds 10 \
    --first_seed 21 \
    --feature_path output/Image_feature \
    --output_dir output/logs/12blur_evnet_full

echo "Job ended: $(date)"
