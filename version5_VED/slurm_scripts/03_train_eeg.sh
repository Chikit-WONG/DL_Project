#!/bin/bash
#SBATCH -p debug
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version5_VED/slurm_scripts/logs/03_train_eeg_%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version5_VED/slurm_scripts/logs/03_train_eeg_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version5_VED

set -eo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
type unclash >/dev/null 2>&1 && unclash || true

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi

python main_eeg_course.py \
    --epoch 1 \
    --train_batch_size 1024 \
    --lr 0.001 \
    --n_seeds 1 \
    --first_seed 999 \
    --feature_path output/Image_feature \
    --output_dir output/logs/main_eeg_course

echo "Job ended: $(date)"
