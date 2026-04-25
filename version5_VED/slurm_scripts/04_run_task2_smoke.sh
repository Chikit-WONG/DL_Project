#!/bin/bash
#SBATCH -p debug
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version5_VED/slurm_scripts/logs/04_run_task2_smoke_%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version5_VED/slurm_scripts/logs/04_run_task2_smoke_%j.err
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

python scripts/run_task2_pipeline.py \
    --data_root /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data \
    --clip_checkpoint /hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/open_clip_pytorch_model.bin \
    --task1_ckpt /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version5_VED/output/logs/main_eeg_course/Brain_Visual_Encoder_EEG/2026-04-23-16-33/Brain_Visual_Encoder_EEG_sub1_seed21_select.pth \
    --epoch 1 \
    --n_seeds 1 \
    --first_seed 999

echo "Job ended: $(date)"
