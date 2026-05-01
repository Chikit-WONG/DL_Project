#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version7_VED_plus_EVNet/slurm_scripts/logs/11_train_8blur_evnet_vith14_split_%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version7_VED_plus_EVNet/slurm_scripts/logs/11_train_8blur_evnet_vith14_split_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version7_VED_plus_EVNet

set -eo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
type unclash >/dev/null 2>&1 && unclash || true

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi

# Ablation: 8 blur + EVNet fixed, ViT-H/14 backbone, 95/5 train/val split
python main_eeg_course.py \
    --blur_config 8 \
    --use_evnet \
    --blur_prefix MultiBlur_ViTH14 \
    --evnet_prefix EVNet_ViTH14 \
    --epoch 200 \
    --train_batch_size 1024 \
    --lr 0.001 \
    --n_seeds 10 \
    --first_seed 21 \
    --feature_path output/Image_feature \
    --output_dir output/logs/8blur_evnet_vith14_split

echo "Job ended: $(date)"
