#!/bin/bash
#SBATCH -p debug
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version7_VED_plus_EVNet/slurm_scripts/logs/07_gen_evnet_gap_features_%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version7_VED_plus_EVNet/slurm_scripts/logs/07_gen_evnet_gap_features_%j.err
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version7_VED_plus_EVNet

set -eo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
type unclash >/dev/null 2>&1 && unclash || true

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi

# Generate EVNet GlobalAvgPool + Linear(512->1024) features (no RN50)
# Outputs: output/Image_feature/EVNet_gap_train.pt and _test.pt
python preprocess/process_image_course.py \
    --clip_checkpoint /hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/open_clip_pytorch_model.bin \
    --save_dir output/Image_feature \
    --evnet_mode gap

echo "Job ended: $(date)"
