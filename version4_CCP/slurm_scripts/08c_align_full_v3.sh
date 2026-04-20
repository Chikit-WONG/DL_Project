#!/bin/bash
#SBATCH -p long_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --time=02:00:00
#SBATCH -J cogcap_align_v3
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.err

# v3 alignment: fixes mode collapse caused by over-long warmup schedule.
# Code changes applied before this script:
#   - diffusion_pipe.py: warmup steps = max(1, total_steps // 10) instead of hardcoded 100/500
#   - main.py: train batch_size reduced to 512 (was 10240) for more gradient steps per epoch
#   - Collapsed checkpoint renamed to diffusion_model_best_collapsed.pth
# Runs 30 epochs on long_gpu for stable convergence.

REPO=/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP
EXP_DIR=$REPO/runs/full_v2/intra-subject_cogcappro_EEGProjectLayer_multimodal_cogcap_list_ViT-H-14/sub-01_seed0
mkdir -p "$REPO/slurm_scripts/logs"
cd "$REPO"

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
set -euo pipefail
module load cuda/12.6

echo "Job started at $(date)"
python -m src.cogcappro.align.main \
  --exp_dir "$EXP_DIR" \
  --device 0 \
  --epoch 30 \
  --lr 3e-4 \
  --model_type diffusion
echo "Job ended at $(date)"
