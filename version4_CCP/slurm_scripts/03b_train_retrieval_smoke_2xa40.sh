#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00
#SBATCH -J cogcap_train_smoke_2xa40
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.err

REPO=/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP
EXP_ROOT=$REPO/runs/smoke_2xa40
EXP_NAME=intra-subject_cogcappro_EEGProjectLayer_multimodal_cogcap_list_ViT-H-14
SUBJECT_RUN=$EXP_ROOT/$EXP_NAME/sub-01_seed0
mkdir -p "$REPO/slurm_scripts/logs"
mkdir -p "$SUBJECT_RUN"
cd "$REPO"

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
set -euo pipefail
module load cuda/12.6

echo "Job started at $(date)"
python main.py \
  --config configs/cogcappro.yaml \
  --subjects sub-01 \
  --devices 0,1 \
  --epoch 1 \
  --vision_backbone ViT-H-14 \
  --data_type EEG \
  --seed 0 \
  --exp_setting intra-subject \
  --brain_backbone EEGProjectLayer_multimodal_cogcap_list \
  --lr 1e-4 \
  --save_dir "$SUBJECT_RUN"
echo "Job ended at $(date)"
