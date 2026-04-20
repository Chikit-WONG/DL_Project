#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --time=04:00:00
#SBATCH -J cogcap_align_smoke
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.err

REPO=/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP
EXP_DIR=$REPO/runs/smoke/intra-subject_cogcappro_EEGProjectLayer_multimodal_cogcap_list_ViT-H-14/sub-01_seed0
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
  --epoch 1 \
  --lr 1e-4 \
  --model_type diffusion
echo "Job ended at $(date)"
