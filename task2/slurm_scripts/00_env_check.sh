#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --time=00:10:00
#SBATCH -J cogcap_env
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.err

REPO=/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP
mkdir -p "$REPO/slurm_scripts/logs"
cd "$REPO"

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
set -euo pipefail
module load cuda/12.6

echo "Job started at $(date)"
python -V
python main.py \
  --config configs/cogcappro.yaml \
  --subjects sub-01 \
  --brain_backbone EEGProjectLayer_multimodal_cogcap_list \
  --vision_backbone ViT-H-14 \
  --data_type EEG \
  --devices 0 \
  --print_config
echo "Job ended at $(date)"
