#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --time=04:00:00
#SBATCH -J cogcap_embed
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.err

REPO=/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP
OUTPUT_DIR=/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data/converted_for_cogcappro/ThingsEEG/Image_feature_new/data_features
mkdir -p "$REPO/slurm_scripts/logs"
cd "$REPO"

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
set -euo pipefail
module load cuda/12.6

echo "Job started at $(date)"
python scripts/prepare_diffusion_embeddings.py \
  --config configs/cogcappro.yaml \
  --data-type EEG \
  --sd-path /hpc2hdd/home/ckwong627/workdir/models/sdxl-turbo \
  --ip-adapter-path /hpc2hdd/home/ckwong627/workdir/models/IP-Adapter \
  --device cuda:0 \
  --output-dir "$OUTPUT_DIR"
echo "Job ended at $(date)"
