#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --time=08:00:00
#SBATCH -J cogcap_eval_full_both_2xa40
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.err

REPO=/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP
RUN_DIR=$REPO/runs/full_v2_2xa40/intra-subject_cogcappro_EEGProjectLayer_multimodal_cogcap_list_ViT-H-14/sub-01_seed0
REAL_ROOT=/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data/converted_for_cogcappro/ThingsEEG/Image_set_Resize/test_images
mkdir -p "$REPO/slurm_scripts/logs"
cd "$REPO"

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
set -euo pipefail
module load cuda/12.6

echo "Job started at $(date)"
python scripts/evaluate_reconstruction.py \
  --real-root "$REAL_ROOT" \
  --fake-root "$RUN_DIR/generated_image/all_before" \
  --output "$RUN_DIR/generated_image/all_before/reconstruction_metrics.json" \
  --device cuda \
  --image-size 256

python scripts/evaluate_reconstruction.py \
  --real-root "$REAL_ROOT" \
  --fake-root "$RUN_DIR/generated_image/all" \
  --output "$RUN_DIR/generated_image/all/reconstruction_metrics.json" \
  --device cuda \
  --image-size 256
echo "Job ended at $(date)"
