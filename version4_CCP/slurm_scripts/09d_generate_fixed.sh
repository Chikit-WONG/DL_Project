#!/bin/bash
#SBATCH -p emergency_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --time=08:00:00
#SBATCH -J cogcap_gen_fixed
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.err

# Fixes applied vs 09b/09c:
#   1. VAE kept in float32 (no float16 force-back) to prevent NaN overflow
#   2. _prepare_embeddings passes [1,1,1024] (cond-only) when guidance_scale=0.0
#      so IP-Adapter cross-attention receives correct single-token conditioning
# Both all/ and all_before/ are cleared and regenerated from scratch.

REPO=/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP
RUN_DIR=$REPO/runs/full_v2/intra-subject_cogcappro_EEGProjectLayer_multimodal_cogcap_list_ViT-H-14/sub-01_seed0
BASE_DIR=$REPO/runs/full_v2

mkdir -p "$REPO/slurm_scripts/logs"
cd "$REPO"

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
set -euo pipefail
module load cuda/12.6

echo "Job started at $(date)"

# Clear stale all-black images so generation starts fresh
echo "Clearing old generated images..."
rm -f "$RUN_DIR/generated_image/all_before/"*.jpg
rm -f "$RUN_DIR/generated_image/all/"*.jpg

echo "Generating with --use_before_align (pre-alignment EEG embeddings)..."
python -u -m src.cogcappro.generate_image.batch_generate \
  --base_dir "$BASE_DIR" \
  --config configs/cogcappro.yaml \
  --data_type EEG \
  --modality_mode all \
  --device cuda:0 \
  --subjects sub-01 \
  --use_before_align

echo "Generating with aligned embeddings (post-alignment)..."
python -u -m src.cogcappro.generate_image.batch_generate \
  --base_dir "$BASE_DIR" \
  --config configs/cogcappro.yaml \
  --data_type EEG \
  --modality_mode all \
  --device cuda:0 \
  --subjects sub-01

echo "Job ended at $(date)"
