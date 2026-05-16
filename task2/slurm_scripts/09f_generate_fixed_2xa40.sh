#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --time=08:00:00
#SBATCH -J cogcap_gen_fixed_2xa40
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.err

REPO=/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP
RUN_DIR=$REPO/runs/full_v2_2xa40/intra-subject_cogcappro_EEGProjectLayer_multimodal_cogcap_list_ViT-H-14/sub-01_seed0
BASE_DIR=$REPO/runs/full_v2_2xa40

mkdir -p "$REPO/slurm_scripts/logs"
cd "$REPO"

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
set -euo pipefail
module load cuda/12.6

echo "Job started at $(date)"
rm -f "$RUN_DIR/generated_image/all_before/"*.jpg
rm -f "$RUN_DIR/generated_image/all/"*.jpg

echo "Generating with --use_before_align ..."
python -u -m src.cogcappro.generate_image.batch_generate \
  --base_dir "$BASE_DIR" \
  --config configs/cogcappro.yaml \
  --data_type EEG \
  --modality_mode all \
  --device cuda:0 \
  --subjects sub-01 \
  --use_before_align

echo "Generating with aligned embeddings ..."
python -u -m src.cogcappro.generate_image.batch_generate \
  --base_dir "$BASE_DIR" \
  --config configs/cogcappro.yaml \
  --data_type EEG \
  --modality_mode all \
  --device cuda:0 \
  --subjects sub-01

echo "Job ended at $(date)"
