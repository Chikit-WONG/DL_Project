#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --time=08:00:00
#SBATCH -J cogcap_gen_before_align
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.err

# Use --use_before_align: condition IP-Adapter on the EEG retrieval model's CLIP-space
# embeddings directly, bypassing the failed diffusion-prior alignment stage.
# Output goes to generated_image/all_before/ instead of generated_image/all/.

REPO=/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP
BASE_DIR=$REPO/runs/full_v2
mkdir -p "$REPO/slurm_scripts/logs"
cd "$REPO"

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
set -euo pipefail
module load cuda/12.6

echo "Job started at $(date)"
python -u -m src.cogcappro.generate_image.batch_generate \
  --base_dir "$BASE_DIR" \
  --config configs/cogcappro.yaml \
  --data_type EEG \
  --modality_mode all \
  --device cuda:0 \
  --subjects sub-01 \
  --use_before_align \
  --resume
echo "Job ended at $(date)"
