#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --time=04:00:00
#SBATCH -J cogcap_gen_smoke
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.err

REPO=/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP
BASE_DIR=$REPO/runs/smoke
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
  --resume
echo "Job ended at $(date)"
