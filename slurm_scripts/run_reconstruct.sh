#!/bin/bash
#SBATCH -p debug
#SBATCH -o temp/reconstruct_%j.out
#SBATCH -e temp/reconstruct_%j.err
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project
#SBATCH --time=00:30:00

set -eo pipefail
mkdir -p temp outputs

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
module load cuda/12.6

echo "Job started at $(date) on $(hostname)"

# Default: reconstruct using the phase-2 main checkpoint, all 10 seeds.
# Pass --ckpt and --seeds via env vars or edit this file as needed.
python -u codes/reconstruct.py \
    --ckpt checkpoints/phase2_main_best.pt \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --tag phase2_main

echo "Job ended at $(date)"
conda deactivate
