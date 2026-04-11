#!/bin/bash
#SBATCH -p debug
#SBATCH -o temp/recon_s0_%j.out
#SBATCH -e temp/recon_s0_%j.err
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

python -u codes/reconstruct.py \
    --ckpt checkpoints/phase2_main_best.pt \
    --seeds 0 \
    --num_inference_steps 20 \
    --tag phase2_main

echo "Job ended at $(date)"
conda deactivate
