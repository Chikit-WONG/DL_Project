#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/logs/reconstruct_%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/logs/reconstruct_%j.err
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2
#SBATCH --time=08:00:00

set -eo pipefail
mkdir -p logs checkpoints results cache

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
module load cuda/12.6

python -u codes/reconstruct.py \
    --encoder_ckpt checkpoints/v2_final_best.pt \
    --prior_ckpt checkpoints/v2_prior_best.pt \
    --tag v2_final \
    --seeds 0 1 2 3 4 5 6 7 8 9
