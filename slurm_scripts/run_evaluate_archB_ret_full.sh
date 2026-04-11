#!/bin/bash
#SBATCH -p debug
#SBATCH -o temp/eval_archB_ret_full_%j.out
#SBATCH -e temp/eval_archB_ret_full_%j.err
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

python -u codes/evaluate.py \
    --ckpt checkpoints/archB_retrieval_best.pt \
    --recon_tag archB_retrieval

echo "Job ended at $(date)"
conda deactivate
