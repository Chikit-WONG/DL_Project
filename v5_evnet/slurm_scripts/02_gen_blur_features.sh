#!/bin/bash
#SBATCH -p debug
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/repository/VisualEEGDecoding/slurm_scripts/logs/02_gen_blur_features_%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/repository/VisualEEGDecoding/slurm_scripts/logs/02_gen_blur_features_%j.err
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/repository/VisualEEGDecoding

set -eo pipefail

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test

echo "Job started: $(date)"
echo "Node: $(hostname)"

python preprocess/process_image_course.py

echo "Job ended: $(date)"
