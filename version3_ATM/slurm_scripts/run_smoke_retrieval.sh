#!/bin/bash
#SBATCH -p debug
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM/logs/smoke_retrieval_%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM/logs/smoke_retrieval_%j.err
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM
#SBATCH --time=00:30:00

set -eo pipefail

mkdir -p logs

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
module load cuda/12.6

python -u Retrieval/smoke_retrieval.py \
  --data_path /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data \
  --num_classes 4 \
  --batch_size 4 \
  --device cuda:0
