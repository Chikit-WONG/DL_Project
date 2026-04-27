#!/bin/bash
#SBATCH --job-name=bpmgd_prior
#SBATCH --partition=i64m1tga40u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=outputs/slurm_prior_%j.out

set -euo pipefail
cd /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version6_BP-MGD
export PYTHONPATH=$PWD/src:$PYTHONPATH
/hpc2hdd/home/ckwong627/miniconda3/bin/conda run --no-capture-output -n test \
  python scripts/03_train_prior.py --mode "${MODE:-dev}" --run-name "${RUN_NAME:-prior_${MODE:-dev}}" --encoder-ckpt "$ENCODER_CKPT" ${EXTRA_ARGS:-}
