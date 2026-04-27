#!/bin/bash
#SBATCH --job-name=bpmgd_cache
#SBATCH --partition=i64m1tga40u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm_cache_%j.out

set -euo pipefail
cd /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version6_BP-MGD
export PYTHONPATH=$PWD/src:$PYTHONPATH
/hpc2hdd/home/ckwong627/miniconda3/bin/conda run --no-capture-output -n test \
  python scripts/01_cache_train_features.py --mode "${MODE:-dev}" --run-name "${RUN_NAME:-cache_${MODE:-dev}}" --device cuda ${EXTRA_ARGS:-}
