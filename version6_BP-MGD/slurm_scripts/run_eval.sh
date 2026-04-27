#!/bin/bash
#SBATCH --job-name=bpmgd_eval
#SBATCH --partition=i64m512u
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/slurm_eval_%j.out

set -euo pipefail
cd /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version6_BP-MGD
export PYTHONPATH=$PWD/src:$PYTHONPATH
/hpc2hdd/home/ckwong627/miniconda3/bin/conda run --no-capture-output -n test \
  python scripts/05_eval_recon.py --run-name "${RUN_NAME:-final_test}" ${EXTRA_ARGS:-}
