#!/bin/bash
#SBATCH -p a128m512u
#SBATCH -n 2
#SBATCH --time=00:30:00
#SBATCH -J cogcap_summary_v2
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.err

REPO=/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP
mkdir -p "$REPO/slurm_scripts/logs"
cd "$REPO"

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
set -euo pipefail

echo "Job started at $(date)"
python scripts/summarize_results.py \
  --runs-root "$REPO/runs/full_v2" \
  --output "$REPO/runs/summary_metrics_v2.json"
echo "Job ended at $(date)"
