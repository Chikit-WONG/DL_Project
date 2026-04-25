#!/bin/bash
set -euo pipefail

ROOT=/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version1
cd "$ROOT"

cache_job=$(sbatch --parsable slurm_scripts/run_cache_clip.sh)
phase1_job=$(sbatch --parsable --dependency=afterok:${cache_job} slurm_scripts/run_train_phase1.sh)
phase2_job=$(sbatch --parsable --dependency=afterok:${phase1_job} slurm_scripts/run_train_phase2.sh)
recon_job=$(sbatch --parsable --dependency=afterok:${phase2_job} slurm_scripts/run_reconstruct.sh)
eval_job=$(sbatch --parsable --dependency=afterok:${recon_job} slurm_scripts/run_evaluate.sh)

echo "cache=${cache_job}"
echo "phase1=${phase1_job}"
echo "phase2=${phase2_job}"
echo "reconstruct=${recon_job}"
echo "evaluate=${eval_job}"
