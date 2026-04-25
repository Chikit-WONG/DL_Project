#!/bin/bash
set -euo pipefail

ROOT=/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM
cd "$ROOT"

TRAIN_SEED="${1:-0}"
RUN_NAME="${2:-run01}"
GEN_SEEDS="${3:-0,1,2,3,4,5,6,7,8,9}"

retrieval_job=$(sbatch --parsable slurm_scripts/run_train_retrieval.sh "${TRAIN_SEED}")
recon_train_job=$(sbatch --parsable slurm_scripts/run_train_reconstruction.sh "${TRAIN_SEED}")
eval_ret_job=$(sbatch --parsable --dependency=afterok:${retrieval_job} slurm_scripts/run_eval_retrieval.sh)
gen_job=$(sbatch --parsable --dependency=afterok:${recon_train_job} slurm_scripts/run_generate_recon.sh ./models/contrast/ATMS/sub-01/LATEST_RECONSTRUCTION/40.pth "${RUN_NAME}" "${GEN_SEEDS}")
eval_rec_job=$(sbatch --parsable --dependency=afterok:${gen_job} slurm_scripts/run_eval_reconstruction.sh "./outputs/reconstructions/${RUN_NAME}" "${RUN_NAME}")

echo "train_retrieval=${retrieval_job}"
echo "train_reconstruction=${recon_train_job}"
echo "eval_retrieval=${eval_ret_job}"
echo "generate_recon=${gen_job}"
echo "eval_reconstruction=${eval_rec_job}"
