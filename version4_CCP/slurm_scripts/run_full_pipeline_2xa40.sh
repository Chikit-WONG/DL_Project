#!/bin/bash
set -euo pipefail

ROOT=/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP
cd "$ROOT"

train_job=$(sbatch --parsable slurm_scripts/07c_train_retrieval_full_2xa40.sh)
align_job=$(sbatch --parsable --dependency=afterok:${train_job} slurm_scripts/08e_simple_align_2xa40.sh)
gen_job=$(sbatch --parsable --dependency=afterok:${align_job} slurm_scripts/09f_generate_fixed_2xa40.sh)
eval_job=$(sbatch --parsable --dependency=afterok:${gen_job} slurm_scripts/10f_eval_full_both_2xa40.sh)
summary_job=$(sbatch --parsable --dependency=afterok:${eval_job} slurm_scripts/11c_summary_v2_2xa40.sh)

echo "train=${train_job}"
echo "align=${align_job}"
echo "generate=${gen_job}"
echo "eval=${eval_job}"
echo "summary=${summary_job}"
