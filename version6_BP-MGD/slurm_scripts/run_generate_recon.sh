#!/bin/bash
#SBATCH --job-name=bpmgd_generate
#SBATCH --partition=i64m1tga40u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm_generate_%j.out

set -euo pipefail
cd /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version6_BP-MGD
export PYTHONPATH=$PWD/src:$PYTHONPATH
cmd=(python scripts/04_generate_recon.py --mode "${MODE:-full_train}" --run-name "${RUN_NAME:-final_test}" --encoder-ckpt "$ENCODER_CKPT" --backend "${BACKEND:-prototype}")
if [[ -n "${PRIOR_CKPT:-}" ]]; then
  cmd+=(--prior-ckpt "$PRIOR_CKPT")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_parts=(${EXTRA_ARGS})
  cmd+=("${extra_parts[@]}")
fi
/hpc2hdd/home/ckwong627/miniconda3/bin/conda run --no-capture-output -n test "${cmd[@]}"
