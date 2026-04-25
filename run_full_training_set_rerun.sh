#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_FILE="$ROOT/plan/full_training_set_rerun_jobs_latest.txt"

cleanup_v3_cache() {
  rm -f /hpc2hdd/home/ckwong627/workdir/models/ViT-H-14_features_train.pt
  rm -f /hpc2hdd/home/ckwong627/workdir/models/ViT-H-14_features_test.pt
}

submit_version1() {
  cd "$ROOT/version1"
  bash slurm_scripts/run_full_pipeline.sh
}

submit_version2() {
  cd "$ROOT/version2"
  local cache warm multi fine prior recon eval
  cache=$(sbatch --parsable slurm_scripts/run_cache_backbone_features.sh)
  warm=$(sbatch --parsable --dependency=afterok:${cache} slurm_scripts/run_train_encoder_warmup.sh)
  multi=$(sbatch --parsable --dependency=afterok:${warm} slurm_scripts/run_train_encoder_multitarget.sh)
  fine=$(sbatch --parsable --dependency=afterok:${multi} slurm_scripts/run_train_encoder_finetune.sh)
  prior=$(sbatch --parsable --dependency=afterok:${fine} slurm_scripts/run_train_prior.sh)
  recon=$(sbatch --parsable --dependency=afterok:${prior} slurm_scripts/run_reconstruct_all.sh)
  eval=$(sbatch --parsable --dependency=afterok:${recon} slurm_scripts/run_evaluate.sh)
  cat <<EOF
cache=${cache}
warmup=${warm}
multitarget=${multi}
finetune=${fine}
prior=${prior}
reconstruct=${recon}
evaluate=${eval}
EOF
}

submit_version3() {
  cleanup_v3_cache
  cd "$ROOT/version3_ATM"
  bash slurm_scripts/run_full_pipeline.sh
}

submit_version4() {
  cd "$ROOT/version4_CCP"
  local env prep embed train align gen eval summary
  env=$(sbatch --parsable slurm_scripts/00_env_check.sh)
  prep=$(sbatch --parsable --dependency=afterok:${env} slurm_scripts/01_prepare_course_data.sh)
  embed=$(sbatch --parsable --dependency=afterok:${prep} slurm_scripts/02b_reprepare_diffusion_embeddings.sh)
  train=$(sbatch --parsable --dependency=afterok:${embed} slurm_scripts/07b_train_retrieval_full_v2.sh)
  align=$(sbatch --parsable --dependency=afterok:${train} slurm_scripts/08d_simple_align.sh)
  gen=$(sbatch --parsable --dependency=afterok:${align} slurm_scripts/09d_generate_fixed.sh)
  eval=$(sbatch --parsable --dependency=afterok:${gen} slurm_scripts/10e_eval_full_both.sh)
  summary=$(sbatch --parsable --dependency=afterok:${eval} slurm_scripts/11b_summary_v2.sh)
  cat <<EOF
env=${env}
prepare=${prep}
embed=${embed}
train=${train}
align=${align}
generate=${gen}
evaluate=${eval}
summary=${summary}
EOF
}

{
  echo "timestamp=$(date '+%F %T %Z')"
  echo "[version1]"
  submit_version1
  echo "[version2]"
  submit_version2
  echo "[version3_ATM]"
  submit_version3
  echo "[version4_CCP]"
  submit_version4
} | tee "$JOB_FILE"

echo
echo "Saved job ids to: $JOB_FILE"
