#!/usr/bin/env bash
# ============================================================================
# Grid runner v2 (force-retrain).
#
# v1 had a silent-no-op problem: 5/8 variants ended up with identical metrics
# because alignment was not re-run, and BLUR_KERNEL_SIZE / SYSTEM_G are
# retrieval-stage knobs that generate-only variants can't actually flip.
#
# v2 fixes that with three explicit reuse strategies and aggressive cleanup
# before each variant runs:
#
#   - generate-only variants: copy retrieval + align from base, wipe
#     generated_image/. SKIP_TRAIN=1 SKIP_ALIGN=1.
#   - align-changing variants: copy retrieval ONLY, then explicitly delete
#     align/ and generated_image/. SKIP_TRAIN=1 SKIP_ALIGN=0.
#   - retrieval-changing variants (pp_k21_g1): wipe the entire variant dir
#     and full retrain. SKIP_TRAIN=0 SKIP_ALIGN=0.
#
# Variant -> strategy:
#   base               full pipeline
#   pp_k21_g1          retrieval-changing  (BLUR/SYSTEM_G affect retrieval)
#   pp_none            generate-only
#   gen_g30            generate-only
#   gen_steps50_g15    generate-only
#   align_e40          align-changing
#   align_lr1e4        align-changing
#   emb_g30            align-changing
#
# All variants are idempotent: re-running this script with the same
# arguments wipes the variant's directory before invoking the inner
# pipeline, so partial or failed earlier attempts can't pollute results.
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

INNER_SCRIPT="${INNER_SCRIPT:-$(dirname "${BASH_SOURCE[0]}")/run_full_experiment.sh}"

export EXP_SETTING="${EXP_SETTING:-intra-subject}"
export SUBJECT="${SUBJECT:-sub-01}"
export SEED="${SEED:-0}"
export DATA_TYPE="${DATA_TYPE:-EEG}"
export VISION_BACKBONE="${VISION_BACKBONE:-ViT-H-14}"
export BRAIN_BACKBONE="${BRAIN_BACKBONE:-EEGProjectLayer_multimodal_cogcap_list}"
export EXP_NAME="${EXP_SETTING}_cogcappro_${BRAIN_BACKBONE}_${VISION_BACKBONE}"
RUNS_ROOT="${RUNS_ROOT:-$REPO/runs}"
export RUNS_ROOT

DEFAULT_VARIANTS=(
  "base"
  "pp_none"
  "gen_g30"
  "gen_steps5"
  "gen_steps50_g15"
  "align_e40"
  "align_lr1e4"
  "emb_g30"
)
# pp_k21_g1 is intentionally NOT in the default list: it changes
# retrieval-stage knobs (BLUR_KERNEL_SIZE, SYSTEM_G), which forces a full
# retrain. If you want it, pass it explicitly:
#   VARIANTS=("pp_k21_g1") bash scripts/run_grid_experiments.sh
VARIANTS=("${VARIANTS[@]:-${DEFAULT_VARIANTS[@]}}")

# Multi-seed support. By default we run one seed per variant (fast). To run
# multiple seeds for a chosen variant (matching the report's 5-seed protocol),
# set SEEDS to a space-separated list:
#   SEEDS="0 1 2 3 4" VARIANTS=("base") bash scripts/run_grid_experiments.sh
SEEDS="${SEEDS:-$SEED}"

BASE_TAG="grid_base"
BASE_SUBJECT_RUN="$RUNS_ROOT/$BASE_TAG/$EXP_NAME/${SUBJECT}_seed${SEED}"

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

variant_subject_run() {
  local tag="$1"
  echo "$RUNS_ROOT/grid_${tag}/$EXP_NAME/${SUBJECT}_seed${SEED}"
}

require_base() {
  if [[ ! -f "$BASE_SUBJECT_RUN/generated_datasets/generated_embeddings.pt" ]]; then
    echo "ERROR: base run not found at $BASE_SUBJECT_RUN" >&2
    echo "Run the 'base' variant first." >&2
    exit 1
  fi
}

run_inner() {
  bash "$INNER_SCRIPT" "${1:-all}"
}

# ---------------------------------------------------------------------------
# Three reuse strategies. Each function leaves $SUBJECT_RUN in the exact
# state the inner script expects.
# ---------------------------------------------------------------------------

# Generation-only: copy retrieval + align, wipe generated_image.
prepare_generate_only() {
  local target_run="$1"
  require_base
  rm -rf "$target_run"
  mkdir -p "$(dirname "$target_run")"
  rsync -a \
    --exclude='generated_image/' \
    --exclude='comparison/' \
    "$BASE_SUBJECT_RUN"/ "$target_run"/
}

# Align-changing: copy retrieval only, then explicitly wipe align/ and
# generated_image/ so the inner script must recompute them from scratch.
prepare_align_rerun() {
  local target_run="$1"
  require_base
  rm -rf "$target_run"
  mkdir -p "$(dirname "$target_run")"
  rsync -a \
    --exclude='generated_image/' \
    --exclude='comparison/' \
    --exclude='align/' \
    "$BASE_SUBJECT_RUN"/ "$target_run"/
  # Belt-and-suspenders cleanup in case rsync exclude misfires.
  rm -rf "$target_run/align"
  rm -rf "$target_run/generated_image"
  echo "[prepare_align_rerun] $target_run after cleanup:"
  ls -la "$target_run" || true
  if [[ -d "$target_run/align" ]]; then
    echo "ERROR: align/ still exists after cleanup; aborting." >&2
    exit 1
  fi
}

# Retrieval-changing: full clean slate.
prepare_full_retrain() {
  local target_run="$1"
  rm -rf "$target_run"
  mkdir -p "$(dirname "$target_run")"
}

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

run_variant_base() {
  export RUN_TAG="$BASE_TAG"
  export EXP_ROOT="$RUNS_ROOT/$RUN_TAG"
  export SUBJECT_RUN="$BASE_SUBJECT_RUN"
  export GENERATED_CONFIG="$REPO/configs/generated_${RUN_TAG}.yaml"

  export BLUR_KERNEL_SIZE=31
  export SYSTEM_G=2
  export MODALITY_MODE=all
  export GENERATE_GUIDANCE=1.5
  export GENERATE_STEPS=30
  export GENERATE_POSTPROCESS=none
  export ALIGN_EPOCHS=60
  export ALIGN_LR=3e-4
  export ALIGN_EMBEDDING_GUIDANCE=5.0
  export SKIP_TRAIN=0
  export SKIP_ALIGN=0

  if [[ "${FORCE_REBUILD_BASE:-0}" == "1" ]]; then
    rm -rf "$BASE_SUBJECT_RUN"
  fi
  run_inner all
}

# ---- generation-only ------------------------------------------------------

setup_gen_only() {
  local tag="$1"
  export RUN_TAG="grid_${tag}"
  export EXP_ROOT="$RUNS_ROOT/$RUN_TAG"
  export SUBJECT_RUN="$(variant_subject_run "$tag")"
  export GENERATED_CONFIG="$REPO/configs/generated_${RUN_TAG}.yaml"
  prepare_generate_only "$SUBJECT_RUN"
  export SKIP_TRAIN=1
  export SKIP_ALIGN=1
}

run_variant_pp_none() {
  setup_gen_only "pp_none"
  export BLUR_KERNEL_SIZE=31
  export SYSTEM_G=2
  export MODALITY_MODE=all
  export GENERATE_GUIDANCE=1.5
  export GENERATE_STEPS=30
  export GENERATE_POSTPROCESS=none
  export ALIGN_EPOCHS=60
  export ALIGN_LR=3e-4
  export ALIGN_EMBEDDING_GUIDANCE=5.0
  run_inner all
}

run_variant_gen_g30() {
  setup_gen_only "gen_g30"
  export BLUR_KERNEL_SIZE=31
  export SYSTEM_G=2
  export MODALITY_MODE=all
  export GENERATE_GUIDANCE=3.0
  export GENERATE_STEPS=30
  export GENERATE_POSTPROCESS=none
  export ALIGN_EPOCHS=60
  export ALIGN_LR=3e-4
  export ALIGN_EMBEDDING_GUIDANCE=5.0
  run_inner all
}

run_variant_gen_steps50_g15() {
  setup_gen_only "gen_steps50_g15"
  export BLUR_KERNEL_SIZE=31
  export SYSTEM_G=2
  export MODALITY_MODE=all
  export GENERATE_GUIDANCE=1.5
  export GENERATE_STEPS=50
  export GENERATE_POSTPROCESS=none
  export ALIGN_EPOCHS=60
  export ALIGN_LR=3e-4
  export ALIGN_EMBEDDING_GUIDANCE=5.0
  run_inner all
}

# SDXL-turbo's native configuration: 5 inference steps, guidance=0.
# This matches the original CogCapPro report and is the most likely
# configuration for them to have hit SSIM 0.356. Strong candidate for SSIM.
run_variant_gen_steps5() {
  setup_gen_only "gen_steps5"
  export BLUR_KERNEL_SIZE=31
  export SYSTEM_G=2
  export MODALITY_MODE=all
  export GENERATE_GUIDANCE=0.0
  export GENERATE_STEPS=5
  export GENERATE_POSTPROCESS=none
  export ALIGN_EPOCHS=60
  export ALIGN_LR=3e-4
  export ALIGN_EMBEDDING_GUIDANCE=5.0
  run_inner all
}

# ---- align-changing -------------------------------------------------------

setup_align_rerun() {
  local tag="$1"
  export RUN_TAG="grid_${tag}"
  export EXP_ROOT="$RUNS_ROOT/$RUN_TAG"
  export SUBJECT_RUN="$(variant_subject_run "$tag")"
  export GENERATED_CONFIG="$REPO/configs/generated_${RUN_TAG}.yaml"
  prepare_align_rerun "$SUBJECT_RUN"
  export SKIP_TRAIN=1
  export SKIP_ALIGN=0
  # Belt-and-suspenders #3: tell the inner script to wipe any align/ dir
  # it finds before invoking align main. With the wrapper's rsync exclude
  # plus the inner script's rm -rf, three layers of defense ensure the
  # align stage cannot silently reuse old outputs.
  export FORCE_ALIGN_RERUN=1
}

run_variant_align_e40() {
  setup_align_rerun "align_e40"
  export BLUR_KERNEL_SIZE=31
  export SYSTEM_G=2
  export MODALITY_MODE=all
  export GENERATE_GUIDANCE=1.5
  export GENERATE_STEPS=30
  export GENERATE_POSTPROCESS=none
  export ALIGN_EPOCHS=40
  export ALIGN_LR=3e-4
  export ALIGN_EMBEDDING_GUIDANCE=5.0
  run_inner all
}

run_variant_align_lr1e4() {
  setup_align_rerun "align_lr1e4"
  export BLUR_KERNEL_SIZE=31
  export SYSTEM_G=2
  export MODALITY_MODE=all
  export GENERATE_GUIDANCE=1.5
  export GENERATE_STEPS=30
  export GENERATE_POSTPROCESS=none
  export ALIGN_EPOCHS=80
  export ALIGN_LR=1e-4
  export ALIGN_EMBEDDING_GUIDANCE=5.0
  run_inner all
}

run_variant_emb_g30() {
  setup_align_rerun "emb_g30"
  export BLUR_KERNEL_SIZE=31
  export SYSTEM_G=2
  export MODALITY_MODE=all
  export GENERATE_GUIDANCE=1.5
  export GENERATE_STEPS=30
  export GENERATE_POSTPROCESS=none
  export ALIGN_EPOCHS=60
  export ALIGN_LR=3e-4
  export ALIGN_EMBEDDING_GUIDANCE=3.0
  run_inner all
}

# ---- retrieval-changing ---------------------------------------------------
# BLUR_KERNEL_SIZE and SYSTEM_G feed the retrieval data pipeline via the
# generated config, so changing them requires a full retrain.

run_variant_pp_k21_g1() {
  export RUN_TAG="grid_pp_k21_g1"
  export EXP_ROOT="$RUNS_ROOT/$RUN_TAG"
  export SUBJECT_RUN="$(variant_subject_run "pp_k21_g1")"
  export GENERATED_CONFIG="$REPO/configs/generated_${RUN_TAG}.yaml"
  prepare_full_retrain "$SUBJECT_RUN"

  export BLUR_KERNEL_SIZE=21
  export SYSTEM_G=1
  export MODALITY_MODE=all
  export GENERATE_GUIDANCE=1.5
  export GENERATE_STEPS=30
  export GENERATE_POSTPROCESS=none
  export ALIGN_EPOCHS=60
  export ALIGN_LR=3e-4
  export ALIGN_EMBEDDING_GUIDANCE=5.0
  export SKIP_TRAIN=0
  export SKIP_ALIGN=0
  run_inner all
}

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

run_one() {
  local v="$1"
  echo
  echo "============================================================"
  echo "  Variant: $v"
  echo "  Started: $(date -Iseconds)"
  echo "============================================================"
  (
    case "$v" in
      base)              run_variant_base ;;
      pp_k21_g1)         run_variant_pp_k21_g1 ;;
      pp_none)           run_variant_pp_none ;;
      gen_g30)           run_variant_gen_g30 ;;
      gen_steps5)        run_variant_gen_steps5 ;;
      gen_steps50_g15)   run_variant_gen_steps50_g15 ;;
      align_e40)         run_variant_align_e40 ;;
      align_lr1e4)       run_variant_align_lr1e4 ;;
      emb_g30)           run_variant_emb_g30 ;;
      *) echo "ERROR: unknown variant '$v'" >&2; exit 2 ;;
    esac
  )
  echo "  Finished: $(date -Iseconds)"
}

main() {
  # Convert SEEDS (string "0 1 2 3 4") into an array.
  read -r -a seed_array <<< "$SEEDS"
  echo "Seeds to run: ${seed_array[*]}"
  echo "Variants to run: ${VARIANTS[*]}"

  # Decide whether base needs to be auto-prepended. base is the checkpoint
  # that every generate-only / align-only variant reuses; without it for a
  # given seed, those variants will fail with "base run not found".
  #
  # Rule: if 'base' is NOT explicitly in VARIANTS but any variant in
  # VARIANTS needs base (anything other than pp_k21_g1, which retrains from
  # scratch), we auto-run base first for each seed that doesn't have one.
  local need_base=0
  local base_in_variants=0
  for v in "${VARIANTS[@]}"; do
    if [[ "$v" == "base" ]]; then base_in_variants=1; fi
    if [[ "$v" != "base" && "$v" != "pp_k21_g1" ]]; then need_base=1; fi
  done

  for seed in "${seed_array[@]}"; do
    # Re-export SEED and recompute the BASE_SUBJECT_RUN path so all variants
    # know which seed they're operating on. Each seed gets its own independent
    # base run; cross-seed checkpoint reuse is not supported.
    export SEED="$seed"
    BASE_SUBJECT_RUN="$RUNS_ROOT/$BASE_TAG/$EXP_NAME/${SUBJECT}_seed${SEED}"

    # Auto-prepend base for this seed if needed and not already there.
    if [[ "$need_base" == "1" && "$base_in_variants" == "0" ]]; then
      if [[ ! -f "$BASE_SUBJECT_RUN/generated_datasets/generated_embeddings.pt" ]]; then
        echo
        echo "############################################################"
        echo "  seed=$seed  auto-running 'base' (required by variants,"
        echo "             missing at $BASE_SUBJECT_RUN)"
        echo "############################################################"
        run_one "base"
      else
        echo "[seed=$seed] base already exists at $BASE_SUBJECT_RUN, skipping."
      fi
    fi

    for v in "${VARIANTS[@]}"; do
      echo
      echo "############################################################"
      echo "  seed=$seed  variant=$v"
      echo "############################################################"
      run_one "$v"
    done
  done

  echo
  echo "============================================================"
  echo "  All runs finished. Aggregating results..."
  echo "============================================================"
  # For aggregation, include 'base' in the variant list only if any base run
  # actually exists (it does if we auto-prepended above).
  local agg_variants=("${VARIANTS[@]}")
  if [[ "$need_base" == "1" && "$base_in_variants" == "0" ]]; then
    agg_variants=("base" "${VARIANTS[@]}")
  fi

  for seed in "${seed_array[@]}"; do
    python "$(dirname "${BASH_SOURCE[0]}")/aggregate_grid_results.py" \
      --runs-root "$RUNS_ROOT" \
      --variants "${agg_variants[@]}" \
      --subject "$SUBJECT" \
      --seed "$seed" \
      --exp-name "$EXP_NAME" \
      --output "$RUNS_ROOT/grid_summary_seed${seed}.json" \
      --markdown "$RUNS_ROOT/grid_summary_seed${seed}.md"
  done

  if [[ "${#seed_array[@]}" -gt 1 ]]; then
    echo
    echo "  Multi-seed mode: producing mean ± std summary across seeds..."
    python "$(dirname "${BASH_SOURCE[0]}")/aggregate_multi_seed.py" \
      --runs-root "$RUNS_ROOT" \
      --variants "${agg_variants[@]}" \
      --subject "$SUBJECT" \
      --seeds "${seed_array[@]}" \
      --exp-name "$EXP_NAME" \
      --output "$RUNS_ROOT/grid_summary_multiseed.json" \
      --markdown "$RUNS_ROOT/grid_summary_multiseed.md"
    echo "Multi-seed summary written to:"
    echo "  $RUNS_ROOT/grid_summary_multiseed.json"
    echo "  $RUNS_ROOT/grid_summary_multiseed.md"
  fi

  echo
  echo "Per-seed summaries written to:"
  for seed in "${seed_array[@]}"; do
    echo "  $RUNS_ROOT/grid_summary_seed${seed}.md"
  done
}

main "$@"