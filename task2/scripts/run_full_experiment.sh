#!/usr/bin/env bash
set -euo pipefail

# Centralized experiment runner for CognitionCapturerPro.
# Edit the variables in this block before running a new experiment.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

#######################################
# Environment
#######################################
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cogcap}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6}"

#######################################
# Paths
#######################################
BASE_CONFIG="${BASE_CONFIG:-configs/cogcappro.yaml}"
DATA_ROOT="${DATA_ROOT:-/hpc2hdd/home/dsaa2012_042/project/image-eeg-data/converted_for_cogcappro}"
COURSE_DATA_ROOT="${COURSE_DATA_ROOT:-$(dirname "$DATA_ROOT")}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/hpc2hdd/home/dsaa2012_042/project/models}"
RUNS_ROOT="${RUNS_ROOT:-$REPO/runs}"
DIFFUSION_EMBEDDINGS_DIR="${DIFFUSION_EMBEDDINGS_DIR:-$WEIGHTS_ROOT/diffusion_embeddings}"
REAL_ROOT="${REAL_ROOT:-$DATA_ROOT/ThingsEEG/Image_set_Resize/test_images}"

#######################################
# Experiment identity
#######################################
RUN_TAG="${RUN_TAG:-full_tune}"
EXP_SETTING="${EXP_SETTING:-intra-subject}"
SUBJECT="${SUBJECT:-sub-01}"
SEED="${SEED:-0}"
DATA_TYPE="${DATA_TYPE:-EEG}"
VISION_BACKBONE="${VISION_BACKBONE:-ViT-H-14}"
BRAIN_BACKBONE="${BRAIN_BACKBONE:-EEGProjectLayer_multimodal_cogcap_list}"
EXP_NAME="${EXP_NAME:-${EXP_SETTING}_cogcappro_${BRAIN_BACKBONE}_${VISION_BACKBONE}}"
EXP_ROOT="${EXP_ROOT:-$RUNS_ROOT/$RUN_TAG}"
SUBJECT_RUN="${SUBJECT_RUN:-$EXP_ROOT/$EXP_NAME/${SUBJECT}_seed${SEED}}"
GENERATED_CONFIG="${GENERATED_CONFIG:-$REPO/configs/generated_${RUN_TAG}.yaml}"

#######################################
# Retrieval training hyperparameters
#######################################
TRAIN_DEVICES="${TRAIN_DEVICES:-0}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-80}"
TRAIN_LR="${TRAIN_LR:-1e-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-200}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-200}"
LOSS_TYPE="${LOSS_TYPE:-ClipLoss_Modified_DDP}"
UNCERTAINTY_AWARE="${UNCERTAINTY_AWARE:-1}"
MASK_COUNT="${MASK_COUNT:-1}"
STAGED_TRAINING="${STAGED_TRAINING:-0}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-40}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-20}"
TEXT_MAX_EPOCHS="${TEXT_MAX_EPOCHS:-30}"
SELECTED_REGION="${SELECTED_REGION:-}"
FILTER_BAND="${FILTER_BAND:-}"

#######################################
# Image feature / blur hyperparameters
# Lowered from (51, 3) -> (31, 2): the old 51-kernel low-pass was bleeding
# mid-frequency structure and pushing SwAV/Inception in the wrong direction
# without buying real SSIM gains. 31/2 is a more conservative anchor.
#######################################
BLUR_KERNEL_SIZE="${BLUR_KERNEL_SIZE:-31}"
SYSTEM_G="${SYSTEM_G:-2}"

#######################################
# Alignment hyperparameters
# ALIGN_EPOCHS lowered 100 -> 60 to curb the regression-to-the-mean
# behavior that boosts SSIM but hurts SwAV / EffNet.
#######################################
ALIGN_DEVICE="${ALIGN_DEVICE:-0}"
ALIGN_MODEL_TYPE="${ALIGN_MODEL_TYPE:-simple}"
ALIGN_EPOCHS="${ALIGN_EPOCHS:-60}"
ALIGN_LR="${ALIGN_LR:-3e-4}"
ALIGN_EMBEDDING_STEPS="${ALIGN_EMBEDDING_STEPS:-50}"
ALIGN_EMBEDDING_GUIDANCE="${ALIGN_EMBEDDING_GUIDANCE:-5.0}"

#######################################
# SDXL / IP-Adapter generation hyperparameters
# GENERATE_GUIDANCE lifted 0.0 -> 1.5: turbo at CFG=0 produces flat textures
# that hurt SwAV/Inception. 1.5 is the empirical sweet spot where CLIP stays
# flat or rises and SwAV improves visibly.
#######################################
SDXL_PATH="${SDXL_PATH:-stabilityai/sdxl-turbo}"
IP_ADAPTER_PATH="${IP_ADAPTER_PATH:-h94/IP-Adapter}"
GENERATE_DEVICE="${GENERATE_DEVICE:-cuda:0}"
MODALITY_MODE="${MODALITY_MODE:-all}"
GENERATE_STEPS="${GENERATE_STEPS:-30}"
GENERATE_GUIDANCE="${GENERATE_GUIDANCE:-1.5}"
GENERATE_POSTPROCESS="${GENERATE_POSTPROCESS:-none}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality}"
RESUME_GENERATION="${RESUME_GENERATION:-0}"
GENERATE_SUBJECT_FILTER="${GENERATE_SUBJECT_FILTER:-${SUBJECT}_seed${SEED}}"

#######################################
# Evaluation / pipeline switches
#######################################
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
EVAL_IMAGE_SIZE="${EVAL_IMAGE_SIZE:-256}"
PREPARE_DIFFUSION_EMBEDDINGS="${PREPARE_DIFFUSION_EMBEDDINGS:-auto}"
PREPARE_COURSE_DATA="${PREPARE_COURSE_DATA:-auto}"
CLEAN_GENERATED_IMAGES="${CLEAN_GENERATED_IMAGES:-0}"

# Stage-skip switches used by the grid wrapper to avoid re-running the
# expensive retrieval / alignment stages when only generation hyperparameters
# changed. Set SKIP_TRAIN=1 / SKIP_ALIGN=1 to bypass.
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_ALIGN="${SKIP_ALIGN:-0}"

usage() {
  cat <<EOF
Usage: $0 [all|prepare|train|align|generate|eval|summary|dry-run]

Edit hyperparameters at the top of this script, or override them with env vars:
  RUN_TAG=lr3e-4 TRAIN_LR=3e-4 ALIGN_LR=1e-4 $0 all
EOF
}

run_cmd() {
  echo "+ $*"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "$@"
  fi
}

prepare_course_data() {
  if [[ "$DATA_TYPE" != "EEG" ]]; then
    return
  fi

  local things_root="$DATA_ROOT/ThingsEEG"
  local depth_dir="$things_root/Image_depth_set_Resize"
  local edge_dir="$things_root/Image_edge_set_Resize"
  local should_prepare=0

  if [[ "$PREPARE_COURSE_DATA" == "1" ]]; then
    should_prepare=1
  elif [[ "$PREPARE_COURSE_DATA" == "auto" && ( ! -d "$depth_dir" || ! -d "$edge_dir" ) ]]; then
    should_prepare=1
  fi

  if [[ "$should_prepare" == "1" ]]; then
    run_cmd python scripts/prepare_course_data.py \
      --course-data-root "$COURSE_DATA_ROOT" \
      --output-root "$DATA_ROOT" \
      --subject "$SUBJECT"
  else
    echo "Skip course data preparation: $depth_dir and $edge_dir already exist."
  fi
}

setup_env() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "+ setup environment: CONDA_ENV=$CONDA_ENV CUDA_MODULE=$CUDA_MODULE"
    return
  fi

  if [[ -f "$CONDA_SH" ]]; then
    # shellcheck source=/dev/null
    source "$CONDA_SH"
    if [[ -n "$CONDA_ENV" ]]; then
      conda activate "$CONDA_ENV"
    fi
  else
    echo "WARN: conda activation script not found: $CONDA_SH"
  fi

  if [[ -n "$CUDA_MODULE" ]] && command -v module >/dev/null 2>&1; then
    module load "$CUDA_MODULE"
  fi
}

write_generated_config() {
  mkdir -p "$(dirname "$GENERATED_CONFIG")" "$SUBJECT_RUN"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "+ write generated config to $GENERATED_CONFIG"
    return
  fi

  CFG_BASE="$BASE_CONFIG" \
  CFG_OUT="$GENERATED_CONFIG" \
  CFG_BLUR_KERNEL_SIZE="$BLUR_KERNEL_SIZE" \
  CFG_SYSTEM_G="$SYSTEM_G" \
  CFG_TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
  CFG_VAL_BATCH_SIZE="$VAL_BATCH_SIZE" \
  CFG_TEST_BATCH_SIZE="$TEST_BATCH_SIZE" \
  CFG_TEXT_MAX_EPOCHS="$TEXT_MAX_EPOCHS" \
  CFG_STAGE1_EPOCHS="$STAGE1_EPOCHS" \
  CFG_STAGE2_EPOCHS="$STAGE2_EPOCHS" \
  CFG_STAGE3_EPOCHS="$STAGE3_EPOCHS" \
  CFG_DATA_ROOT="$DATA_ROOT" \
  CFG_WEIGHTS_ROOT="$WEIGHTS_ROOT" \
  CFG_RUNS_ROOT="$RUNS_ROOT" \
  CFG_DIFFUSION_EMBEDDINGS_DIR="$DIFFUSION_EMBEDDINGS_DIR" \
  python - <<'PY'
import os
from pathlib import Path
from omegaconf import OmegaConf

cfg = OmegaConf.load(os.environ["CFG_BASE"])
cfg.blur_kernel_size = int(os.environ["CFG_BLUR_KERNEL_SIZE"])
cfg.system_g = int(os.environ["CFG_SYSTEM_G"])
cfg.data.train_batch_size = int(os.environ["CFG_TRAIN_BATCH_SIZE"])
cfg.data.val_batch_size = int(os.environ["CFG_VAL_BATCH_SIZE"])
cfg.data.test_batch_size = int(os.environ["CFG_TEST_BATCH_SIZE"])
cfg.train.text_max_epochs = int(os.environ["CFG_TEXT_MAX_EPOCHS"])
cfg.train.stage1_epochs = int(os.environ["CFG_STAGE1_EPOCHS"])
cfg.train.stage2_epochs = int(os.environ["CFG_STAGE2_EPOCHS"])
cfg.train.stage3_epochs = int(os.environ["CFG_STAGE3_EPOCHS"])
cfg.paths.data_root = os.environ["CFG_DATA_ROOT"]
cfg.paths.weights_root = os.environ["CFG_WEIGHTS_ROOT"]
cfg.paths.runs_root = os.environ["CFG_RUNS_ROOT"]
cfg.paths.diffusion_embeddings_root = os.environ["CFG_DIFFUSION_EMBEDDINGS_DIR"]

out = Path(os.environ["CFG_OUT"])
out.parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(cfg, out)
print(f"Wrote generated config: {out}")
PY
}

maybe_prepare_embeddings() {
  local train_file="$DIFFUSION_EMBEDDINGS_DIR/diffusion_clip_embeddings_train.pt"
  local test_file="$DIFFUSION_EMBEDDINGS_DIR/diffusion_clip_embeddings_test.pt"
  local should_prepare=0

  if [[ "$PREPARE_DIFFUSION_EMBEDDINGS" == "1" ]]; then
    should_prepare=1
  elif [[ "$PREPARE_DIFFUSION_EMBEDDINGS" == "auto" && ( ! -f "$train_file" || ! -f "$test_file" ) ]]; then
    should_prepare=1
  fi

  if [[ "$should_prepare" == "1" ]]; then
    mkdir -p "$DIFFUSION_EMBEDDINGS_DIR"
    run_cmd python scripts/prepare_diffusion_embeddings.py \
      --config "$GENERATED_CONFIG" \
      --data-type "$DATA_TYPE" \
      --sd-path "$SDXL_PATH" \
      --ip-adapter-path "$IP_ADAPTER_PATH" \
      --device "$GENERATE_DEVICE" \
      --output-dir "$DIFFUSION_EMBEDDINGS_DIR"
  else
    echo "Skip diffusion embedding preparation: $train_file and $test_file already exist."
  fi
}

train_retrieval() {
  if [[ "$SKIP_TRAIN" == "1" ]]; then
    echo "Skip retrieval training (SKIP_TRAIN=1)."
    return
  fi
  local cmd=(
    python main.py
    --config "$GENERATED_CONFIG"
    --subjects "$SUBJECT"
    --devices "$TRAIN_DEVICES"
    --vision_backbone "$VISION_BACKBONE"
    --data_type "$DATA_TYPE"
    --seed "$SEED"
    --exp_setting "$EXP_SETTING"
    --brain_backbone "$BRAIN_BACKBONE"
    --epoch "$TRAIN_EPOCHS"
    --lr "$TRAIN_LR"
    --loss_type "$LOSS_TYPE"
    --mask_count "$MASK_COUNT"
    --save_dir "$SUBJECT_RUN"
  )
  [[ "$UNCERTAINTY_AWARE" == "1" ]] && cmd+=(--uncertainty_aware)
  [[ "$STAGED_TRAINING" == "1" ]] && cmd+=(--staged_training)
  [[ -n "$SELECTED_REGION" ]] && cmd+=(--selected_region "$SELECTED_REGION")
  [[ -n "$FILTER_BAND" ]] && cmd+=(--filter_band "$FILTER_BAND")
  run_cmd "${cmd[@]}"
}

align_embeddings() {
  if [[ "$SKIP_ALIGN" == "1" ]]; then
    echo "Skip alignment (SKIP_ALIGN=1)."
    return
  fi
  # Force-clean stale align outputs so this run starts from scratch. The
  # grid wrapper sets FORCE_ALIGN_RERUN=1 for align-changing variants.
  # Without this, the underlying align main may load an existing checkpoint
  # and silently produce identical metrics to a previous run.
  if [[ "${FORCE_ALIGN_RERUN:-0}" == "1" && -d "$SUBJECT_RUN/align" ]]; then
    echo "FORCE_ALIGN_RERUN=1: removing $SUBJECT_RUN/align"
    run_cmd rm -rf "$SUBJECT_RUN/align"
  fi
  run_cmd python -m src.cogcappro.align.main \
    --exp_dir "$SUBJECT_RUN" \
    --device "$ALIGN_DEVICE" \
    --epoch "$ALIGN_EPOCHS" \
    --lr "$ALIGN_LR" \
    --model_type "$ALIGN_MODEL_TYPE" \
    --embedding_inference_steps "$ALIGN_EMBEDDING_STEPS" \
    --embedding_guidance_scale "$ALIGN_EMBEDDING_GUIDANCE"
}

generate_images() {
  if [[ "$CLEAN_GENERATED_IMAGES" == "1" ]]; then
    run_cmd rm -f "$SUBJECT_RUN/generated_image/${MODALITY_MODE}_before/"*.jpg
    run_cmd rm -f "$SUBJECT_RUN/generated_image/${MODALITY_MODE}/"*.jpg
  fi

  local resume_arg=()
  [[ "$RESUME_GENERATION" == "1" ]] && resume_arg=(--resume)

  run_cmd python -u -m src.cogcappro.generate_image.batch_generate \
    --base_dir "$EXP_ROOT" \
    --config "$GENERATED_CONFIG" \
    --data_type "$DATA_TYPE" \
    --sd_path "$SDXL_PATH" \
    --ip_adapter_path "$IP_ADAPTER_PATH" \
    --modality_mode "$MODALITY_MODE" \
    --device "$GENERATE_DEVICE" \
    --subjects "$GENERATE_SUBJECT_FILTER" \
    --num_inference_steps "$GENERATE_STEPS" \
    --guidance_scale "$GENERATE_GUIDANCE" \
    --postprocess "$GENERATE_POSTPROCESS" \
    --negative_prompt "$NEGATIVE_PROMPT" \
    --use_before_align \
    "${resume_arg[@]}"

  run_cmd python -u -m src.cogcappro.generate_image.batch_generate \
    --base_dir "$EXP_ROOT" \
    --config "$GENERATED_CONFIG" \
    --data_type "$DATA_TYPE" \
    --sd_path "$SDXL_PATH" \
    --ip_adapter_path "$IP_ADAPTER_PATH" \
    --modality_mode "$MODALITY_MODE" \
    --device "$GENERATE_DEVICE" \
    --subjects "$GENERATE_SUBJECT_FILTER" \
    --num_inference_steps "$GENERATE_STEPS" \
    --guidance_scale "$GENERATE_GUIDANCE" \
    --postprocess "$GENERATE_POSTPROCESS" \
    --negative_prompt "$NEGATIVE_PROMPT" \
    "${resume_arg[@]}"
}

evaluate_reconstruction() {
  run_cmd python scripts/evaluate_reconstruction.py \
    --real-root "$REAL_ROOT" \
    --fake-root "$SUBJECT_RUN/generated_image/${MODALITY_MODE}_before" \
    --output "$SUBJECT_RUN/generated_image/${MODALITY_MODE}_before/reconstruction_metrics.json" \
    --device "$EVAL_DEVICE" \
    --image-size "$EVAL_IMAGE_SIZE"

  run_cmd python scripts/evaluate_reconstruction.py \
    --real-root "$REAL_ROOT" \
    --fake-root "$SUBJECT_RUN/generated_image/${MODALITY_MODE}" \
    --output "$SUBJECT_RUN/generated_image/${MODALITY_MODE}/reconstruction_metrics.json" \
    --device "$EVAL_DEVICE" \
    --image-size "$EVAL_IMAGE_SIZE"
}

summarize_results() {
  run_cmd python scripts/summarize_results.py \
    --runs-root "$EXP_ROOT" \
    --output "$RUNS_ROOT/summary_${RUN_TAG}.json"
}

main() {
  local stage="${1:-all}"
  DRY_RUN=0
  if [[ "$stage" == "dry-run" ]]; then
    DRY_RUN=1
    stage="all"
  fi

  case "$stage" in
    all|prepare|train|align|generate|eval|summary) ;;
    -h|--help|help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac

  setup_env
  write_generated_config

  case "$stage" in
    all)
      prepare_course_data
      maybe_prepare_embeddings
      train_retrieval
      align_embeddings
      generate_images
      evaluate_reconstruction
      summarize_results
      ;;
    prepare) prepare_course_data; maybe_prepare_embeddings ;;
    train) prepare_course_data; train_retrieval ;;
    align) align_embeddings ;;
    generate) generate_images ;;
    eval) evaluate_reconstruction ;;
    summary) summarize_results ;;
  esac

  echo "Done. Subject run: $SUBJECT_RUN"
}

main "$@"
