#!/bin/bash
# Submit from the repository root:
#   sbatch task2/slurm_scripts/02b_reprepare_diffusion_embeddings.sh
#
# Reads sdxl_root and ip_adapter_root from task2/configs/local.yaml.
#
#SBATCH -p i64m1tga40u
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --time=04:00:00
#SBATCH -J cogcap_embed_v2
#SBATCH -o task2/slurm_scripts/logs/%x-%j.out
#SBATCH -e task2/slurm_scripts/logs/%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK2_DIR="$(dirname "$SCRIPT_DIR")"
mkdir -p "$SCRIPT_DIR/logs"
cd "$TASK2_DIR"

set -euo pipefail
for _conda in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    [ -f "$_conda/etc/profile.d/conda.sh" ] && { source "$_conda/etc/profile.d/conda.sh"; break; }
done
conda activate DL_Project
module load cuda/12.6

# Read paths from local.yaml (data_root auto-detected if not set)
SDXL_PATH=$(python -c "import yaml; c=yaml.safe_load(open('configs/local.yaml')); print(c['paths']['sdxl_root'])")
IP_ADAPTER_PATH=$(python -c "import yaml; c=yaml.safe_load(open('configs/local.yaml')); print(c['paths']['ip_adapter_root'])")
DATA_ROOT=$(python scripts/get_data_root.py)
OUTPUT_DIR="$DATA_ROOT/ThingsEEG/Image_feature_new/data_features"

echo "Job started at $(date)"
python scripts/prepare_diffusion_embeddings.py \
  --config configs/cogcappro.yaml \
  --data-type EEG \
  --sd-path "$SDXL_PATH" \
  --ip-adapter-path "$IP_ADAPTER_PATH" \
  --device cuda:0 \
  --output-dir "$OUTPUT_DIR"
echo "Job ended at $(date)"
