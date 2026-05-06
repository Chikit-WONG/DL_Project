#!/bin/bash
# Download all required model weights to DL_Project/models/.
# Submit from the repository root:
#   sbatch task1/slurm_scripts/00b_download_models.sh
#
# Optional environment variables (set before sbatch, e.g. HF_TOKEN=xxx sbatch ...):
#   HF_TOKEN    HuggingFace API token (only needed for gated models)
#   MODEL_DEST  Custom download path (default: DL_Project/models/)
#
# Can run in parallel with 00_setup_env.sh — no GPU required.
#
#SBATCH -p i64m512u
#SBATCH -o task1/slurm_scripts/logs/00b_download_models_%j.out
#SBATCH -e task1/slurm_scripts/logs/00b_download_models_%j.err
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p "$REPO_ROOT/task1/slurm_scripts/logs"

set -eo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
type unclash >/dev/null 2>&1 && unclash || true

for _conda in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    [ -f "$_conda/etc/profile.d/conda.sh" ] && { source "$_conda/etc/profile.d/conda.sh"; break; }
done

# Prefer DL_Project env; fall back to base with a quick huggingface_hub install
if conda env list 2>/dev/null | grep -q "^DL_Project "; then
    conda activate DL_Project
else
    conda activate base 2>/dev/null || true
    python -c "import huggingface_hub" 2>/dev/null \
        || pip install -q "huggingface_hub>=0.20" tqdm
fi

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"

EXTRA_ARGS=""
[ -n "$HF_TOKEN" ]   && EXTRA_ARGS="$EXTRA_ARGS --hf-token $HF_TOKEN"
[ -n "$MODEL_DEST" ] && EXTRA_ARGS="$EXTRA_ARGS --dest $MODEL_DEST"

python "$REPO_ROOT/scripts/download_models.py" $EXTRA_ARGS

echo "Job ended: $(date)"
