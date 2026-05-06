#!/bin/bash
# Creates the DL_Project conda environment and installs all dependencies.
# Submit from the repository root:
#   sbatch task1/slurm_scripts/00_setup_env.sh
#
#SBATCH -p i64m512u
#SBATCH -o task1/slurm_scripts/logs/00_setup_env_%j.out
#SBATCH -e task1/slurm_scripts/logs/00_setup_env_%j.err
#SBATCH -n 8
#SBATCH --time=02:00:00
#SBATCH -J dl_setup_env

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p "$REPO_ROOT/task1/slurm_scripts/logs"

set -eo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
type unclash >/dev/null 2>&1 && unclash || true

for _conda in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    [ -f "$_conda/etc/profile.d/conda.sh" ] && { source "$_conda/etc/profile.d/conda.sh"; break; }
done

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Repo root: $REPO_ROOT"

# Create env (skip if already exists)
if conda env list | grep -q "^DL_Project "; then
    echo "Conda env DL_Project already exists, skipping creation."
else
    echo "Creating conda env DL_Project (Python 3.10)..."
    conda create -n DL_Project python=3.10 -y
fi

conda activate DL_Project

echo "Python: $(which python) — $(python --version)"

# Install PyTorch 2.5.0 with CUDA 12.4 wheels (compatible with cluster CUDA 12.6)
echo "Installing PyTorch..."
pip install torch==2.5.0 torchvision==0.20.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install shared requirements (covers Task 1 and Task 2)
echo "Installing shared requirements..."
pip install -r "$REPO_ROOT/requirements.txt"

# Install Task 2 pinned requirements (superset that also satisfies Task 1)
echo "Installing Task 2 requirements..."
pip install -r "$REPO_ROOT/task2/requirements.txt"

echo "Verifying installation..."
python -c "import torch; print('torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
python -c "import open_clip; print('open_clip OK')"
python -c "import diffusers; print('diffusers:', diffusers.__version__)"

echo "Job ended: $(date)"
echo "DONE — conda env DL_Project is ready."
