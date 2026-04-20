#!/bin/bash
#SBATCH -p long_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --time=02:00:00
#SBATCH -J cogcap_simple_align
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/slurm_scripts/logs/%x-%j.err

# SimpleAlignPipe: direct MLP alignment (EEG CLIP → diffusion embedding space).
# Fixes applied to diffusion_pipe.py:
#   - Removed L2 normalization from SimpleAlignMLP.forward()
#   - SDEmbeddingLoss now normalizes both pred and target before MSE
#   - OneCycleLR max_lr lowered to lr*3, modality masking re-enabled
# DiffusionPipe checkpoints renamed to not contain 'best', so no checkpoint
# is found and training runs fresh. Output → generated_embeddings.pt (same as
# what batch_generate.py expects).

REPO=/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP
EXP_DIR=$REPO/runs/full_v2/intra-subject_cogcappro_EEGProjectLayer_multimodal_cogcap_list_ViT-H-14/sub-01_seed0
mkdir -p "$REPO/slurm_scripts/logs"
cd "$REPO"

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
set -euo pipefail
module load cuda/12.6

echo "Job started at $(date)"
python -m src.cogcappro.align.main \
  --exp_dir "$EXP_DIR" \
  --device 0 \
  --epoch 100 \
  --lr 3e-4 \
  --model_type simple
echo "Job ended at $(date)"
