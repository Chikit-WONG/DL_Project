#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM/logs/generate_recon_%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM/logs/generate_recon_%j.err
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM
#SBATCH --time=24:00:00

# Usage: edit CHECKPOINT and RUN_NAME before submitting
# CHECKPOINT: path to the trained ATMS reconstruction checkpoint
# RUN_NAME:   identifier for the output directory

set -eo pipefail
mkdir -p logs

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
module load cuda/12.6

CHECKPOINT="${1:-./models/contrast/ATMS/sub-01/LATEST_RECONSTRUCTION/40.pth}"
RUN_NAME="${2:-run01}"
SEEDS_CSV="${3:-0,1,2,3,4,5,6,7,8,9}"
BASE_OUTPUT_DIR="./outputs/reconstructions/${RUN_NAME}"

echo "Checkpoint: ${CHECKPOINT}"
echo "Output root: ${BASE_OUTPUT_DIR}"
echo "Generation seeds: ${SEEDS_CSV}"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
for seed in "${SEEDS[@]}"; do
  seed="$(echo "${seed}" | xargs)"
  seed_dir="${BASE_OUTPUT_DIR}/seed$(printf '%02d' "${seed}")"
  echo "Generating seed ${seed} -> ${seed_dir}"
  python -u Generation/generate_reconstructions.py \
    --checkpoint   "${CHECKPOINT}" \
    --output_dir   "${seed_dir}" \
    --data_path    /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data \
    --subject      sub-01 \
    --sd_model_path   /hpc2hdd/home/ckwong627/workdir/models/stable-diffusion-v1-5 \
    --ip_adapter_dir  /hpc2hdd/home/ckwong627/workdir/models/IP-Adapter/models \
    --ip_adapter_weight ip-adapter_sd15.bin \
    --device       cuda:0 \
    --encode_batch_size 64 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --seed "${seed}"
done
