#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH -o /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/logs/evaluate_%j.out
#SBATCH -e /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/logs/evaluate_%j.err
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2
#SBATCH --time=02:00:00

set -eo pipefail
mkdir -p logs checkpoints results cache

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
module load cuda/12.6

python -u codes/evaluate.py \
    --tag v2_final \
    --encoder_ckpt checkpoints/v2_final_best.pt \
    --compare_v1

python -u codes/make_task2_montage.py --tag v2_final --seed_index 0 --num_samples 20
python -u codes/summarize_results.py --tag v2_final
