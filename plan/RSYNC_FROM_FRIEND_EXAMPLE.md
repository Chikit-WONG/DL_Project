# Rsync Guide for Full-Training-Set Rerun Outputs

## One-line submission on your friend's HPC

From the `DL_Project/` root:

```bash
bash run_full_training_set_rerun.sh
```

To monitor submitted jobs:

```bash
bash monitor_full_training_set_rerun.sh
```

## Key output paths to copy back

### `version1`

- `version1/checkpoints/phase1_main_best.pt`
- `version1/checkpoints/phase2_main_best.pt`
- `version1/outputs/metrics_phase2_main_best.json`
- `version1/outputs/recon_images_phase2_main.pt`
- `version1/outputs/recon_meta_phase2_main.json`
- `version1/outputs/`
- `version1/logs/`

### `version2`

- `version2/checkpoints/`
- `version2/results/metrics_v2_final.json`
- `version2/results/recon_images_v2_final.pt`
- `version2/results/recon_meta_v2_final.json`
- `version2/results/task2_montage_v2_final_s00.png`
- `version2/results/results_summary_en.md`
- `version2/results/results_summary_zh.md`
- `version2/logs/`

### `version3_ATM`

- `version3_ATM/models/contrast/ATMS/sub-01/`
- `version3_ATM/outputs/retrieval_eval_run01.csv`
- `version3_ATM/outputs/reconstruction_eval_run01.csv`
- `version3_ATM/outputs/reconstructions/run01/`
- `version3_ATM/logs/`

### `version4_CCP`

- `version4_CCP/runs/full_v2/`
- `version4_CCP/runs/summary_metrics_v2.json`
- `version4_CCP/slurm_scripts/logs/`

## Example `rsync` commands

Replace `YOUR_FRIEND_HOST` with your friend's SSH host.

```bash
rsync -avP YOUR_FRIEND_HOST:/path/to/DL_Project/version1/checkpoints/ /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version1/checkpoints/
```

```bash
rsync -avP YOUR_FRIEND_HOST:/path/to/DL_Project/version1/outputs/ /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version1/outputs/
```

```bash
rsync -avP YOUR_FRIEND_HOST:/path/to/DL_Project/version2/results/ /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/results/
```

```bash
rsync -avP YOUR_FRIEND_HOST:/path/to/DL_Project/version2/checkpoints/ /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/checkpoints/
```

```bash
rsync -avP YOUR_FRIEND_HOST:/path/to/DL_Project/version3_ATM/models/contrast/ATMS/sub-01/ /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM/models/contrast/ATMS/sub-01/
```

```bash
rsync -avP YOUR_FRIEND_HOST:/path/to/DL_Project/version3_ATM/outputs/ /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version3_ATM/outputs/
```

```bash
rsync -avP YOUR_FRIEND_HOST:/path/to/DL_Project/version4_CCP/runs/full_v2/ /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/runs/full_v2/
```

```bash
rsync -avP YOUR_FRIEND_HOST:/path/to/DL_Project/version4_CCP/runs/summary_metrics_v2.json /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version4_CCP/runs/
```

## Recommended copy order

1. Copy `logs/` first if you need to inspect failures remotely.
2. Copy small metric files and summaries next.
3. Copy checkpoints and reconstruction tensors last because they are large.
