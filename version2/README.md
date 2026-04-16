# THINGS-EEG Retrieval and Reconstruction, Version 2

> [中文文档](README-CN.md)

DSAA2012 Deep Learning Project

`version2` is an upgraded EEG-to-image pipeline built on top of the `version1` baseline, but implemented as an independent workspace. However, the result was not satisfactory. Therefore, I decided to take a different approach and launch a new version. It keeps the same shared THINGS-EEG data source and extends the pipeline with:

- a stronger EEG encoder with dual-path temporal modeling
- electrode position encoding and region-aware channel gating
- multi-target visual supervision with `CLIP ViT-H/14`, `ViT-B/32`, `RN50`, and SD VAE latents
- a lightweight prior network for semantic refinement
- image reconstruction with `SDXL-Turbo + IP-Adapter + img2img`

The project target is to improve both retrieval and reconstruction quality without modifying the `version1` codebase.

## Project Layout

```text
version2/
├── cache/                  # Cached image backbone features
├── checkpoints/            # Saved encoder / prior checkpoints
├── codes/                  # Main source code
├── logs/                   # SLURM stdout / stderr and training logs
├── plan/                   # Planning docs and references
├── results/                # Metrics, reconstructions, montage, summaries
├── slurm_scripts/          # HPC job scripts
├── README.md
└── README-CN.md
```

## Method Overview

### 1. Shared-data setup

`version2` does not copy the dataset. It reads the shared directory configured in [codes/config.py](/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/codes/config.py):

- EEG and image data:
  `/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data`
- train images:
  `training_images/`
- test images:
  `test_images/`
- electrode positions:
  `EEG_CHANNELS.jsonl`

The loader reads `train.pt` and `test.pt`, performs `80-trial averaging` by default, and keeps train/test ordering aligned with cached visual features.

### 2. EEG encoder

The encoder in [codes/model.py](/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/codes/model.py) consists of:

- electrode position encoding from channel coordinates
- region-aware gating with higher initial weight on posterior channels
- dual-path temporal convolution
- transformer token mixing
- subject embedding adapter
- dual heads:
  - semantic head: predicts image-semantic embeddings
  - structural head: predicts SD VAE latents

### 3. Multi-target supervision

Before encoder training, [codes/cache_backbone_features.py](/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/codes/cache_backbone_features.py) caches:

- `CLIP ViT-H/14` image embeddings
- `CLIP ViT-B/32` image embeddings
- `RN50` image embeddings
- `Stable Diffusion v1.5 VAE` latents

The encoder is trained in three stages:

1. `warmup`
   `ViT-H/14 InfoNCE + 0.5 * MSE`
2. `multitarget`
   add `ViT-B/32`, `RN50`, and `VAE latent` supervision
3. `finetune`
   add hard-negative and supervised contrastive terms

### 4. Prior and reconstruction

[codes/train_prior.py](/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/codes/train_prior.py) trains a lightweight denoising prior over semantic embeddings.  
[codes/reconstruct.py](/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/codes/reconstruct.py) then runs:

- encoder semantic head
- optional prior sampling
- structural latent decoding into a blurry image
- `SDXL-Turbo` img2img refinement
- `IP-Adapter SDXL` image-conditioning from EEG-predicted semantics

## Environment

The implementation was developed and run in the `test` conda environment on the HPC cluster.

### Core runtime requirements

- Python 3.10
- PyTorch with CUDA support
- `transformers`
- `diffusers`
- `accelerate`
- `torchvision`
- `scikit-image`
- `numpy`
- `Pillow`
- optional `clip` package for `RN50` caching fallback

The job scripts assume:

```bash
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
module load cuda/12.6
```

### Required model directories

Configured in [codes/config.py](/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/codes/config.py):

- `/hpc2hdd/home/ckwong627/workdir/models/CLIP-ViT-H-14-laion2B-s32B-b79K`
- `/hpc2hdd/home/ckwong627/workdir/models/CLIP-ViT-B-32-laion2B-s34B-b79K`
- `/hpc2hdd/home/ckwong627/workdir/models/stable-diffusion-v1-5`
- `/hpc2hdd/home/ckwong627/workdir/models/IP-Adapter`
- `/hpc2hdd/home/ckwong627/workdir/models/sdxl-turbo`

Important `IP-Adapter` files:

- `models/ip-adapter_sd15.bin`
- `sdxl_models/ip-adapter_sdxl_vit-h.safetensors`
- `sdxl_models/image_encoder/`

## How To Run

All commands below assume working directory:

```bash
cd /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2
```

### 1. Smoke test

Use this first to verify the environment, data paths, and a tiny end-to-end training loop.

```bash
sbatch slurm_scripts/run_smoke_test.sh
```

### 2. Cache visual backbone features

```bash
sbatch slurm_scripts/run_cache_backbone_features.sh
```

This creates:

- `cache/backbone_cache_train.pt`
- `cache/backbone_cache_test.pt`

### 3. Train the encoder

Warmup:

```bash
sbatch slurm_scripts/run_train_encoder_warmup.sh
```

Multitarget:

```bash
sbatch slurm_scripts/run_train_encoder_multitarget.sh
```

Finetune:

```bash
sbatch slurm_scripts/run_train_encoder_finetune.sh
```

### 4. Train the prior

```bash
sbatch slurm_scripts/run_train_prior.sh
```

### 5. Reconstruct images

```bash
sbatch slurm_scripts/run_reconstruct_all.sh
```

### 6. Evaluate and summarize

```bash
sbatch slurm_scripts/run_evaluate.sh
```

This writes:

- `results/metrics_v2_final.json`
- `results/task2_montage_v2_final_s00.png`
- `results/results_summary_en.md`
- `results/results_summary_zh.md`

## Manual CLI Entry Points

Cache:

```bash
python -u codes/cache_backbone_features.py --split all --batch_size 32
```

Encoder stages:

```bash
python -u codes/train_encoder.py --stage warmup --tag v2_warmup
python -u codes/train_encoder.py --stage multitarget --tag v2_multitarget --resume checkpoints/v2_warmup_best.pt
python -u codes/train_encoder.py --stage finetune --tag v2_final --resume checkpoints/v2_multitarget_best.pt
```

Prior:

```bash
python -u codes/train_prior.py --encoder_ckpt checkpoints/v2_final_best.pt --tag v2_prior
```

Reconstruction:

```bash
python -u codes/reconstruct.py \
  --encoder_ckpt checkpoints/v2_final_best.pt \
  --prior_ckpt checkpoints/v2_prior_best.pt \
  --tag v2_final \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

Evaluation:

```bash
python -u codes/evaluate.py \
  --tag v2_final \
  --encoder_ckpt checkpoints/v2_final_best.pt \
  --compare_v1
python -u codes/make_task2_montage.py --tag v2_final --seed_index 0 --num_samples 20
python -u codes/summarize_results.py --tag v2_final
```

## Current Results

The latest completed run is stored in:

- [results/metrics_v2_final.json](/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/results/metrics_v2_final.json)
- [results/results_summary_en.md](/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/results/results_summary_en.md)
- [results/task2_montage_v2_final_s00.png](/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/results/task2_montage_v2_final_s00.png)

### Retrieval

| Model | Top-1 | Top-5 |
|---|---:|---:|
| Version2 `v2_final` | 15.00% | 35.00% |
| Version1 Joint | 13.50% | 36.50% |
| Version1 Retrieval-only | 14.50% | 34.50% |

### Reconstruction

| Model | PixCorr | SSIM | CLIP-like score |
|---|---:|---:|---:|
| Version2 `v2_final` | 0.2754 | 0.3709 | 0.2779 |
| Version1 Joint | 0.0628 | 0.2762 | 0.7081 |

### Important evaluation note

The reconstruction `CLIP` score in `version2` is not directly comparable to `version1`.

- `version1` uses the TA-style `two-way identification` metric with `openai/CLIP ViT-L/14`
- `version2` currently reports mean cosine similarity against cached `ViT-H/14` image embeddings

So the `0.2779` value should not be interpreted as “much worse than 0.7081” without first unifying the evaluation protocol.

## Known Caveats

- `v2_final_best.pt` is selected by `Top-1`, not by `Top-5` or reconstruction quality.
- A higher intermediate `Top-5` appeared during finetuning, but it was not the checkpoint selected for final evaluation.
- The current finetune stage can slightly hurt retrieval relative to the best multitarget checkpoint.
- Reconstruction metrics are partially on a different scale from `version1`; use caution when comparing them.

## Recommended Next Steps

- save separate checkpoints for `best_top1`, `best_top5`, and reconstruction-selected models
- re-evaluate `v2_multitarget_best.pt` as a retrieval-oriented checkpoint
- switch `version2` reconstruction evaluation to the same official metric set used in `version1`
- retune the finetune stage so retrieval does not degrade after multitarget training
