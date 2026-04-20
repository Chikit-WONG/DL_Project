# EEG Brain-to-Image Retrieval & Reconstruction (ATMS)

**[English]** | [中文说明](README-CN.md)

---

DSAA2012 Deep Learning Final Project — Prof. Chen Liang  
Dataset: THINGS-EEG (sub-01, 63 EEG channels × 250 time points, 1654 training classes, 200 test classes)

## Overview

This project implements EEG-based brain-to-image decoding using the **ATMS** (Attention-based Time-series to Multi-modal Space) encoder.  
The pipeline has two branches:

| Branch | Description |
|--------|-------------|
| **Retrieval** | Train ATMS to align EEG embeddings with CLIP ViT-H-14 image features; evaluate 200-way Top-1/Top-5 accuracy |
| **Reconstruction** | Fine-tune ATMS for regression; feed embeddings into SD v1.5 + IP-Adapter to generate images |

### Architecture

```
EEG [B, 63, 250]
  → iTransformer encoder (subject embedding + attention)
  → PatchEmbedding (temporal CNN) → flatten [B, 1440]
  → Linear projection → 1024-dim CLIP ViT-H-14 space
  → [Retrieval] cosine similarity against 200 CLIP image features
  → [Generation] IP-Adapter → Stable Diffusion v1.5 → 512×512 image
```

---

## Results (sub-01, 40 training epochs)

### Retrieval (200-way)

| Metric | Score |
|--------|-------|
| Top-1 Accuracy | **29.00%** |
| Top-5 Accuracy | **62.00%** |

### Reconstruction

| Metric | Score |
|--------|-------|
| CLIP Score | **0.6696** |
| SSIM | **0.2852** |
| PixCorr | **0.0387** |
| AlexNet-2 | 0.7164 |
| AlexNet-5 | 0.7684 |
| Inception | 0.6603 |
| EffNet (↓ lower=better) | 0.9380 |
| SwAV (↓ lower=better) | 0.6453 |

Detailed CSV results: [`outputs/retrieval_eval_run01.csv`](outputs/retrieval_eval_run01.csv) · [`outputs/reconstruction_eval_run01.csv`](outputs/reconstruction_eval_run01.csv)

---

## Environment Setup

### Prerequisites

- Python 3.10
- CUDA 12.6
- Conda (recommended)

### Install

```bash
# Create and activate a conda environment
conda create -n eeg_atm python=3.10 -y
conda activate eeg_atm

# Install PyTorch (adjust CUDA version if needed)
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install all other dependencies
pip install -r requirements.txt
```

### Pretrained Models Required

| Model | Path on Cluster |
|-------|----------------|
| CLIP ViT-H-14 (OpenCLIP) | `/hpc2hdd/home/ckwong627/workdir/models/CLIP-ViT-H-14-laion2B-s32B-b79K/` |
| Stable Diffusion v1.5 | `/hpc2hdd/home/ckwong627/workdir/models/stable-diffusion-v1-5/` |
| IP-Adapter SD1.5 ViT-H | `/hpc2hdd/home/ckwong627/workdir/models/IP-Adapter/models/ip-adapter_sd15.bin` |

### Data

Place the THINGS-EEG preprocessed data at:

```
/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/
  ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data/
    train.pt
    test.pt
    test_images/
      <class_name>/
        *.jpg
```

### Smoke Test (CPU, no GPU required)

```bash
cd version3_ATM
python smoke_test.py
```

All 12 checks should print `[OK]` and the final line should read `SMOKE TEST PASSED`.

---

## Full Pipeline

### Step 1 — Train Retrieval Model

```bash
sbatch slurm_scripts/run_train_retrieval.sh
# Saves checkpoints to: models/contrast/ATMS/sub-01/<timestamp>/{5,10,...,40}.pth
```

### Step 2 — Train Reconstruction Model

```bash
sbatch slurm_scripts/run_train_reconstruction.sh
# Saves checkpoints to: models/contrast/ATMS/sub-01/<timestamp>/{5,10,...,40}.pth
```

### Step 3 — Evaluate Retrieval

```bash
sbatch slurm_scripts/run_eval_retrieval.sh \
  ./models/contrast/ATMS/sub-01/<timestamp>/40.pth run01
# Output: outputs/retrieval_eval_run01.csv
```

### Step 4 — Generate Reconstructed Images

```bash
sbatch slurm_scripts/run_generate_recon.sh \
  ./models/contrast/ATMS/sub-01/<timestamp>/40.pth run01
# Output: outputs/reconstructions/run01/{ground_truth/, generated/, recon_tensors.pt}
```

### Step 5 — Evaluate Reconstruction Metrics

```bash
sbatch slurm_scripts/run_eval_reconstruction.sh \
  ./outputs/reconstructions/run01/recon_tensors.pt run01
# Output: outputs/reconstruction_eval_run01.csv
```

---

## Output Locations

| Output | Path |
|--------|------|
| Training checkpoints | `models/contrast/ATMS/sub-01/<timestamp>/<epoch>.pth` |
| Training loss curves | `outputs/contrast/ATMS/sub-01/<timestamp>/ATMS_sub-01.csv` |
| Retrieval eval CSV | `outputs/retrieval_eval_<run>.csv` |
| Ground-truth images (256×256) | `outputs/reconstructions/<run>/ground_truth/<idx>.png` |
| Generated images (256×256) | `outputs/reconstructions/<run>/generated/<idx>.png` |
| Image tensors for eval | `outputs/reconstructions/<run>/recon_tensors.pt` |
| Reconstruction metrics CSV | `outputs/reconstruction_eval_<run>.csv` |
| SLURM job logs | `logs/<job_type>_<jobid>.{out,err}` |

---

## Key Bugs Fixed

Three non-trivial bugs were found and fixed during this project:

1. **`num_subjects` mismatch** — The training script creates an iTransformer with `num_subjects=10` (a [10, 250] subject embedding table). The eval and generation scripts were using the default `num_subjects=2`, causing a `size mismatch` error on checkpoint load. Fixed: pass `num_subjects=10` explicitly.

2. **Segmentation fault on diffusers import** — Importing `diffusers` lazily inside a function body (after a CUDA context was already established by the EEG encoder) caused a C-extension conflict. Fixed: move all `diffusers` imports to module top level, and free the EEG model (`del model; torch.cuda.empty_cache()`) before loading SD.

3. **Double IP-Adapter projection** — The original script pre-projected EEG embeddings through `ImageProjection` (1024→4×768), then passed them to `pipe(ip_adapter_image_embeds=...)`. But diffusers' UNet applies `encoder_hid_proj` (the same `ImageProjection`) on the image embeds again internally, causing a shape mismatch (`8×768` × `1024×3072`). Fixed: pass raw [N, 1024] embeddings and let the UNet project them.

---

## Limitations

- **Single subject**: all results are for `sub-01` only; cross-subject generalisation was not evaluated.
- **Short training**: 40 epochs on a single A40 GPU; the original ATMS paper uses longer schedules.
- **Reconstruction quality**: CLIP score 0.67 is noticeably below state-of-the-art (~0.83 from CognitionCapturerPro with 80-epoch retrieval + IP-Adapter). The ATMS reconstruction branch uses a regression loss rather than contrastive alignment, which limits embedding quality.
- **EffNet / SwAV distances** are high (closer to random), indicating generated textures and low-level features do not closely match ground truth.
- **No augmentation or multi-trial ensemble**: averaging trials at test time is required by the course protocol but loses temporal variability information.

---

## Project Structure

```
version3_ATM/
├── EEG-preprocessing/          # Raw EEG preprocessing utilities
├── eval/
│   ├── eval_retrieval_200way.py    # 200-way retrieval evaluation
│   └── eval_reconstruction_metrics.py  # SSIM / CLIP / AlexNet / EffNet metrics
├── Generation/
│   ├── ATMS_reconstruction.py      # Reconstruction training script
│   └── generate_reconstructions.py # SD v1.5 + IP-Adapter image generation
├── Retrieval/
│   └── ATMS_retrieval.py           # Retrieval training script
├── models/
│   ├── data_bridge.py              # Unified data loader (train/test .pt)
│   ├── clip_bridge.py              # OpenCLIP ViT-H-14 helper
│   ├── loss.py                     # CLIP contrastive loss
│   └── subject_layers/             # iTransformer, attention, embedding layers
├── slurm_scripts/                  # SBATCH submission scripts
├── outputs/                        # Evaluation CSVs and generated images
├── smoke_test.py                   # CPU-only sanity check
├── requirements.txt
└── README.md / README-CN.md
```

---

## License

Original codebase: [EEG_Image_decode](https://github.com/eegatlas/EEG_Image_decode) — see [LICENSE](LICENSE).  
Modifications and additions: Chi Kit Wong, April 2026.
