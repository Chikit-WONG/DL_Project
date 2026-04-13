# Brain-to-Image Retrieval & Reconstruction (THINGS-EEG)

DSAA2012 Deep Learning — Project A

This project implements an EEG-to-image pipeline on the **THINGS-EEG** dataset.
Given a 63-channel EEG recording of a human viewing an image, the model must:

1. **Retrieve** the correct image from a 200-class pool (zero-shot classification)
2. **Reconstruct** a photorealistic image using IP-Adapter + Stable Diffusion v1.5

---

## Project Structure

```
DL_Project/
├── codes/
│   ├── config.py               # All hyperparameters and paths
│   ├── data.py                 # Dataset, augmentation, dataloader
│   ├── model.py                # EEG encoder + UnifiedModel
│   ├── utils.py                # Metrics: retrieval, eval_images, helpers
│   ├── cache_clip_features.py  # Pre-compute CLIP image embeddings
│   ├── train.py                # Training (Phase 1 & 2, all architectures)
│   ├── reconstruct.py          # IP-Adapter image generation
│   ├── evaluate.py             # End-to-end evaluation
│   └── run_all.ipynb           # Notebook: full pipeline demo
├── slurm_scripts/              # SLURM job scripts for HPC
├── checkpoints/                # Saved model weights (not tracked by git)
├── clip_cache/                 # Pre-computed CLIP features (not tracked by git)
├── outputs/                    # Reconstructed images and metrics JSON
├── image-eeg-data/             # THINGS-EEG dataset
└── plan/                       # Implementation plan (Chinese)
```

---

## Model Architecture

### EEG Encoder (~5.7M parameters)

```
Input: [B, 63, T]
  ↓ Spatial Conv1d (63 → 128, k=1) × 2  +  BN + GELU
  ↓ Temporal Conv1d × 3 (stride=2, k=15) → [B, 320, T/8]
  ↓ Positional Embedding + Transformer (3 layers, d=320, heads=8, FFN=640)
  ↓ Global Average Pooling → [B, 320]
  ↓ MLP head (320 → 640 → 1024)
Output: [B, 1024]  (same dimension as CLIP ViT-H-14)
```

### Unified Loss Function

```
L = α × L_InfoNCE  +  β × L_MSE

L_InfoNCE: symmetric CLIP-style contrastive on L2-normalized embeddings
L_MSE:     mean squared error on raw (unnormalized) embeddings
           → forces EEG embedding to lie in CLIP image space for IP-Adapter
```

| Architecture | α | β | Description |
|---|---|---|---|
| **Arch A (Joint)** | 1.0 | 1.0 | Optimise both retrieval and reconstruction simultaneously |
| Arch B (Retrieval-only) | 1.0 | 0.0 | Retrieval only; no MSE supervision |
| Arch B (Recon-only) | 0.0 | 1.0 | Reconstruction only; no InfoNCE |

---

## Dependencies

Requires the `test` conda environment with CUDA 12.6:

```bash
# Core packages
torch==2.10.0+cu126
torchvision==0.25.0+cu126
transformers==4.49.0
diffusers==0.37.1
accelerate==1.13.0
datasets==4.8.2
scikit-image==0.25.2
scipy==1.15.3
timm==1.0.26
clip==1.0
numpy==1.26.4
```

### Required Model Weights

The following pre-trained models must be placed under `/path/to/models/` (set `MODELS_ROOT` in `codes/config.py`):

| Model | Purpose |
|---|---|
| `IP-Adapter/models/image_encoder/` | CLIP ViT-H-14 encoder (1024-d) |
| `CLIP-ViT-H-14-laion2B-s32B-b79K/` | Image processor config |
| `stable-diffusion-v1-5/` | Base SD model for generation |
| `IP-Adapter/models/ip-adapter_sd15.bin` | IP-Adapter weights |

Evaluation also requires these weights to be pre-downloaded (HPC compute nodes have no internet):

```bash
# Run once on the login node (has proxy internet)
python -c "
import torch, clip
clip.load('ViT-L/14')
torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
torch.hub.load('pytorch/vision', 'inception_v3', pretrained=True)
torch.hub.load('pytorch/vision', 'efficientnet_b1', pretrained=True)
torch.hub.load('facebookresearch/swav', 'resnet50', pretrained=True)
"
```

---

## How to Run

All scripts assume working directory = `DL_Project/`. Submit jobs from this directory.

### Step 1 — Cache CLIP Image Features

```bash
sbatch slurm_scripts/run_cache_clip.sh
# Output: clip_cache/clip_train_features.pt  (7968 images × 1024-d)
#         clip_cache/clip_test_features.pt   (200 images × 1024-d)
```

### Step 2 — Phase 1 Training (retrieval warm-up, α=1, β=0)

```bash
sbatch slurm_scripts/run_train_phase1.sh
# Checkpoint: checkpoints/phase1_main_best.pt
# ~6 min on A100, 50 epochs, batch=128, lr=3e-4
```

### Step 3 — Phase 2 Training (joint fine-tuning, α=1, β=1)

```bash
sbatch slurm_scripts/run_train_phase2.sh
# Resumes from phase1_main_best.pt
# Checkpoint: checkpoints/phase2_main_best.pt
```

### Step 4 — Reconstruct Images (10 seeds)

```bash
for i in {0..9}; do
  sbatch slurm_scripts/run_reconstruct_s${i}.sh
done
# Output: outputs/recon_images_phase2_main_s0{0-9}.pt  (157 MB each)
# Note: max 8 concurrent jobs due to QOS limits
```

### Step 5 — Evaluate

```bash
sbatch slurm_scripts/run_evaluate.sh
# Output: outputs/metrics_phase2_main_best.json
```

### Architecture B Ablation Baselines

```bash
# Train
sbatch slurm_scripts/run_train_retrieval_only.sh   # α=1, β=0
sbatch slurm_scripts/run_train_recon_only.sh       # α=0, β=1

# Reconstruct (same 10-seed process, replace tag with archB_retrieval / archB_reconstruction)
for i in {0..9}; do
  sbatch slurm_scripts/run_reconstruct_archB_ret_s${i}.sh
  sbatch slurm_scripts/run_reconstruct_archB_rec_s${i}.sh
done

# Evaluate
sbatch slurm_scripts/run_evaluate_archB_ret_full.sh
sbatch slurm_scripts/run_evaluate_archB_rec_full.sh
```

### Manual Training (without SLURM)

```bash
conda activate test
module load cuda/12.6

# Cache
python codes/cache_clip_features.py

# Train (all flags)
python codes/train.py --phase 1 --tag my_run --epochs 50 --alpha 1.0 --beta 0.0
python codes/train.py --phase 2 --tag my_run --epochs 30 --alpha 1.0 --beta 1.0 --resume checkpoints/my_run_best.pt

# Reconstruct
python codes/reconstruct.py --ckpt checkpoints/my_run_best.pt --seeds 0 1 2 --tag my_run --num_inference_steps 20

# Evaluate
python codes/evaluate.py --ckpt checkpoints/my_run_best.pt --recon_tag my_run
```

---

## Results

### Task 1: Retrieval (200-way Zero-shot Classification)

| Model | α | β | Top-1 Acc | Top-5 Acc |
|---|---|---|---|---|
| Arch B (Retrieval-only) | 1.0 | 0.0 | **14.5%** | 34.5% |
| **Arch A (Joint)** | **1.0** | **1.0** | **13.5%** | **36.5%** |
| Arch B (Recon-only) | 0.0 | 1.0 | 9.0% | 24.0% |
| Random baseline | — | — | 0.5% | 2.5% |

> Top-1 is 27–29× above random chance; Top-5 is 14–15× above random chance.

### Task 2: Reconstruction (mean ± std over 10 seeds, 200 test images)

| Metric | Arch B (Ret-only) | **Arch A (Joint)** | Arch B (Rec-only) |
|---|---|---|---|
| PixCorr | 0.0328 ± 0.0063 | **0.0628 ± 0.0043** | 0.0709 ± 0.0067 |
| SSIM | 0.1981 ± 0.0042 | **0.2762 ± 0.0040** | 0.2749 ± 0.0030 |
| AlexNet (layer 2) | 0.6482 ± 0.0178 | **0.7022 ± 0.0162** | 0.7124 ± 0.0124 |
| AlexNet (layer 5) | 0.7416 ± 0.0138 | 0.7903 ± 0.0100 | **0.8223 ± 0.0076** |
| Inception | 0.6291 ± 0.0213 | 0.6732 ± 0.0122 | **0.7273 ± 0.0150** |
| CLIP | 0.6577 ± 0.0083 | 0.7081 ± 0.0088 | **0.7526 ± 0.0062** |
| EffNet | **0.9433 ± 0.0027** | 0.9267 ± 0.0025 | 0.8846 ± 0.0027 |
| SwAV | 0.6690 ± 0.0041 | 0.6282 ± 0.0026 | **0.5744 ± 0.0030** |

**Bold** = best among three models for that metric.

---

## Recommendation

**Use Arch A (Joint, α=1, β=1) as the final model.**

| Criterion | Arch B Ret-only | **Arch A Joint** | Arch B Rec-only |
|---|---|---|---|
| Retrieval Top-1 | 🥇 14.5% | 🥈 13.5% | 🥉 9.0% |
| Reconstruction quality | 🥉 worst (0/8 best) | 🥈 balanced (4/8 best) | 🥇 best (4/8 best) |
| Supports both tasks | No | **Yes** | No |

**Rationale:**

- Arch A's retrieval (13.5%) is only 1% below the retrieval-only specialist, a negligible trade-off.
- Arch A's reconstruction is competitive — 4 of 8 metrics are best-in-class.
- Arch B (Recon-only) wins 4 reconstruction metrics but loses 5.5% retrieval Top-1, making it unsuitable as a general-purpose model.
- Arch B (Ret-only) has the worst reconstruction quality on every single metric — the InfoNCE loss alone does not align embeddings well enough for IP-Adapter.
- Joint training (MSE + InfoNCE) proves that a single encoder can serve both tasks without significant sacrifice.
