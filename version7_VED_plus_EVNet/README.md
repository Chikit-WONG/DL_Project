# Version 7: VED + EVNet — EEG-to-Image Retrieval

[← Back to project root](../README.md) | [中文 README](README-CN.md)

This version extends [`version5_VED`](../version5_VED/README.md) by adding a biologically-inspired EVNet visual frontend to the multi-scale blur CLIP pipeline. It covers **Task 1 (EEG-to-image retrieval) only**; for Task 2 reconstruction see `version5_VED` or `version6_BP-MGD`.

## Table of Contents

- [Method Overview](#method-overview)
- [Architecture](#architecture)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Running the Pipeline](#running-the-pipeline)
- [Experimental Results](#experimental-results)
- [Ablation Studies](#ablation-studies)
- [Limitations](#limitations)

---

## Method Overview

The core task is **EEG-to-image retrieval**: given an EEG recording of a subject viewing an image, identify that image from a candidate pool of 200 test images.

**Version 7** extends the baseline VED pipeline with an **EVNet fixed frontend**. EVNet simulates the primate early visual pathway — specifically the retina/LGN (subcortical processing) and primary visual cortex V1 — and is used here as a frozen image feature extractor. Its output is fused with multi-scale Gaussian-blur CLIP features to form a richer image representation that the EEG encoder is trained to align with.

The fusion is:

```
fused = w₀ · blur_agg + w₁ · evnet_feat
```

where `w₀` and `w₁` are softmax-normalised learnable scalars (initialised to 0.7 / 0.3), and `blur_agg` is an attention-weighted sum over the multi-scale blur feature stack.

---

## Architecture

### Image Feature Extraction (offline, pre-computed)

```
Input image (224×224)
       │
       ├──── Multi-scale Gaussian blur (8 or 12 levels)
       │            └── CLIP encoder (RN50 or ViT-H/14)
       │                   └── blur feature stack  [num_levels × 1024-dim]
       │
       └──── EVNet Frontend (frozen)
                  ├── SubcorticalBlock   (retina / LGN)
                  ├── VOneBlock          (V1 Gabor filters)
                  ├── Conv2d adapter     (512→3 ch, Kaiming init, frozen)
                  └── CLIP encoder (RN50 or ViT-H/14)
                         └── EVNet feature  [1024-dim]
```

**Blur level presets:**

| Config | Levels |
|--------|--------|
| 8-blur | `l_1, l_3, l_15, l_21, l_33, l_45, l_57, l_63` |
| 12-blur | `l_1, l_3, l_9, l_15, l_21, l_27, l_33, l_39, l_45, l_51, l_57, l_63` |

### EEG Encoder (`Brain_Visual_Encoder_EEG`)

```
EEG input [B, 63ch, 250t]
       │
       ├── Conv2dWithAbs  (spatial: 63ch → 25 filters)
       ├── BatchNorm2d
       ├── Linear(250→200) + ELU + Dropout(0.25)
       ├── Linear(200→200) + ELU + Dropout(0.65)
       └── Linear(25×200 → 1152-dim)   ← EEG embedding
```

### Fusion & Loss

At training time, the model receives pre-computed `blur_stack` and `evnet_feat` tensors. The image branch computes:

```
blur_agg = Σ softmax(learned_scale) · blur_stack    # attention over blur levels
fused    = softmax(fusion_logits) · [blur_agg, evnet_feat]
img_emb  = fusion_adapter(fused)                    # MLP: 1152→768→1152
```

Loss: **InfoNCE** (symmetric contrastive loss) between EEG embeddings and image embeddings.

---

## Environment Setup

### Requirements

- Python 3.9+
- PyTorch ≥ 2.0 with CUDA
- `open-clip-torch`
- `numpy`, `scipy`, `pandas`, `tqdm`, `opencv-python`, `Pillow`
- EVNet (bundled in `evnet/`)

### Installation

```bash
# 1. Activate your environment
conda activate test   # or create a new one

# 2. Install Python dependencies
pip install torch torchvision open-clip-torch numpy scipy pandas tqdm opencv-python Pillow

# 3. EVNet is imported directly from the bundled evnet/ directory.
#    No separate install is required — process_image_course.py adds evnet/ to sys.path automatically.
```

### CLIP Model Checkpoints

| Backbone | File | Size |
|----------|------|------|
| RN50 (OpenAI) | `open_clip_pytorch_model.bin` | ~102 MB |
| ViT-H/14 (LAION-2B) | `open_clip_pytorch_model.bin` | ~3.9 GB |

Place each checkpoint in a directory of your choice and pass it via `--clip_checkpoint`.

---

## Data Preparation

### EEG Data

Expected layout under `data/things-eeg/` (or via `--eeg_data_dir`):

```
Preprocessed_data_250Hz_whiten/
└── sub-01/
    ├── train.pt    # dict: {'eeg': Tensor[N,1,63,250], 'img': array[N,k,path]}
    └── test.pt
```

Symlinks to the course dataset are already configured:

```
data/things-eeg/Image_set/train_images -> .../image-eeg-data/training_images
data/things-eeg/Image_set/test_images  -> .../image-eeg-data/test_images
```

### Image Features

Pre-computed features are stored in `output/Image_feature/`. Approximate sizes:

| File | Description | Size |
|------|-------------|------|
| `MultiBlur_RN50_train.pt` | 8/12-level blur, RN50 (symlink to v5) | — |
| `MultiBlur_RN50_test.pt` | | — |
| `EVNet_RN50_train.pt` | EVNet+RN50 features | ~67 MB |
| `EVNet_RN50_test.pt` | | ~896 KB |
| `MultiBlur_ViTH14_train.pt` | 8/12-level blur, ViT-H/14 | ~791 MB |
| `MultiBlur_ViTH14_test.pt` | | ~9.7 MB |
| `EVNet_ViTH14_train.pt` | EVNet+ViT-H/14 features | ~67 MB |
| `EVNet_ViTH14_test.pt` | | ~896 KB |
| `EVNet_xavier_RN50_*.pt` | Xavier-init adapter variant | ~67 MB each (train) |
| `EVNet_gap_*.pt` | GAP+Linear, no backbone | ~67 MB each (train) |

---

## Running the Pipeline

### Step 1: Generate Image Features

```bash
# RN50 + EVNet (random/Kaiming init adapter)
python preprocess/process_image_course.py \
    --clip_checkpoint /path/to/CLIP-RN50/open_clip_pytorch_model.bin \
    --save_dir output/Image_feature \
    --backbone rn50 \
    --evnet_mode random \
    --batch_size 128

# ViT-H/14 + EVNet
python preprocess/process_image_course.py \
    --clip_checkpoint /path/to/CLIP-ViT-H-14/open_clip_pytorch_model.bin \
    --save_dir output/Image_feature \
    --backbone vit_h_14 \
    --evnet_mode random \
    --batch_size 64
```

`--evnet_mode` options: `random` (Kaiming Conv2d init), `xavier` (Xavier Conv2d init), `gap` (GlobalAvgPool + Linear, no backbone).

Output file naming convention:

| `--backbone` | `--evnet_mode` | Blur prefix | EVNet prefix |
|---|---|---|---|
| `rn50` | `random` | `MultiBlur_RN50` | `EVNet_RN50` |
| `rn50` | `xavier` | `MultiBlur_RN50` | `EVNet_xavier_RN50` |
| `rn50` | `gap` | `MultiBlur_RN50` | `EVNet_gap` |
| `vit_h_14` | `random` | `MultiBlur_ViTH14` | `EVNet_ViTH14` |

### Step 2: Train

```bash
# 8-blur + EVNet fixed, RN50, 95/5 split
python main_eeg_course.py \
    --blur_config 8 \
    --use_evnet \
    --epoch 200 \
    --train_batch_size 1024 \
    --lr 0.001 \
    --n_seeds 10 \
    --first_seed 21 \
    --feature_path output/Image_feature \
    --output_dir output/logs/8blur_evnet_split

# Full training set (no validation split)
python main_eeg_course.py \
    --blur_config 8 \
    --use_evnet \
    --use_full_train \
    --epoch 200 \
    --train_batch_size 1024 \
    --lr 0.001 \
    --n_seeds 10 \
    --first_seed 21 \
    --feature_path output/Image_feature \
    --output_dir output/logs/8blur_evnet_full

# ViT-H/14 backbone variant
python main_eeg_course.py \
    --blur_config 8 \
    --use_evnet \
    --blur_prefix MultiBlur_ViTH14 \
    --evnet_prefix EVNet_ViTH14 \
    --epoch 200 \
    --train_batch_size 1024 \
    --lr 0.001 \
    --n_seeds 10 \
    --first_seed 21 \
    --feature_path output/Image_feature \
    --output_dir output/logs/8blur_evnet_vith14_split
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--blur_config` | `8` | Blur level preset: `8` or `12` |
| `--use_evnet` | off | Enable EVNet feature fusion |
| `--blur_prefix` | `MultiBlur_RN50` | Prefix of blur `.pt` files |
| `--evnet_prefix` | `EVNet_RN50` | Prefix of EVNet `.pt` files |
| `--use_full_train` | off | Train on the full training set (no val split) |
| `--epoch` | 200 | Number of training epochs |
| `--train_batch_size` | 1024 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--n_seeds` | 10 | Number of random seeds to run |
| `--first_seed` | 21 | First seed value (seeds = first_seed … first_seed+n_seeds-1) |
| `--eeg_data_dir` | — | Override EEG data path |

### SLURM Scripts

Pre-configured SLURM scripts are in `slurm_scripts/`:

| Script | Description |
|--------|-------------|
| `01_gen_evnet_features.sh` | Generate RN50 EVNet features |
| `02_train_8blur_evnet_split.sh` | 8-blur + EVNet, RN50, 95/5 split |
| `03_train_12blur_evnet_split.sh` | 12-blur + EVNet, RN50, 95/5 split |
| `04_full_train_8blur_evnet.sh` | 8-blur + EVNet, RN50, full train |
| `05_full_train_12blur_evnet.sh` | 12-blur + EVNet, RN50, full train |
| `06_gen_evnet_xavier_features.sh` | Generate Xavier-init adapter features |
| `07_gen_evnet_gap_features.sh` | Generate GAP+Linear features |
| `08_train_8blur_evnet_xavier_split.sh` | Ablation: Xavier init |
| `09_train_8blur_evnet_gap_split.sh` | Ablation: GAP+Linear (no backbone) |
| `10_gen_vith14_features.sh` | Generate ViT-H/14 features |
| `11_train_8blur_evnet_vith14_split.sh` | Ablation: ViT-H/14 backbone |

---

## Experimental Results

All experiments: 10 random seeds (seeds 21–30), 200 epochs, batch size 1024, lr 0.001, single subject (sub-01).

**Val-selected**: checkpoint selected by best validation Top-1, evaluated on the 200-way test set.  
**Best-test**: best test Top-1 observed across all epochs.

### Main Experiments (95/5 Train/Val Split)

| Experiment | Val-sel Top-1 | Val-sel Top-5 | Best-test Top-1 | Best-test Top-5 |
|---|---|---|---|---|
| 8-blur + EVNet fixed (RN50) | 0.8460 ± 0.0135 | 0.9870 ± 0.0059 | 0.8715 ± 0.0091 | 0.9860 ± 0.0081 |
| 12-blur + EVNet fixed (RN50) | 0.8400 ± 0.0186 | 0.9860 ± 0.0046 | 0.8715 ± 0.0111 | 0.9855 ± 0.0028 |

### Full Training Set (No Val Split)

| Experiment | Val-sel Top-1 | Val-sel Top-5 | Best-test Top-1 | Best-test Top-5 |
|---|---|---|---|---|
| 8-blur + EVNet fixed (RN50) | 0.8530 ± 0.0136 | 0.9860 ± 0.0046 | 0.8785 ± 0.0082 | 0.9855 ± 0.0037 |
| 12-blur + EVNet fixed (RN50) | 0.8505 ± 0.0169 | 0.9845 ± 0.0037 | 0.8810 ± 0.0074 | 0.9850 ± 0.0041 |

Using the full training set consistently improves best-test Top-1 by ~0.007–0.010 over the 95/5 split.

---

## Ablation Studies

All ablations use the 8-blur + RN50 baseline as reference (95/5 split).

| Ablation | Val-sel Top-1 | Best-test Top-1 | Δ vs Baseline (val-sel) |
|---|---|---|---|
| **Baseline**: EVNet fixed, Kaiming init (RN50) | 0.8460 ± 0.0135 | 0.8715 ± 0.0091 | — |
| Xavier init adapter (RN50) | 0.8275 ± 0.0175 | 0.8495 ± 0.0086 | −0.019 |
| GAP + Linear (no CLIP backbone) | 0.8285 ± 0.0173 | 0.8620 ± 0.0092 | −0.018 |
| EVNet fixed, Kaiming init (ViT-H/14) | 0.7365 ± 0.0208 | 0.7790 ± 0.0115 | −0.110 |

**Key findings:**

- **Kaiming > Xavier**: Random (Kaiming) init outperforms Xavier uniform init for the frozen convolutional adapter by ~0.019 val Top-1. Xavier tends to produce smaller initial magnitudes, which may leave the frozen adapter in a less expressive state.
- **GAP ablation**: Removing the CLIP backbone entirely (replacing with AdaptiveAvgPool2d + Linear) loses only ~0.018 val Top-1, showing that EVNet's V1-like features alone carry substantial relevant information. The best-test gap is only 0.009.
- **ViT-H/14 substantially underperforms RN50** (−0.110 val Top-1). ViT-H/14 uses patch-based self-attention and expects clean pixel patches as input; the EVNet convolutional adapter produces spatially-transformed feature maps that disrupt the token structure ViT relies on. RN50 as a CNN backbone is natively compatible with the spatially-processed EVNet output.

---

## Limitations

1. **Single subject only.** All experiments use subject sub-01. Generalisation across subjects has not been evaluated.

2. **ViT-H/14 incompatibility with EVNet.** The EVNet adapter (SubcorticalBlock → VOneBlock → Conv2d) produces a spatially-transformed image that is well-suited for CNN backbones like RN50. Transformer-based CLIP encoders (ViT-*) process images as fixed-size patch sequences and do not handle EVNet-preprocessed inputs as effectively, leading to a ~11% absolute drop in Top-1 accuracy.

3. **EVNet adapter is frozen at random initialisation.** The Conv2d adapter weights are initialised with Kaiming normal (default) or Xavier uniform and immediately frozen. No fine-tuning is performed on the visual pathway. An end-to-end learned adapter could potentially close the gap.

4. **No cross-subject or cross-session validation.** The 95/5 split and full-train experiments both use intra-subject data. Temporal generalisation (different recording sessions) is not addressed.

5. **Single modality.** The model accepts only EEG signals. Multi-modal brain signals (e.g., fMRI) or richer EEG paradigms have not been explored.

6. **Gaussian blur as the sole image degradation.** The multi-scale feature approach uses only Gaussian blur. Other perceptually meaningful transforms (frequency masking, spatial phase scrambling) may be complementary.

7. **Evaluation is retrieval-only.** Top-1 / Top-3 / Top-5 accuracy on a 200-way forced-choice retrieval task is the sole metric. Image generation or semantic similarity metrics are not evaluated.
