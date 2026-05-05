# DSAA2012 Final Project: EEG-to-Image Retrieval and Reconstruction

[中文 README](README-CN.md)

This repository implements a complete EEG-to-image system on the THINGS-EEG dataset, covering both required tasks:

1. **Task 1 — Brain-to-Image Retrieval**: given an EEG segment, rank the correct stimulus image among a 200-class candidate pool.
2. **Task 2 — Brain-to-Image Reconstruction**: given an EEG segment, generate an image that is structurally and semantically consistent with the viewed stimulus.

---

## Environment Setup

Task 1 and Task 2 share one conda environment. Task 2's pinned dependencies (`torch==2.5.0`, `open-clip-torch==3.2.0`, `numpy==2.0.2`) satisfy Task 1's looser requirements, so a single environment covers both tasks.

```bash
# 1. Create and activate the environment (Python 3.10 required)
conda create -n DL_Project python=3.10 -y
conda activate DL_Project

# 2. Install PyTorch with CUDA support
#    Adjust cu121 to match your CUDA version (e.g. cu118, cu124)
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt
```

EVNet (Task 1) is bundled in `task1/evnet/` — no separate install is needed.

**Required external models (download separately):**

| Model | Size | Purpose |
|-------|------|---------|
| OpenCLIP RN50 (`open_clip_pytorch_model.bin`) | ~102 MB | Task 1 blur and EVNet image encoder |
| OpenCLIP ViT-H-14 LAION-2B (`open_clip_pytorch_model.bin`) | ~4.4 GB | Task 2 multi-modal supervision encoder |
| SDXL-Turbo | ~6 GB | Task 2 image generation backbone |
| IP-Adapter (`ip-adapter-plus_sdxl_vit-h`) | ~1 GB | Task 2 image conditioning |

---

## Repository Layout

```text
DL_Project/
├── requirements.txt                # Shared dependencies for Task 1 and Task 2
├── task1/                          # Task 1: EEG-to-image retrieval (VED + EVNet)
│   ├── main_eeg_course.py          # Training & evaluation entry point
│   ├── preprocess/
│   │   └── process_image_course.py # Offline image feature extraction
│   ├── models/                     # EEG encoder definitions
│   ├── scripts/
│   │   └── evaluate_course_metrics.py
│   ├── evnet/                      # Bundled EVNet library
│   └── slurm_scripts/              # HPC job scripts
└── task2/                          # Task 2: EEG-to-image reconstruction (CognitionCapturerPro)
    ├── main.py                     # Training entry point
    ├── smoke_test.py               # 13-check validation script
    ├── configs/                    # YAML configs
    ├── src/cogcappro/              # Core package
    └── slurm_scripts/              # HPC job scripts
```

---

## Task 1: EEG-to-Image Retrieval

### Method Overview

The retrieval model learns to map raw EEG signals into the CLIP embedding space, where they can be directly compared against pre-computed image embeddings.

The image representation fuses two complementary sources:

1. **Multi-scale blur features**: each training/test image is rendered at 8 or 12 levels of Gaussian blur and encoded by OpenCLIP RN50. The resulting feature stack is aggregated with learned softmax attention weights.

2. **EVNet features**: a biologically-inspired visual frontend — SubcorticalBlock (retina/LGN) + VOneBlock (V1 Gabor filters) — processes the image before an OpenCLIP RN50 encoder. All EVNet and adapter weights are **frozen at random initialisation**; only the downstream fusion and EEG encoder are trained.

The two sources are blended by learnable softmax weights (initialised 0.7 / 0.3):

```
fused_img = softmax([w_blur, w_evnet]) · [blur_agg, evnet_feat]
img_emb   = fusion_adapter(fused_img)    # MLP 1152→768→1152
```

The EEG encoder maps raw signals to the same 1152-dimensional space, and the model is trained with **InfoNCE (symmetric contrastive loss)**.

### Architecture

```
EEG input [B, 63 ch, 250 t]
  └─ Conv2dWithAbs (63→25 filters, spatial)
  └─ BatchNorm2d
  └─ Linear(250→200) + ELU + Dropout(0.25)
  └─ Linear(200→200) + ELU + Dropout(0.65)
  └─ Linear(25×200→1152)
  └─ EEG embedding [B, 1152]

Image input (offline, pre-computed)
  ├─ Multi-blur: CLIP RN50 × 8 levels → [B, 8, 1024]
  │    └─ Attention aggregation → blur_agg [B, 1024]
  └─ EVNet: SubcorticalBlock → VOneBlock → Conv2d adapter → CLIP RN50 → [B, 1024]
       └─ evnet_feat [B, 1024]

Fusion: softmax([w0, w1]) · [blur_agg, evnet_feat] → MLP → img_emb [B, 1152]

Loss: InfoNCE(EEG_emb, img_emb)
```

**Blur level preset (8-level, used for final submission):**
`σ ∈ {l_1, l_3, l_15, l_21, l_33, l_45, l_57, l_63}`

### Data Layout

Place the course-provided `image-eeg-data/` folder directly inside `DL_Project/`:

```text
DL_Project/
├── image-eeg-data/          ← drop the dataset folder here
│   ├── train.pt
│   ├── test.pt
│   ├── training_images/
│   ├── test_images/
│   └── converted_for_cogcappro/   ← pre-built, no extra steps needed
└── task1/
```

Both `main_eeg_course.py` and `process_image_course.py` auto-detect `image-eeg-data/` from this location. No `--eeg_data_dir` flag or manual symlinks are needed.

### Step-by-Step: Running Task 1

**Step 1 — Generate offline image features** (run once, ~15 min on A40):

```bash
cd task1
python preprocess/process_image_course.py \
    --clip_checkpoint /path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin \
    --save_dir output/Image_feature \
    --backbone rn50 \
    --evnet_mode random \
    --batch_size 128
```

Outputs `MultiBlur_RN50_train.pt`, `MultiBlur_RN50_test.pt`, `EVNet_RN50_train.pt`, `EVNet_RN50_test.pt` into `task1/output/Image_feature/`.

**Step 2 — Train the EEG retrieval model** (10 seeds, 200 epochs, ~8–16 h on A40):

```bash
python main_eeg_course.py \
    --blur_config 8 \
    --use_evnet \
    --epoch 200 \
    --train_batch_size 1024 \
    --lr 0.001 \
    --n_seeds 10 \
    --first_seed 21 \
    --eeg_data_dir /path/to/Preprocessed_data_250Hz_whiten/sub-01 \
    --feature_path output/Image_feature \
    --output_dir output/logs/8blur_evnet_full
```

Add `--use_full_train` to train on the full training set (recommended for best score).

**Step 3 — Evaluate:**

```bash
python scripts/evaluate_course_metrics.py \
    --log_dir output/logs/8blur_evnet_full
```

**Key CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--blur_config` | `8` | Blur level preset: `8` or `12` |
| `--use_evnet` | off | Enable EVNet feature fusion |
| `--use_full_train` | off | Train on full set (no validation split) |
| `--epoch` | `200` | Training epochs |
| `--train_batch_size` | `1024` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--n_seeds` | `10` | Number of random seeds |
| `--first_seed` | `21` | First seed (runs seeds 21–30) |
| `--eeg_data_dir` | — | Path to `sub-01/` preprocessed EEG folder |
| `--feature_path` | `output/Image_feature` | Directory containing `.pt` feature files |
| `--output_dir` | `output/logs/main_eeg_course` | Output directory |

**SLURM (HPC):** pre-configured scripts are in `task1/slurm_scripts/`. Set `CLIP_RN50` before submitting (the script fails immediately if unset). `EEG_DATA_DIR` is auto-detected from `image-eeg-data/` and is only needed if your data is in a non-standard location.

```bash
export CLIP_RN50=/path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin
# export EEG_DATA_DIR=/path/to/Preprocessed_data_250Hz_whiten/sub-01  # only if not using default layout

sbatch task1/slurm_scripts/01_gen_evnet_features.sh       # generate features (once)
sbatch task1/slurm_scripts/04_full_train_8blur_evnet.sh   # full train, best result
```

### Task 1 Results

All experiments: 10 random seeds (seeds 21–30), 200 epochs, batch size 1024, lr 0.001, single subject (sub-01), 200-way retrieval.

**Val-selected**: checkpoint selected by best validation Top-1 (from 5% held-out split).
**Best-test**: highest test Top-1 observed across all epochs.

#### Main Experiments

| Setting | Val-sel Top-1 | Val-sel Top-5 | Best-test Top-1 | Best-test Top-5 |
|---|---|---|---|---|
| 8-blur + EVNet, 95/5 split | 0.8460 ± 0.0135 | 0.9870 ± 0.0059 | 0.8715 ± 0.0091 | 0.9860 ± 0.0081 |
| 12-blur + EVNet, 95/5 split | 0.8400 ± 0.0186 | 0.9860 ± 0.0046 | 0.8715 ± 0.0111 | 0.9855 ± 0.0028 |
| **8-blur + EVNet, full train** | **0.8530 ± 0.0136** | **0.9860 ± 0.0046** | **0.8785 ± 0.0082** | **0.9855 ± 0.0037** |
| 12-blur + EVNet, full train | 0.8505 ± 0.0169 | 0.9845 ± 0.0037 | 0.8810 ± 0.0074 | 0.9850 ± 0.0041 |

**Chosen submission: 8-blur + EVNet, full train — Best-test Top-1 = 87.85%, Top-5 = 98.55%.**

Using the full training set improves best-test Top-1 by ~0.007–0.010 over the 95/5 split.

#### Ablation Studies

All ablations use 8-blur, RN50 backbone, 95/5 split.

| Ablation | Val-sel Top-1 | Best-test Top-1 | Δ vs Baseline |
|---|---|---|---|
| **Baseline: EVNet fixed, Kaiming init** | 0.8460 ± 0.0135 | 0.8715 ± 0.0091 | — |
| Xavier init adapter | 0.8275 ± 0.0175 | 0.8495 ± 0.0086 | −0.019 |
| GAP + Linear (no CLIP backbone) | 0.8285 ± 0.0173 | 0.8620 ± 0.0092 | −0.018 |
| ViT-H/14 backbone | 0.7365 ± 0.0208 | 0.7790 ± 0.0115 | −0.110 |

**Findings:**
- **Kaiming > Xavier**: Kaiming normal init outperforms Xavier by ~0.019 val Top-1. Xavier's smaller initial weight magnitudes leave the frozen adapter in a less expressive state.
- **GAP is surprisingly competitive**: removing the CLIP backbone entirely (replace with global average pooling + linear projection) only loses ~0.018 val Top-1, showing that EVNet's V1-like features alone carry substantial visual information useful for EEG alignment.
- **ViT-H/14 is incompatible with EVNet**: ViT uses patch-based attention and expects clean pixel patches; EVNet's spatial preprocessing disrupts the token structure, causing a −0.11 drop. RN50 as a CNN is natively compatible with EVNet's convolutional output.

**Running ablations:**

```bash
# Xavier init
python preprocess/process_image_course.py --evnet_mode xavier ...
python main_eeg_course.py --evnet_prefix EVNet_xavier_RN50 ...

# GAP+Linear (no backbone)
python preprocess/process_image_course.py --evnet_mode gap ...
python main_eeg_course.py --evnet_prefix EVNet_gap ...

# ViT-H/14
python preprocess/process_image_course.py --backbone vit_h_14 --clip_checkpoint /path/to/ViT-H-14/...
python main_eeg_course.py --blur_prefix MultiBlur_ViTH14 --evnet_prefix EVNet_ViTH14 ...
```

---

## Task 2: EEG-to-Image Reconstruction

### Method Overview

The reconstruction pipeline is adapted from CognitionCapturerPro. It trains a multi-modal EEG encoder to produce CLIP embeddings, then uses a diffusion prior and SDXL-Turbo + IP-Adapter to generate images from those embeddings.

**Pipeline:**

```
EEG → EEGProjectLayer → CLIP embedding
  ├─ [Retrieval] cosine similarity against image/text/depth/edge CLIP embeddings
  └─ [Reconstruction] SimpleAlignPipe (MLP diffusion prior) → CLIP image embedding
                       └─ SDXL-Turbo + IP-Adapter → Generated image
```

**Key stages:**

1. **EEG encoder training** (80 epochs): `EEGProjectLayer` maps EEG [63 ch × 250 t] → 1024-dim CLIP space. It is supervised by contrastive loss against four modality embeddings (image, text caption, Fovea-blurred image, edge map) simultaneously. Uncertainty-aware modality masking is used to prevent the model from memorising a single modality.

2. **Alignment** (100 epochs): `SimpleAlignPipe` (lightweight MLP) maps the EEG CLIP embedding into the CLIP image embedding sub-space, using a frozen IP-Adapter image encoder as the target. This removes the distribution gap between EEG-derived and image-derived CLIP embeddings.

3. **Image generation**: the aligned CLIP embedding is passed to SDXL-Turbo as the IP-Adapter conditioning signal. No text prompt is used. Output resolution: 512 × 512.

### Architecture

```
EEG [B, 63, 250]
  └─ EEGProjectLayer
       ├─ Linear(15750→1024) + GELU + Linear(1024→1024) + Dropout(0.3) + LayerNorm
       └─ Projected into shared CLIP space via contrastive loss

Multi-modal supervision (four parallel encoders, all frozen):
  image    → CLIP ViT-H-14 → z_image  [1024]
  text     → CLIP ViT-H-14 → z_text   [1024]
  depth    → CLIP ViT-H-14 → z_depth  [1024]  (FoveaBlur augmented)
  edge     → CLIP ViT-H-14 → z_edge   [1024]

SimpleAlignPipe: MLP(1024→1024) supervised by IP-Adapter image encoder output

Generation: SDXL-Turbo + IP-Adapter (IP-Adapter-Plus-Face variant)
```

### Configuration

`local.example.yaml` is a template committed to the repository. Copy it to `local.yaml` (which is gitignored) and fill in the paths to your pretrained model weights:

```bash
cp task2/configs/local.example.yaml task2/configs/local.yaml
# Edit local.yaml — set weights_root, sdxl_root, ip_adapter_root
```

**`data_root` is optional** — if `image-eeg-data/` is placed in the `DL_Project/` root (the default layout), the code auto-detects `image-eeg-data/converted_for_cogcappro/` and no extra data preparation step is needed. Only set `data_root` explicitly if your dataset is in a non-standard location.

### Step-by-Step: Running Task 2

**Step 0 — Validate environment** (no GPU needed):

```bash
cd task2 && python smoke_test.py && cd ..
```

> All `sbatch` commands below must be submitted from the **repository root** (the directory containing `task1/` and `task2/`), not from inside `task2/`.

**Step 1 — Prepare diffusion embeddings** (run once):

```bash
sbatch task2/slurm_scripts/02b_reprepare_diffusion_embeddings.sh
# or directly:
python task2/scripts/prepare_diffusion_embeddings.py
```

**Step 2 — Train the EEG encoder** (80 epochs, ~4 h on A40):

```bash
python task2/main.py \
    --config task2/configs/cogcappro.yaml \
    --subjects sub-01 \
    --epoch 80 \
    --lr 1e-4 \
    --staged_training \
    --vision_backbone ViT-H-14 \
    --devices 0
```

Or via SLURM:

```bash
sbatch task2/slurm_scripts/07b_train_retrieval_full_v2.sh
```

**Step 3 — SimpleAlignPipe training** (100 epochs):

```bash
sbatch task2/slurm_scripts/08d_simple_align.sh
```

**Step 4 — Generate reconstructed images:**

```bash
sbatch task2/slurm_scripts/09d_generate_fixed.sh
```

**Step 5 — Evaluate:**

```bash
sbatch task2/slurm_scripts/10e_eval_full_both.sh
python task2/scripts/summarize_results.py
```

#### Multi-seed Run (5 seeds, recommended for reliable results)

Each stage is a SLURM array job (`--array=0-4`) so all 5 seeds run in parallel. Submit stages sequentially using job dependencies:

```bash
# Step 1: train seeds 0–4 in parallel (~24 h each)
JID_TRAIN=$(sbatch --parsable task2/slurm_scripts/06_multiseed_train.sh)

# Step 2: align seeds 0–4 in parallel, start after all training jobs finish
JID_ALIGN=$(sbatch --parsable --dependency=afterok:${JID_TRAIN} task2/slurm_scripts/07_multiseed_align.sh)

# Step 3: generate images for all seeds in one job
JID_GEN=$(sbatch --parsable --dependency=afterok:${JID_ALIGN} task2/slurm_scripts/08_multiseed_generate.sh)

# Step 4: evaluate seeds 0–4 in parallel
JID_EVAL=$(sbatch --parsable --dependency=afterok:${JID_GEN} task2/slurm_scripts/09_multiseed_eval.sh)

# Step 5: summarise — prints mean ± std across all seeds
sbatch --dependency=afterok:${JID_EVAL} task2/slurm_scripts/10_multiseed_summary.sh
```

Results are written to `task2/runs/multiseed/summary.json`.

**Key CLI arguments for `main.py`:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/cogcappro.yaml` | Config file path |
| `--subjects` | `sub-08` | Subject ID (use `sub-01` for course data) |
| `--epoch` | `80` | Max training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--staged_training` | off | 3-stage training (20+40+20 epochs) |
| `--vision_backbone` | `RN50` | CLIP backbone (`ViT-H-14` recommended) |
| `--uncertainty_aware` | off | Enable uncertainty-aware modality masking |
| `--devices` | `0,1` | GPU device IDs |

### Task 2 Results

Single run (sub-01, seed 0). Results from the SimpleAlignPipe + SDXL-Turbo generation (`all` mode).

#### Effect of SimpleAlignPipe (Ablation)

| Metric | Without SimpleAlignPipe | **With SimpleAlignPipe** | Δ |
|--------|------------------------|--------------------------|---|
| **SSIM** | 0.3106 | **0.3732** | +0.063 |
| **CLIP Score (ViT-H-14)** | 0.7160 | **0.8981** | +0.182 |
| PixCorr | 0.131 | 0.159 | +0.028 |
| AlexNet-2 | 0.662 | 0.782 | +0.120 |
| AlexNet-5 | 0.690 | 0.889 | +0.199 |
| Inception | 0.621 | 0.810 | +0.189 |
| EfficientNet | 0.941 | 0.835 | −0.106 |
| SwAV | 0.695 | 0.533 | −0.162 |

SimpleAlignPipe closes the distribution gap between EEG-derived CLIP embeddings and image-space CLIP embeddings. It substantially improves semantic metrics (CLIP, Inception, AlexNet), while EfficientNet and SwAV — which capture low-level texture and self-supervised features — decline slightly, consistent with the alignment shifting the generation toward semantic content over pixel-level fidelity.

**Retrieval (any-modality fusion, as auxiliary output):** Top-1 61.5%, Top-5 89.0%

---

## External Resources

The following pretrained models and open-source codebases are used:

| Resource | Purpose |
|----------|---------|
| OpenCLIP RN50 (OpenAI) | Task 1 image feature extraction |
| OpenCLIP ViT-H-14 (LAION-2B) | Task 2 multi-modal supervision |
| EVNet (Ponce et al., 2023) | Task 1 biologically-inspired frontend |
| SDXL-Turbo (Stability AI) | Task 2 image generation |
| IP-Adapter (Ye et al., 2023) | Task 2 image conditioning |
| VisualEEGDecoding (Liu et al.) | Task 1 multi-blur retrieval approach |
| CognitionCapturerPro | Task 2 multi-modal EEG→image pipeline |

---

## Limitations

**Task 1:**
- Single subject (sub-01); cross-subject generalisation not evaluated.
- EVNet adapter is frozen at random initialisation; an end-to-end learned adapter could improve performance.
- ViT-based CLIP encoders are incompatible with EVNet's spatial preprocessing.

**Task 2:**
- Single subject (sub-01). Multi-seed runs (5 seeds) are supported via `task2/slurm_scripts/06–10_multiseed_*.sh` — see the multi-seed section above.
- Reconstruction is conditioned on a retrieved training image (via IP-Adapter), not on a direct EEG-to-image decoding. Semantically nearby but structurally different training images may produce off-target reconstructions.
- SDXL-Turbo generation (1–4 denoising steps) trades sample quality for speed.
- No text prompt is used; adding a class-level text prompt could improve semantic fidelity.
