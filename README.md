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

**Required external models:**

| Model | Size | Purpose |
|-------|------|---------|
| OpenCLIP RN50 (`open_clip_pytorch_model.bin`) | ~350 MB | Task 1 blur and EVNet image encoder |
| OpenCLIP ViT-H-14 LAION-2B (`open_clip_pytorch_model.bin`) | ~2.5 GB | Task 2 multi-modal supervision encoder |
| SDXL-Turbo | ~6 GB | Task 2 image generation backbone |
| IP-Adapter (SDXL ViT-H variant + image encoder) | ~300 MB | Task 2 image conditioning |

**Option A — One-click download script (recommended):**

```bash
# Interactive (on a node with internet access):
python scripts/download_models.py

# Slurm CPU job (can run in parallel with 00_setup_env.sh):
sbatch task1/slurm_scripts/00b_download_models.sh

# Custom destination or with HuggingFace token:
python scripts/download_models.py --dest /path/to/models --hf-token hf_xxxx
```

Downloads all four models to `DL_Project/models/` and auto-updates `weights_root` in `task2/configs/local.yaml`.

**Option B — Manual download:**

Download each model from HuggingFace and place them under any directory (`<weights_root>`):

```
<weights_root>/
├── CLIP-ViT-H-14-laion2B-s32B-b79K/   # laion/CLIP-ViT-H-14-laion2B-s32B-b79K
│   └── open_clip_pytorch_model.bin
├── CLIP-RN50-openai/                    # laion/CLIP-RN50-openai
│   └── open_clip_pytorch_model.bin
├── sdxl-turbo/                          # stabilityai/sdxl-turbo
│   ├── model_index.json
│   ├── unet/, vae/, text_encoder*/...
└── IP-Adapter/                          # h94/IP-Adapter (sdxl subset only)
    ├── models/image_encoder/
    └── sdxl_models/ip-adapter_sdxl_vit-h.safetensors
```

Then set `weights_root: <weights_root>` in `task2/configs/local.yaml` (see [Configuration](#configuration) below).

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
│   │   ├── evaluate_course_metrics.py
│   │   └── make_greybg_images.py   # Grey-background image generation for ablations
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
img_emb   = fusion_adapter(fused_img)    # MLP 1024→768→1024
```

In the reported Task 1 runs, the EEG encoder maps raw signals to the same 1024-dimensional space, and the model is trained with **InfoNCE (symmetric contrastive loss)**.

### Architecture

```
EEG input [B, 63 ch, 250 t]
  └─ Conv2dWithAbs (63→25 filters, spatial)
  └─ BatchNorm2d
  └─ Linear(250→200) + ELU + Dropout(0.25)
  └─ Linear(200→200) + ELU + Dropout(0.65)
  └─ Linear(25×200→1024)
  └─ EEG embedding [B, 1024]

Image input (offline, pre-computed)
  ├─ Multi-blur: CLIP RN50 × 8 levels → [B, 8, 1024]
  │    └─ Attention aggregation → blur_agg [B, 1024]
  └─ EVNet: SubcorticalBlock → VOneBlock → Conv2d adapter → CLIP RN50 → [B, 1024]
       └─ evnet_feat [B, 1024]

Fusion: softmax([w0, w1]) · [blur_agg, evnet_feat] → MLP → img_emb [B, 1024]

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
| `--blur_config` | `8` | Blur level preset: `1` (no blur), `8`, or `12` |
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

**SLURM (HPC):** pre-configured scripts are in `task1/slurm_scripts/`. All paths are read automatically from `task2/configs/local.yaml` — set `weights_root` there once and no other configuration is needed. `EEG_DATA_DIR` is auto-detected from `image-eeg-data/` in the repo root.

```bash
# No exports needed — just submit from the repository root:
sbatch task1/slurm_scripts/01_gen_evnet_features.sh       # generate features (once)
sbatch task1/slurm_scripts/04_full_train_8blur_evnet.sh   # full train, best result
```

**Optional ablation jobs:** the repository also includes `05a_ablation_train.sh`, `05b_greybg_features.sh`, and `05c_greybg_train.sh` for the `EVNet / blur / grey background` ablation set.

### Task 1 Results

All experiments: 10 random seeds (seeds 21–30), 200 epochs, batch size 1024, lr 0.001, single subject (sub-01), 200-way retrieval.

**Val-selected**: checkpoint selected by best validation Top-1 (from 5% held-out split).
**Final epoch**: test performance at the last training epoch.
**Best-test**: highest test Top-1 observed across all epochs; kept as a diagnostic reference only.

**Primary sources:**
- Report-aligned full-train run: `task1/output/logs/8blur_evnet_full/Brain_Visual_Encoder_EEG/full_8blur_EVNet_2026-05-05-20-57/all_metrics.csv`
- Version7 comparison and backbone summary: `task1/output/task1_version7_evnet_results_summary.md`
- Blur / EVNet / grey-background ablations: `report/git_overleaf/temp/task1_ablation_summary_yliu674.md`

#### Main Experiments

| Setting | Val-sel Top-1 | Val-sel Top-5 | Final Epoch Top-1 | Final Epoch Top-5 | Best-test Top-1 | Best-test Top-5 |
|---|---|---|---|---|---|---|
| 8-blur + EVNet, 95/5 split | 0.8460 ± 0.0128 | 0.9870 ± 0.0056 | 0.8480 ± 0.0173 | 0.9890 ± 0.0030 | 0.8715 ± 0.0087 | 0.9860 ± 0.0077 |
| 12-blur + EVNet, 95/5 split | 0.8400 ± 0.0176 | 0.9860 ± 0.0044 | 0.8520 ± 0.0150 | 0.9850 ± 0.0045 | 0.8715 ± 0.0105 | 0.9855 ± 0.0027 |
| **8-blur + EVNet, full train (report-aligned run)** | N/A | N/A | **0.8630 ± 0.0200** | **0.9855 ± 0.0042** | **0.8935 ± 0.0095** | **0.9880 ± 0.0040** |
| 12-blur + EVNet, full train (version7) | N/A | N/A | 0.8505 ± 0.0160 | 0.9845 ± 0.0035 | 0.8810 ± 0.0070 | 0.9850 ± 0.0039 |

The primary reported Task 1 result, matching the report, is the **full-train 8-blur + EVNet final-epoch metric**: Top-1 `86.30% ± 2.00%`, Top-5 `98.55% ± 0.42%`.
Best-test numbers are retained for diagnostic comparison only and are not used as the main reported metric.

#### Ablation Studies

The table below summarizes the source-verified Task 1 ablation results. All settings use 10 seeds, the RN50 backbone, and the 95/5 split protocol unless noted otherwise.

| Setting | Val-sel Top-1 | Val-sel Top-5 | Final Epoch Top-1 | Final Epoch Top-5 | Best-test Top-1 | Best-test Top-5 |
|---|---|---|---|---|---|---|
| 12-blur | 0.8240 ± 0.0191 | 0.9780 ± 0.0051 | 0.8455 ± 0.0123 | 0.9765 ± 0.0067 | 0.8685 ± 0.0059 | 0.9810 ± 0.0049 |
| 12-blur + EVNet | 0.8325 ± 0.0237 | 0.9825 ± 0.0040 | 0.8525 ± 0.0136 | 0.9810 ± 0.0037 | 0.8825 ± 0.0051 | 0.9820 ± 0.0060 |
| 8-blur + EVNet | 0.8360 ± 0.0214 | 0.9820 ± 0.0046 | 0.8535 ± 0.0169 | 0.9795 ± 0.0047 | 0.8815 ± 0.0092 | 0.9825 ± 0.0056 |
| EVNet with no blur | 0.7340 ± 0.0214 | 0.9565 ± 0.0090 | 0.7350 ± 0.0204 | 0.9575 ± 0.0075 | 0.7785 ± 0.0150 | 0.9660 ± 0.0080 |
| No blur and no EVNet | 0.6120 ± 0.0235 | 0.9060 ± 0.0176 | 0.6430 ± 0.0183 | 0.9100 ± 0.0112 | 0.6705 ± 0.0101 | 0.9110 ± 0.0170 |
| No blur and no EVNet, grey background | 0.6950 ± 0.0226 | 0.9205 ± 0.0079 | 0.6960 ± 0.0203 | 0.9185 ± 0.0063 | 0.7380 ± 0.0075 | 0.9230 ± 0.0114 |
| EVNet with no blur, grey background | 0.7765 ± 0.0148 | 0.9590 ± 0.0092 | 0.7695 ± 0.0154 | 0.9555 ± 0.0088 | 0.8185 ± 0.0081 | 0.9630 ± 0.0078 |
| 8-blur + EVNet, grey background | 0.8225 ± 0.0155 | 0.9750 ± 0.0067 | 0.8315 ± 0.0145 | 0.9710 ± 0.0073 | 0.8595 ± 0.0069 | 0.9735 ± 0.0055 |
| 8-blur, grey background | 0.8105 ± 0.0106 | 0.9795 ± 0.0061 | 0.8165 ± 0.0114 | 0.9815 ± 0.0045 | 0.8415 ± 0.0125 | 0.9805 ± 0.0072 |

**Findings:**
- **EVNet consistently helps**: at matched blur settings, EVNet improves both val-selected and best-test Top-1 over the corresponding non-EVNet baseline.
- **12-blur and 8-blur are close once EVNet is enabled**: `12-blur + EVNet` and `8-blur + EVNet` remain close across both final-epoch and best-test Top-1.
- **Grey background helps most in the no-blur regime**: compared with plain no-blur, grey background raises best-test Top-1 from `0.6705` to `0.7380` without EVNet, and from `0.7785` to `0.8185` with EVNet.

#### Backbone / EVNet Initialisation Ablation

These rows correspond to the backbone ablation table used in the report.

| Setting | Val-sel Top-1 | Final Epoch Top-1 | Best-test Top-1 |
|---|---|---|---|
| RN50 + EVNet, Kaiming init | 0.8460 ± 0.0128 | 0.8480 ± 0.0173 | 0.8715 ± 0.0087 |
| RN50 + EVNet, Xavier init | 0.8275 ± 0.0166 | 0.8190 ± 0.0130 | 0.8495 ± 0.0082 |
| RN50 + GAP (no CLIP backbone) | 0.8285 ± 0.0164 | 0.8315 ± 0.0160 | 0.8620 ± 0.0087 |
| ViT-H/14 + EVNet | 0.7365 ± 0.0198 | 0.7360 ± 0.0197 | 0.7790 ± 0.0109 |

---

## Task 2: EEG-to-Image Reconstruction

### Method Overview

The reconstruction pipeline is adapted from CognitionCapturerPro. It trains a multi-modal EEG encoder to produce CLIP embeddings, then uses a diffusion prior and SDXL-Turbo + IP-Adapter to generate images from those embeddings.

**Pipeline:**

```
EEG → EEGProjectLayer → CLIP embedding
  ├─ [Retrieval] cosine similarity against image/text/depth/edge CLIP embeddings
  └─ [Reconstruction] SimpleAlignPipe → aligned image/depth/edge embeddings
                       └─ SDXL-Turbo + IP-Adapter → Generated image
```

**Key stages:**

1. **EEG encoder training** (80 epochs): `EEGProjectLayer` maps EEG [63 ch × 250 t] → 1024-dim CLIP space. It is supervised by contrastive loss against four modality embeddings (image, text, depth map, edge map) simultaneously. Uncertainty-aware modality masking is used to prevent the model from memorising a single modality.

2. **Alignment** (100 epochs): `SimpleAlignPipe` (lightweight MLP) aligns the EEG-derived image, depth, and edge embeddings to the embedding distributions expected by the generation pipeline.

3. **Image generation**: the aligned image, depth, and edge embeddings are passed to SDXL-Turbo through IP-Adapter as conditioning signals. No text prompt is used. Output resolution: 512 × 512.

### Architecture

```
EEG [B, 63, 250]
  └─ EEGProjectLayer
       ├─ Linear(15750→1024) + GELU + Linear(1024→1024) + Dropout(0.3) + LayerNorm
       └─ Projected into shared CLIP space via contrastive loss

Multi-modal supervision (four parallel encoders, all frozen):
  image    → CLIP ViT-H-14 → z_image  [1024]
  text     → CLIP ViT-H-14 → z_text   [1024]
  depth    → CLIP ViT-H-14 → z_depth  [1024]
  edge     → CLIP ViT-H-14 → z_edge   [1024]

SimpleAlignPipe: modality-wise alignment into the IP-Adapter conditioning space

Generation: SDXL-Turbo + IP-Adapter (SDXL ViT-H variant)
```

### Configuration

`local.example.yaml` is a template committed to the repository. It needs to be copied to `local.yaml` (gitignored, local-only) and have `weights_root` filled in.

**If you used Option A (download script):** `local.yaml` is created and updated automatically — no manual steps needed.

**If you used Option B (manual download):** copy the template and edit it yourself:

```bash
cp task2/configs/local.example.yaml task2/configs/local.yaml
# Edit task2/configs/local.yaml — set weights_root to your model directory
```

**Only one line needs editing:** `weights_root: /path/to/model_weights`. All other paths (`clip_weights_rel`, `sdxl_rel`, `ip_adapter_rel`) are expressed relative to `weights_root` and work out of the box if your weights follow the default directory names. Task 1 slurm scripts also read `weights_root` from this file automatically.

**`eeg_data_dir` is optional** — if `image-eeg-data/` is placed in the `DL_Project/` root (the default layout), both tasks auto-detect the data and no extra configuration is needed. Set `eeg_data_dir` only if your data is in a non-standard location:

```yaml
# task2/configs/local.yaml
paths:
  weights_root: /path/to/model_weights
  eeg_data_dir: /path/to/image-eeg-data   # only if NOT in DL_Project root
```

Setting `eeg_data_dir` covers both tasks: Task 1 uses it directly as the EEG data directory, and Task 2 derives `converted_for_cogcappro/` from it automatically.

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
    --vision_backbone ViT-H-14 \
    --devices 0
```

Add `--staged_training` to enable the optional 3-stage curriculum. The reported multi-seed results use the multiseed scripts and joint training by default.

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

Reported numbers are mean ± std across 5 seeds (sub-01, seeds 0-4). Results compare direct EEG-conditioned generation (`all_before`) against SimpleAlignPipe + SDXL-Turbo generation (`all` mode).
Source: `task2/runs/multiseed/summary.json`.

#### Effect of SimpleAlignPipe (Ablation)

| Metric | Without SimpleAlignPipe | **With SimpleAlignPipe** | Δ |
|--------|------------------------|--------------------------|---|
| **SSIM** | 0.2997 ± 0.0154 | **0.3564 ± 0.0083** | +0.057 |
| **CLIP Score (ViT-H-14)** | 0.6940 ± 0.0261 | **0.8927 ± 0.0067** | +0.199 |
| PixCorr | 0.1393 ± 0.0089 | 0.1477 ± 0.0157 | +0.008 |
| AlexNet-2 two-way ID | 0.6574 ± 0.0080 | 0.7621 ± 0.0133 | +0.105 |
| AlexNet-5 two-way ID | 0.6982 ± 0.0182 | 0.8826 ± 0.0100 | +0.184 |
| Inception two-way ID | 0.5982 ± 0.0100 | 0.8169 ± 0.0087 | +0.219 |
| EfficientNet corr. dist. | 0.9517 ± 0.0065 | **0.8284 ± 0.0047** | −0.123 |
| SwAV corr. dist. | 0.7060 ± 0.0089 | **0.5318 ± 0.0027** | −0.174 |

SimpleAlignPipe closes the distribution gap between EEG-derived CLIP embeddings and image-space CLIP embeddings.
It substantially improves semantic metrics (CLIP, Inception, AlexNet) and also reduces EfficientNet and SwAV correlation distance, where lower values are better.

**Retrieval (any-modality fusion, 5 seeds, auxiliary output):** Top-1 0.6370 ± 0.0258, Top-5 0.8730 ± 0.0125

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
