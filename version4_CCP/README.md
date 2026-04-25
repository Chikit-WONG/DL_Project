# CognitionCapturerPro — EEG Brain-to-Image Retrieval & Reconstruction

[中文版 README](./README-CN.md)

This repository is the **CognitionCapturerPro** branch of the DSAA2012 final project (Prof. Chen Liang).  
It reproduces and adapts the [CognitionCapturerPro paper](https://arxiv.org/abs/2401.07935) on the course Things-EEG dataset for a single subject (sub-01).

The pipeline decodes EEG brain signals evoked by natural images into:
1. **Retrieval** — given an EEG signal, rank 200 candidate images and return the most likely one.
2. **Reconstruction** — generate a photorealistic image conditioned on the EEG signal using SDXL-Turbo + IP-Adapter.

---

## Pipeline Overview

```
EEG signal
    ↓
EEG Encoder (EEGProjectLayer, 80 epochs)
    ↓
CLIP-compatible embedding (ViT-H-14 space)
    ↓
┌─────────────────────────────────┐
│ Retrieval (cosine similarity)   │ → Top-1 / Top-5 accuracy
└─────────────────────────────────┘
    ↓ (optional alignment step)
SimpleAlignPipe MLP (100 epochs)
    ↓
┌─────────────────────────────────┐
│ Image Generation (SDXL-Turbo   │ → Reconstructed image
│ + IP-Adapter ViT-H-14)         │
└─────────────────────────────────┘
```

---

## Environment Setup

### Prerequisites

- Python 3.10
- CUDA 12.6
- Conda (recommended)

### Installation

```bash
conda create -n cogcap python=3.10 -y
conda activate cogcap
pip install -r requirements.txt
```

### Configure Paths

Copy the example config and fill in your local paths:

```bash
cp configs/local.example.yaml configs/local.yaml
```

Edit `configs/local.yaml`:

```yaml
paths:
  data_root: /path/to/image-eeg-data/converted_for_cogcappro   # Things-EEG course dataset
  weights_root: /path/to/models                                  # CLIP, SDXL-Turbo, IP-Adapter
  runs_root: /path/to/version4_CCP/runs                         # Training outputs
  sdxl_root: /path/to/models/sdxl-turbo
  ip_adapter_root: /path/to/models/IP-Adapter
  clip_weights_rel:
    ViT-H-14: CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin
```

### Verify Installation

```bash
python smoke_test.py
```

All 13 checks should pass (no GPU required for the smoke test).

---

## Reproducing Results

### Step 1 — Prepare Diffusion Embeddings

```bash
sbatch slurm_scripts/02b_reprepare_diffusion_embeddings.sh
```

### Step 2 — Train EEG Retrieval Model (80 epochs)

```bash
sbatch slurm_scripts/07b_train_retrieval_full_v2.sh
```

### Step 3 — Train Alignment (SimpleAlignPipe, 100 epochs)

```bash
sbatch slurm_scripts/08d_simple_align.sh
```

### Step 4 — Generate Images

```bash
sbatch slurm_scripts/09d_generate_fixed.sh   # both all_before and all modes
```

### Step 5 — Evaluate

```bash
sbatch slurm_scripts/10e_eval_full_both.sh
sbatch slurm_scripts/11b_summary_v2.sh
```

---

## Model Scores (sub-01, seed 0)

### Retrieval (200-way, any-modality fusion)

| Metric | Ours | Paper (10-subject avg) |
|--------|------|------------------------|
| **Top-1** | **61.0%** | 61.2% |
| **Top-5** | **88.0%** | 90.8% |

### Reconstruction

Two generation modes are provided:

| Metric | `all_before` (EEG → IP-Adapter, **best**) | `all` (SimpleAlignPipe) | Paper |
|--------|------------------------------------------|--------------------------|-------|
| CLIP (↑) | **0.707** | 0.659 | 0.830 |
| PixCorr (↑) | 0.130 | **0.133** | 0.163 |
| SSIM (↑) | **0.316** | 0.236 | 0.398 |
| AlexNet-2 (↑) | **0.663** | 0.618 | 0.831 |
| AlexNet-5 (↑) | **0.698** | 0.682 | 0.937 |
| Inception (↑) | 0.597 | **0.607** | 0.720 |

**`all_before` is the recommended mode**: it feeds EEG CLIP embeddings directly into IP-Adapter without an intermediate alignment step, achieving better overall quality.

---

## Outputs

| Path | Contents |
|------|----------|
| `outputs/generated_all_before/` | 200 generated images (direct EEG → IP-Adapter, best quality) |
| `outputs/generated_all/` | 200 generated images (SimpleAlignPipe alignment) |
| `outputs/comparison/` | 3-way comparison grids (Ground Truth \| all_before \| all) |
| `outputs/comparison/grid_all200.png` | Overview grid of all 200 triplets |
| `outputs/comparison/comparison_page01-10.png` | Page-by-page comparison (20 triplets each) |
| `outputs/comparison/single/` | 200 individual 3-panel comparison images |
| `results/reconstruction_metrics_all_before.json` | Reconstruction metrics for `all_before` |
| `results/reconstruction_metrics_all.json` | Reconstruction metrics for `all` |
| `results/retrieval_test_results.json` | Retrieval accuracy and per-modality breakdown |
| `results/summary_metrics.json` | Aggregated summary |

---

## Bugs Fixed

Six bugs were identified and fixed compared to the original repository:

1. **Embedding key collision** (`generator.py`): images sharing the same filename across different class directories overwrote each other's diffusion embeddings. Fixed by keying with `class/filename`.
2. **Insufficient training epochs**: retrieval used 10 epochs instead of the config-specified 80; alignment used 1 epoch. Fixed in `07b`/`08b` scripts.
3. **Uncertainty-aware masking bypassed** (`align/data.py`): hardcoded `DirectT` override removed; original UM module restored.
4. **VAE float16 NaN** (`generator.py`): `force_upcast=False` caused VAE to produce NaN → all-black images. Fixed with `vae.config.force_upcast = True`.
5. **IP-Adapter embedding shape** (`generator.py`): with `guidance_scale=0.0`, stacked `[2,1,1024]` instead of `[1,1,1024]`. Fixed with `embed.unsqueeze(0)` when `do_cfg=False`.
6. **PyTorch 2.6 `weights_only`** (`align/main.py`): default changed to `True`; added `weights_only=False` for custom dataset class.

---

## Limitations

- **Single subject**: the paper reports 10-subject averages. Using 1 subject produces less reliable EEG embeddings (irreducible gap ~10–20%).
- **EEG noise**: raw EEG signals are very noisy; even after 80 epochs the embedding quality is limited by signal SNR.
- **Semantic gap**: CLIP-based retrieval/generation captures high-level category correctly (61% top-1) but cannot encode fine-grained visual details (viewpoint, texture, lighting) from EEG alone.
- **Alignment stage introduces noise**: the EEG retrieval model already outputs CLIP-space embeddings that IP-Adapter expects. Mapping to a separate "diffusion embedding space" introduces distortion rather than improvement (CLIP drops 0.707 → 0.659).

---

## Key Files

| File | Role |
|------|------|
| `src/cogcappro/models/brain_backbone.py` | EEG encoder (EEGProjectLayer) |
| `src/cogcappro/models/fusion_backbone.py` | Multi-modal fusion backbone |
| `src/cogcappro/training/module.py` | PyTorch Lightning training module |
| `src/cogcappro/align/diffusion_pipe.py` | SimpleAlignPipe + DiffusionPriorUNet |
| `src/cogcappro/generate_image/generator.py` | SDXL-Turbo + IP-Adapter generation |
| `configs/cogcappro.yaml` | Main model/training config |
| `configs/local.yaml` | Local paths (not committed) |
| `plan/Reproduce_CognitionCapturerPro_Fix_Plan_en.md` | Full bug-fix and tuning notes (English) |
| `plan/Reproduce_CognitionCapturerPro_Fix_Plan_zh.md` | Full bug-fix and tuning notes (Chinese) |
