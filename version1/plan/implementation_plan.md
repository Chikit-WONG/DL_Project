# Project 1: Brain-to-Image Retrieval & Reconstruction - Implementation Plan

## Context

DSAA2012 course project (Project A). Team of 3 must build an EEG-to-image system with two mandatory tasks:

1. **Retrieval** - rank correct image among 200 candidates
2. **Reconstruction** - generate image from EEG

**Timeline:** Presentation April 28, report due May 10.

**Grading:** 25pts retrieval + 25pts reconstruction + 20pts methodology + 20pts report + 10pts code quality. Both tasks mandatory (incomplete = max 60/100). Must be reproducible (non-reproducible = 0).

**Dataset:** 63-channel EEG, 250 time steps. Training ~1654 categories (80 trials each), test 200 categories (4 trials each). Test must use `avg_trials=True`. This is a very small dataset -- the core challenge.

---

## Overall Approach: CLIP-Aligned EEG Encoder

Train an EEG encoder to project EEG signals into CLIP ViT-L/14 embedding space (768-dim). This shared encoder serves both tasks:

- **Retrieval:** cosine similarity between EEG embedding and CLIP image embeddings
- **Reconstruction:** feed EEG-CLIP embedding into IP-Adapter + Stable Diffusion

```
EEG [B, 63, 250]
      |
  EEG Encoder (CNN + Transformer)
      |
  EEG Embedding [B, 768] (in CLIP space)
      |
      +---> Task 1: Retrieval (cosine sim with CLIP image features)
      |
      +---> Task 2: Reconstruction (IP-Adapter + Stable Diffusion)
```

---

## Step 1: Cache CLIP Image Features

**File:** `src/clip_features.py`

- Load pretrained CLIP ViT-L/14
- Extract features for all training images (map `image_id` -> 768-dim vector)
- Extract features for all 200 test images
- Save to `clip_features_train.pt` and `clip_features_test.pt`
- Build `image_id` -> image path mapping by globbing `training_images/` and `test_images/`

**Run once on HPC, then reuse everywhere.**

---

## Step 2: Data Pipeline

**File:** `src/data.py`

- Wrap `load_eeg_dataset()` (from sample code) into PyTorch DataLoader
- Support both `avg_trials=True` (1654 clean samples) and `avg_trials=False` (~132K noisy samples)
- **Data augmentation** (critical for small dataset):
  - Temporal jitter: shift +/- 5 time steps
  - Channel dropout: zero out 5-10% channels randomly
  - Gaussian noise: std=0.01-0.05
  - Time masking: mask contiguous block of 10-25 steps
  - Amplitude scaling: factor in [0.8, 1.2]

---

## Step 3: EEG Encoder Architecture

**File:** `src/eeg_encoder.py`

**Architecture: CNN + Transformer hybrid (~2-3M params)**

```
Input [B, 63, 250]
  -> Spatial Conv: Conv1d(63, 128, k=1) x2 with BN+GELU  (channel mixing)
  -> Temporal Conv: Conv1d(128->256, k=15, stride=2) x3 with BN+GELU+Dropout
     (250 -> 125 -> 63 -> 32 time steps)
  -> Transformer: 3 layers, 4 heads, d=256, FFN=512, dropout=0.1
     (32 tokens of dim 256)
  -> Global average pooling -> [B, 256]
  -> Projection MLP: Linear(256,512) + GELU + Linear(512,768)
  -> L2 normalize -> [B, 768]
```

Output dimension = 768 to match CLIP ViT-L/14.

---

## Step 4: Train EEG Encoder (Contrastive Learning)

**File:** `src/train_encoder.py`

**Loss:** CLIP-style symmetric InfoNCE (contrastive) loss with learnable temperature.

```python
def clip_loss(eeg_embs, img_embs, temperature):
    logits = (eeg_embs @ img_embs.T) / temperature  # [B, B]
    labels = torch.arange(B, device=logits.device)
    loss_eeg = F.cross_entropy(logits, labels)
    loss_img = F.cross_entropy(logits.T, labels)
    return (loss_eeg + loss_img) / 2
```

**Two-phase training:**

| | Phase 1 (coarse) | Phase 2 (fine-tune) |
|---|---|---|
| Data | avg_trials=False (~132K) | avg_trials=True (1654) |
| Batch size | 128 | 64 |
| LR | 3e-4, cosine decay | 5e-5, cosine decay |
| Epochs | 30-50 | 100-200 |
| Optimizer | AdamW, wd=0.05 | AdamW, wd=0.05 |
| Temperature | Learnable, init=0.07 | Continue from Phase 1 |
| Augmentation | All 5 types | Jitter + noise only |

---

## Step 5: Retrieval Evaluation (Task 1)

**File:** `src/retrieval.py`

1. Encode all 200 test EEG -> 768-dim embeddings (L2-normalized)
2. Load cached CLIP features for 200 test images (L2-normalized)
3. Similarity matrix = `eeg_embs @ img_embs.T` -> [200, 200]
4. Use provided `compute_retrieval_metrics()` for Top-1 / Top-5
5. Repeat over 10 seeds, report mean +/- std

**Target:** Top-1 ~15-30%, Top-5 ~40-60% (random baseline: 0.5% / 2.5%)

---

## Step 6: Image Reconstruction (Task 2)

**File:** `src/reconstruction.py`

**Approach: EEG -> CLIP embedding -> IP-Adapter + Stable Diffusion**

- Use `stable-diffusion-v1-5` with `h94/IP-Adapter`
- The EEG encoder already produces CLIP-space embeddings
- Feed EEG embedding as image prompt to IP-Adapter
- Generate at 512x512, resize to 256x256 for evaluation

**Key parameters:**
- IP-Adapter scale: ~0.7
- Guidance scale: 7.5
- Inference steps: 50

**Fallback:** If IP-Adapter embedding format is incompatible, train a small MLP projector to convert EEG-CLIP embeddings to IP-Adapter's expected token format.

Evaluate with provided `eval_images()` over 10 diffusion seeds.

---

## Step 7: Evaluation & Report

**File:** `src/evaluate.py`

- Use provided evaluation code verbatim (do not modify)
- Retrieval: Top-1, Top-5 (mean +/- std, 10 seeds)
- Reconstruction: all 8 metrics from `eval_images()` (mean +/- std, 10 seeds)
  - PixCorr, SSIM, AlexNet(2), AlexNet(5), Inception, CLIP, EfficientNet, SwAV
- Generate 8-12 qualitative examples (mix of success and failure cases)

---

## Code Structure

```
Project1/
├── image-eeg-data/                # Given data (DO NOT modify)
│   ├── train.pt
│   ├── test.pt
│   ├── EEG_CHANNELS.jsonl
│   ├── training_images/           # 1654 categories
│   └── test_images/               # 200 categories
├── eeg_project_sample_code.ipynb  # Given sample code
│
├── src/
│   ├── data.py                    # Dataset + augmentation
│   ├── eeg_encoder.py             # EEG encoder model
│   ├── clip_features.py           # CLIP feature caching (run once)
│   ├── train_encoder.py           # Training script
│   ├── retrieval.py               # Retrieval evaluation
│   ├── reconstruction.py          # Image generation pipeline
│   └── utils.py                   # Seeds, metrics (from sample code)
│
├── checkpoints/                   # Model weights
└── outputs/                       # Generated images, metrics
```

---

## Team Division

| Person | Module | Files | Report Sections |
|--------|--------|-------|-----------------|
| A | EEG Encoder | `data.py`, `eeg_encoder.py`, `train_encoder.py` | Data intro, encoder methodology |
| B | Retrieval | `clip_features.py`, `retrieval.py` | Retrieval method, retrieval results |
| C | Reconstruction | `reconstruction.py`, `evaluate.py` | Reconstruction method, reconstruction results |

Shared: `utils.py`, report intro/conclusion

---

## HPC Execution Order

```bash
# 1. Setup environment
pip install torch transformers diffusers datasets clip scikit-image scipy Pillow accelerate peft

# 2. Cache CLIP features (one-time, needs GPU)
python src/clip_features.py

# 3. Train EEG encoder (GPU, main compute)
python src/train_encoder.py --phase 1  # ~132K samples, 30-50 epochs
python src/train_encoder.py --phase 2  # 1654 samples, 100-200 epochs

# 4. Evaluate retrieval
python src/retrieval.py  # 10 seeds, fast

# 5. Generate reconstructed images (GPU, ~17 min for 200 images)
python src/reconstruction.py

# 6. Evaluate reconstruction
python src/evaluate.py  # 10 seeds, needs pretrained models
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Overfitting on 1654 samples | Two-phase training + augmentation + dropout + weight decay |
| IP-Adapter poor quality | Try MLP projector; fallback to simple DCGAN decoder |
| Compute limits | Cache CLIP features; small encoder trains fast |
| 10-seed requirement | Train 1 model, vary diffusion seed for reconstruction; if needed, train 10 models (feasible given small model size) |
| `image_id` path mapping | Glob all image files, build dict from stem to path |
