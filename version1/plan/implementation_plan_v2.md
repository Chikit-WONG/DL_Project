# EEG-to-Image: Unified Architecture Implementation Plan (v2)

## Context

DSAA2012 Project A: Brain-to-Image Retrieval & Reconstruction on THINGS-EEG.

**Why v2 exists**: The previous plans split the work into two separate architectures (A: shared encoder, B: independent encoders) and required two parallel codebases. We realized that **Architecture B is mathematically a special case of Architecture A** — when one of the loss weights (alpha or beta) is set to zero, the joint training collapses into single-task training, equivalent to an independent encoder. This insight lets us write **one** codebase, treat Architecture B as a configuration of Architecture A, and additionally make the loss weights **learnable** so the model can auto-balance the two tasks.

**Goal**: Build a working EEG-to-Image system that scores well on Top-1/Top-5 retrieval (25 pts) and SSIM/CLIP reconstruction (25 pts), with clean methodology (20 pts) and reproducible code (10 pts). One person focuses on the main scoring architecture; two teammates do micro-ablations later.

**Timeline**: Today is 2026-04-09. Presentation 2026-04-28 (~20 days), report 2026-05-10.

---

## 1. Available Models (Verified)

All models are at `/hpc2hdd/home/ckwong627/workdir/models/`. Everything we need is already downloaded:

| Model | Path | Output dim | Used for |
|-------|------|-----------|----------|
| Stable Diffusion v1.5 | `stable-diffusion-v1-5/` | — | Image generation backbone |
| IP-Adapter (full) | `IP-Adapter/models/ip-adapter_sd15.bin` | — | Conditions SD on image embeddings |
| **IP-Adapter image encoder** | `IP-Adapter/models/image_encoder/` | **1024-d** | **CLIP-ViT-H-14, IP-Adapter's native CLIP** |
| LAION CLIP ViT-L/14 | `CLIP-ViT-L-14-laion2B-s32B-b82K/` | 768-d | Alternative (smaller, faster) |
| LAION CLIP ViT-H/14 | `CLIP-ViT-H-14-laion2B-s32B-b79K/` | 1024-d | Same as IP-Adapter's encoder |
| LAION CLIP ViT-B/32 | `CLIP-ViT-B-32-laion2B-s34B-b79K/` | 512-d | Tiny baseline |

**LAION vs OpenAI CLIP**: Using LAION CLIP is **fine and actually better** for this project:
1. The TA `eval_images()` function uses OpenAI ViT-L/14 internally only for the `eval_clip` metric, but this is a black-box comparison between **generated images** and **real images** in pixel space. It does **not** see our model's embeddings.
2. For retrieval (Top-1/Top-5), we provide our own [N, N] similarity matrix. ANY embedding space works as long as we use the same encoder for both EEG-side alignment target and image-side test embeddings.
3. For IP-Adapter, LAION ViT-H/14 is what it **natively expects**. Using OpenAI CLIP would require an extra learned projector and lose information.

**Decision**: Use IP-Adapter's bundled image encoder (LAION ViT-H/14, 1024-d) for the entire pipeline. EEG embeddings are 1024-d, fed directly to IP-Adapter without any projector.

If retrieval underperforms with the 1024-d head, fall back to LAION ViT-L/14 (768-d) and add a 768→1024 projector for IP-Adapter.

---

## 2. Unified Architecture

### Core Math

```
EEG [B, 63, 250]
      |
  EEG Encoder (CNN + Transformer, ~3M params)
      |
  EEG Embedding [B, 1024] (aligned to LAION ViT-H/14 space)
      |
      +---> L_retrieval = symmetric InfoNCE       (cosine ranking)
      |
      +---> L_reconstruction = MSE in CLIP space  (absolute proximity)

Total Loss = alpha * L_retrieval + beta * L_reconstruction
```

### How Architecture B emerges as a special case

| Mode | alpha | beta | Equivalent to |
|------|-------|------|---------------|
| Pure retrieval (Arch B Task 1) | 1.0 | 0.0 | Independent retrieval encoder |
| Pure reconstruction (Arch B Task 2) | 0.0 | 1.0 | Independent reconstruction encoder |
| Joint training (Arch A) | >0 | >0 | Shared encoder with both losses |
| **Learnable** | `exp(log_alpha)` | `exp(log_beta)` | Model auto-balances |

### Why two losses?
- **InfoNCE (retrieval)**: Maximally discriminative ranking — but only enforces correct ordering, not absolute position in CLIP space.
- **MSE (reconstruction)**: Pushes EEG embeddings to occupy the same region as real CLIP embeddings — essential for IP-Adapter, which expects vectors that look like real CLIP outputs.

These losses are complementary. InfoNCE alone could let embeddings drift from the CLIP manifold (still ranking-correct but not generation-friendly). MSE alone collapses to mean (no discriminative signal). Together: discriminative AND well-positioned.

### Why learnable weights?
Using `nn.Parameter` for `log_alpha`, `log_beta` (then `exp()` for positivity) lets the model balance tasks automatically. Early in training, retrieval gradient is stronger; later, reconstruction can take over. This is the homoscedastic uncertainty approach (Kendall et al. 2018), simplified.

### EEG Encoder Architecture

```
Input [B, 63, 250]
  → Spatial: Conv1d(63→128, k=1) + BN + GELU      [B, 128, 250]
            Conv1d(128→128, k=1) + BN + GELU      [B, 128, 250]
  → Temporal: Conv1d(128→192, k=15, s=2) + BN + GELU + Drop  [B, 192, 125]
              Conv1d(192→256, k=15, s=2) + BN + GELU + Drop  [B, 256, 63]
              Conv1d(256→320, k=15, s=2) + BN + GELU + Drop  [B, 320, 32]
  → Transpose to [B, 32, 320] + learnable positional encoding (32 pos)
  → 3× TransformerEncoderLayer(d=320, heads=8, FFN=640, drop=0.1)  [B, 32, 320]
  → Global Average Pool                             [B, 320]
  → MLP: Linear(320→640) + GELU + Drop + Linear(640→1024)
  → L2 normalize                                    [B, 1024]
```

Estimated parameter count: ~3M. Trains fast on a single GPU within 30-min SLURM jobs.

### Reconstruction Pipeline

```
EEG → EEG Encoder → 1024-d embedding (in LAION ViT-H/14 space)
              ↓
       IP-Adapter (h94/ip-adapter_sd15.bin)
              ↓
       Stable Diffusion v1.5 UNet (frozen)
              ↓
       512×512 image → resize to 256×256 for eval
```

No projector needed. The diffusers library accepts pre-computed image embeddings via `ip_adapter_image_embeds=` argument.

---

## 3. File Structure

### Python code: `DL_Project/codes/`

| File | Purpose |
|------|---------|
| `config.py` | Dataclass with all hyperparameters (paths, model dims, loss weights, training, augmentation) |
| `utils.py` | Verbatim from sample code: `set_seed`, `compute_retrieval_metrics`, `summarize_metrics_over_seeds`, `eval_images()` and all sub-functions, plus `build_image_id_to_path()` |
| `data.py` | `load_eeg_dataset()` (verbatim), `EEGImageDataset` (pairs EEG with cached CLIP embeddings via image_id), `EEGAugmentation` (5 types) |
| `model.py` | `EEGEncoder`, `UnifiedModel` with dual loss + fixed/learnable alpha/beta + learnable temperature |
| `cache_clip_features.py` | One-time: extract & cache 1024-d features for all training+test images using IP-Adapter's `image_encoder/`. Saves to `../clip_cache/` |
| `train.py` | Main training: two-phase, checkpointing, validation. CLI args for phase/alpha/beta/learnable_weights/resume |
| `reconstruct.py` | IP-Adapter + SD pipeline: encode 200 test EEGs, generate 200 images per seed × 10 seeds |
| `evaluate.py` | Retrieval (10 seeds) + reconstruction (10 seeds) eval, prints summary, saves JSON |
| `run_all.ipynb` | Final notebook for TA: imports modules, runs full pipeline, displays metrics + qualitative grid |

### SLURM scripts: `DL_Project/slurm_scripts/`

All use: `partition=debug`, `1 GPU`, `conda env=test`, `module load cuda/12.1`, `--time=00:30:00`.

| Script | Est. duration | Runs |
|--------|---------------|------|
| `run_cache_clip.sh` | ~15 min | `cache_clip_features.py` |
| `run_train_phase1.sh` | ~15 min | `train.py --phase 1 --alpha 1.0 --beta 0.5 --epochs 50` |
| `run_train_phase2.sh` | ~25 min | `train.py --phase 2 --resume phase1.pt --alpha 0.5 --beta 1.0 --epochs 100` |
| `run_train_learnable.sh` | ~25 min | `train.py --phase 2 --resume phase1.pt --learnable_weights --epochs 100` |
| `run_train_retrieval_only.sh` | ~15 min | `train.py --phase 1 --alpha 1.0 --beta 0.0` (Arch B retrieval) |
| `run_train_recon_only.sh` | ~15 min | `train.py --phase 1 --alpha 0.0 --beta 1.0` (Arch B recon) |
| `run_reconstruct.sh` | ~20 min | `reconstruct.py --seeds 0..9` |
| `run_evaluate.sh` | ~20 min | `evaluate.py` |

### Output directories (auto-created on first run)
```
DL_Project/
  clip_cache/      # Cached LAION ViT-H/14 features (~70MB)
  checkpoints/     # Model weights, one per experiment config
  outputs/         # Generated images (.pt) + metrics (.json)
  temp/            # SLURM logs
  plan/            # Plan documents (English + Chinese)
```

---

## 4. Training Strategy

### Phase 1: Coarse Training
- **Data**: `avg_trials=True`. Code prints actual N at runtime.
- **Augmentation (full)**: temporal jitter ±5, channel dropout 10%, Gaussian noise std=0.02, time mask 20 steps, amplitude scale [0.8, 1.2]. Each applied independently with p=0.5.
- **Hyperparams**: batch=128, lr=3e-4, AdamW(wd=0.05), cosine decay, 50 epochs
- **Loss**: alpha=1.0, beta=0.5, learnable temperature init=0.07

### Phase 2: Fine-tune
- Resume Phase 1 best checkpoint
- batch=64, lr=5e-5, 100 epochs
- Loss: alpha=0.5, beta=1.0 (shift toward reconstruction)
- Lighter augmentation (jitter + noise only)

### Phase 2 Variant: Learnable Weights
- Same as Phase 2 but `--learnable_weights` flag
- Initialize from Phase 1 alpha/beta values
- Log alpha/beta each epoch to track auto-balancing

### Architecture B Baselines (for ablation)
- `--alpha 1.0 --beta 0.0`: pure retrieval encoder
- `--alpha 0.0 --beta 1.0`: pure reconstruction encoder
- These are the "Architecture B" baselines, run with the SAME code

---

## 5. Key Implementation Details

### Reusing existing functions (DO NOT rewrite)

From `sample_codes/eeg_project_sample_code.ipynb`, copy verbatim into `utils.py`:
- `set_seed(seed)` — reproducibility
- `compute_retrieval_metrics(logits)` — Top-1/Top-5 from [N,N] matrix
- `summarize_metrics_over_seeds(metric_list)` — mean ± std aggregation
- `eval_images(real_images, fake_images, device)` — official evaluation
- All sub-functions: `pixcorr`, `ssim`, `alexnet`, `inception`, `clip_`, `effnet`, `swav`, `two_way_identification`

From `sample_codes/eeg_project_sample_code.ipynb`, copy into `data.py`:
- `_selected_channel_indices_from_jsonl()` and `load_eeg_dataset()` verbatim

### CLIP feature caching (`cache_clip_features.py`)

```python
# Load IP-Adapter's image encoder (CLIP-ViT-H-14, 1024-d projection)
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

encoder = CLIPVisionModelWithProjection.from_pretrained(
    "/hpc2hdd/home/ckwong627/workdir/models/IP-Adapter/models/image_encoder"
).to("cuda").eval()
processor = CLIPImageProcessor.from_pretrained(
    "/hpc2hdd/home/ckwong627/workdir/models/IP-Adapter/models/image_encoder"
)

# For each image:
inputs = processor(images=PIL_image, return_tensors="pt").to("cuda")
with torch.no_grad():
    image_embeds = encoder(**inputs).image_embeds  # [1, 1024]
image_embeds = F.normalize(image_embeds, dim=-1)
```

Save dict: `{image_id (str): tensor[1024]}` to `clip_cache/clip_train_features.pt` and `clip_cache/clip_test_features.pt`.

### Loss computation (`model.py`)

```python
def compute_loss(self, eeg_emb, clip_emb):
    alpha, beta = self.get_weights()  # fixed scalar or exp of nn.Parameter
    
    # Retrieval: symmetric InfoNCE
    temp = torch.exp(self.log_temperature)
    logits = (eeg_emb @ clip_emb.T) * temp           # [B, B]
    labels = torch.arange(len(eeg_emb), device=eeg_emb.device)
    L_ret = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    
    # Reconstruction: MSE in CLIP space
    L_rec = F.mse_loss(eeg_emb, clip_emb)
    
    return alpha * L_ret + beta * L_rec, L_ret.detach(), L_rec.detach()
```

### Reconstruction inference (`reconstruct.py`)

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "/hpc2hdd/home/ckwong627/workdir/models/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")
pipe.load_ip_adapter(
    "/hpc2hdd/home/ckwong627/workdir/models/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter_sd15.bin",
)
pipe.set_ip_adapter_scale(0.7)

# For each test EEG:
with torch.no_grad():
    eeg_emb = model.eeg_encoder(eeg_tensor)  # [1, 1024]

image = pipe(
    prompt="",
    ip_adapter_image_embeds=[eeg_emb.unsqueeze(0)],
    guidance_scale=7.5,
    num_inference_steps=50,
    generator=torch.Generator("cuda").manual_seed(seed),
).images[0]
```

### Evaluation (`evaluate.py`)
- **Retrieval**: encode all 200 test EEGs → [200, 1024]; load cached test image embeds → [200, 1024]; logits = `eeg @ img.T`; call `compute_retrieval_metrics(logits)`. Deterministic.
- **Reconstruction**: load 200 generated images per seed × 10 seeds; load 200 real test images as [200, 3, 256, 256]; call `eval_images(real, fake, device)` per seed; report mean ± std.

---

## 6. Execution Plan (Priority Order)

### P0: Must Complete (Days 1-12)

**Days 1-2: Environment + CLIP caching**
- Verify `test` conda env: `python -c "import torch, transformers, diffusers, datasets, clip"`
- Implement `config.py`, `utils.py`, `data.py`
- Implement `cache_clip_features.py`, run via `sbatch run_cache_clip.sh`

**Days 3-5: EEG Encoder + Retrieval**
- Implement `model.py` (EEGEncoder + UnifiedModel)
- Implement `train.py` Phase 1
- First run: `--alpha 1.0 --beta 0.0` (pure retrieval, simplest debug)
- Verify Top-1 > 1% (10× random baseline)
- Then run: `--alpha 1.0 --beta 0.5` (joint training)

**Days 6-8: Phase 2 Fine-tuning + Score Optimization**
- Run Phase 2 with various alpha/beta combos: {(1,0), (0.8,0.2), (0.5,0.5), (0.2,0.8), (0,1)}
- Run learnable-weights variant
- Target: Top-1 ~15-25%, Top-5 ~40-55%

**Days 9-12: Reconstruction Pipeline**
- Implement `reconstruct.py`
- Generate 200 images for 1 seed first → eyeball quality
- If quality is poor: tune IP-Adapter scale (0.5-1.0), guidance scale (5-10)
- Once working: generate 10 seeds × 200 images
- Run `evaluate.py` for full reconstruction metrics
- **Fallback if IP-Adapter generates garbage**: nearest-neighbor in CLIP space

### P1: Score Optimization (Days 13-16)
- Test-time augmentation for retrieval
- Tune IP-Adapter inference parameters
- Try alternative reconstruction losses: cosine similarity loss, Smooth L1
- Re-train winning configs with multiple seeds

### P2: Micro-Ablation (Days 13-16, teammates take over)
- Teammates implement EEG encoder variants (Pure CNN, Pure Transformer, EEGNet) by adding new files in `codes/encoders/`
- Same `train.py` works — they just swap the encoder via config
- Image encoder ablation: try LAION ViT-L/14 (768-d) variant

### P3: Report & Presentation (Days 17-20)
- Generate 8-12 qualitative examples (success + failure cases)
- Write technical report
- Finalize `run_all.ipynb` for TA reproducibility check
- Verify full pipeline reproducibility from scratch

---

## 7. Score Maximization Strategies

### Retrieval (25 pts)
- Aligning to LAION ViT-H/14 (1024-d) gives more representational capacity than ViT-L/14 (768-d)
- Two-phase training (large LR → small LR) + heavy augmentation in phase 1
- Test-time augmentation: encode same test EEG ~5 times with small perturbations, average
- The joint InfoNCE + MSE loss should outperform pure InfoNCE

### Reconstruction (25 pts)
- **SSIM (12.5 pts)**: pixel-level. Higher IP-Adapter scale (0.8-0.9) → more deterministic generation; lower guidance scale (5-6) → less creative deviation
- **CLIP Score (12.5 pts)**: 2-way identification. Generated images need to be semantically clearly identifiable
- Generate at 512×512 then resize to 256×256

### Methodology (20 pts)
- The "Architecture B is a special case of Architecture A" insight is a clean methodological contribution — make it the centerpiece of the report
- Ablation table: alpha/beta sweep including pure retrieval (Arch B Task 1), pure recon (Arch B Task 2), joint, and learnable
- Show learnable weights converge to a meaningful ratio (or don't, and explain)

### Code Quality (10 pts)
- Clean separation: one concern per file
- Single config dataclass — no argparse sprawl
- Reproducibility: all seeds in config
- README explaining how to reproduce all numbers in the report

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| IP-Adapter `ip_adapter_image_embeds` API not as expected | Read diffusers source; alternative: hook directly into UNet cross-attention |
| Generated images are noise | Verify L2-normalization handling; tune IP-Adapter scale |
| Training does not converge | Start with `--alpha 1 --beta 0` (pure retrieval, simpler) |
| 30-min SLURM limit insufficient | Math checks out: 16540 samples × 50 epochs / batch 128 = ~6450 batches × 100ms = ~11 min |
| Conda env `test` missing packages | Install with `pip install <pkg>` (NOT `conda install` to avoid env conflicts) |
| Embedding magnitude mismatch (L2-normalized vs raw CLIP) | Try both; the IP-Adapter image encoder outputs unnormalized embeddings |
| Overfitting on small dataset | 5 augmentations + dropout 0.1 + weight decay 0.05 + early stopping |

---

## 9. Verification Plan (End-to-End)

After implementation, run in order:

1. **CLIP cache**: `sbatch slurm_scripts/run_cache_clip.sh`
2. **Phase 1 sanity check**: `train.py --alpha 1 --beta 0 --epochs 5`
3. **Phase 1 full**: `sbatch slurm_scripts/run_train_phase1.sh`
4. **Phase 2 full**: `sbatch slurm_scripts/run_train_phase2.sh`
5. **Retrieval eval**: Top-1 > 5% (well above random 0.5%)
6. **Reconstruction**: `sbatch slurm_scripts/run_reconstruct.sh`
7. **Reconstruction eval**: SSIM and CLIP scores reported
8. **Final notebook**: open `codes/run_all.ipynb`
9. **Reproducibility check**: delete `checkpoints/` and `outputs/`, re-run from CLIP cache forward

---

## 10. Open Questions to Verify at Implementation Time

1. **Actual training set size**: Print `len(train_ds)` after `load_eeg_dataset(avg_trials=True)`. Could be ~1654 or ~16540 — affects batch sizing and epoch count.
2. **IP-Adapter embedding format**: Verify whether `ip_adapter_image_embeds=` accepts L2-normalized embeddings or wants raw projection output.
3. **Conda `test` env package check**: Verify `diffusers` and `clip` are installed.
4. **`open_clip_torch`**: May be needed if loading laion CLIP via open_clip API.
5. **Channel name `Cz`** in `EEG_CHANNELS.jsonl`: 62 channels listed but data has 63 — inspect.
