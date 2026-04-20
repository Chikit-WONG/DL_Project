# CognitionCapturerPro Fix Plan: Closing the Paper Score Gap

## Context

The repository CognitionCapturerPro has been made runnable on the course EEG dataset by a previous Codex pass, but the achieved scores are far below the paper's reported values:

| Metric | Current | Paper (10-subject avg) |
|--------|---------|------------------------|
| Retrieval Top-1 (any modality) | 23.5% | 61.2% |
| Retrieval Top-5 (any modality) | 55.5% | 90.8% |
| CLIP reconstruction | 0.5 (chance) | 0.830 |
| PixCorr | 0.0 | 0.163 |
| SSIM | 0.006 | 0.398 |

Reconstruction metrics (CLIP, AlexNet2/5, Inception) are all at 0.5, which is the chance level for two-way identification tests. PixCorr = 0.0. This indicates the generated images are nearly constant or meaningless — not a case of "slightly undertrained", but a fundamental pipeline failure. This plan identifies and addresses the root causes in priority order.

---

## Root Cause Analysis

### 1. CRITICAL — Basename Collision in Diffusion Embeddings

**File**: `src/cogcappro/generate_image/generator.py` ~L272–290

The diffusion embedding dictionary is keyed by `os.path.basename(img_path)` (just the filename). Since the course image tree has many class directories, images across different classes often share the same basename (e.g., `class_A/n01234.jpg` and `class_B/n01234.jpg` are both stored under key `n01234.jpg`). The second image silently overwrites the first. The code even explicitly logs "Duplicate filename … previous embedding will be overwritten" warnings during preparation.

**Consequence**: Most test images receive a random wrong conditioning embedding when passed to SDXL-Turbo. The model generates images that are effectively unconditioned — nearly identical or degenerate — regardless of the EEG input. This single bug fully explains:
- CLIP/AlexNet/Inception all at 0.5 (two-way identification at chance)
- PixCorr = 0.0 (no pixel-level correlation between generated and real images)
- SSIM ≈ 0.006 (near-zero structural similarity)

**Fix**: Replace `os.path.basename(img_path)` with `f"{class_name}/{img_filename}"` as the embedding key, where `class_name = os.path.basename(os.path.dirname(img_path))`. This produces unique keys like `n01234_aircraft_carrier/n01234_aircraft_carrier_06s.jpg` that do not collide across class directories.

Update the corresponding lookup in `src/cogcappro/align/data.py` to use the same key scheme.

---

### 2. CRITICAL — Severely Insufficient Training Epochs

**Files**: `slurm_scripts/07_train_retrieval_full.sh`, `slurm_scripts/08_align_full.sh`

The base config (`configs/cogcappro.yaml`) specifies three training stages: 20 + 40 + 20 = **80 epochs total** for retrieval. The full run used only **10 epochs** (8× less). Alignment used only **1 epoch** (approximately 20× less than needed).

**Consequence**: The EEG projection head is severely undertrained. It cannot reliably map EEG signals to the correct region of CLIP embedding space. Even with correct diffusion embeddings, the generation quality will be limited by a poorly trained backbone.

**Fix**: Update retrieval training to 80 epochs (matching the config default). Update alignment to 15 epochs. Switch from the `debug` partition (30-min limit) to `long_gpu` or `emergency_gpua40`. Create new scripts `07b_train_retrieval_full_v2.sh` and `08b_align_full_v2.sh` to preserve the originals.

---

### 3. HIGH — Uncertainty-Aware Masking (UM) Bypassed in Alignment

**File**: `src/cogcappro/align/data.py` ~L89–94

The paper's Uncertainty-weighted Masking (UM) module is described as a core architectural contribution. The current code hardcodes `DirectT` (identity/pass-through) in the alignment phase, replacing UM entirely:

```python
config.data.uncertainty_aware = False
config.data.blur_type = OmegaConf.create(
    {"target": "cogcappro.models.inpainting_data.DirectT", "params": {}}
)
```

**Consequence**: The alignment model cannot leverage the spatially-selective blur signal that UM provides to focus on high-certainty regions. This degrades the quality of the diffusion-space alignment.

**Fix**: Remove these 6 lines. If the FoveaBlur feature cache (`Image_feature_new/FoveaBlur/`) does not exist for the course dataset, it will be computed automatically during the next alignment run (adding ~10–30 min of pre-computation).

---

### 4. MEDIUM — Dataset Scale vs. Paper

The paper evaluates on 10 subjects × the full Things-EEG release. This run uses 1 subject from the course dataset with a different preprocessing protocol. Some score gap is irreducible. However, fixing the above three issues should move scores substantially above chance and into a plausible range for single-subject performance.

---

## Implementation Steps

### Step 1: Fix Embedding Key Collision

**File**: `src/cogcappro/generate_image/generator.py`

In the `prepare_embedding()` function, change lines ~272–290:

- Add: `class_name = os.path.basename(os.path.dirname(img_path))`
- Add: `embed_key = f"{class_name}/{img_filename}"`
- Replace: `target_dict[img_filename] = valid_embedding` → `target_dict[embed_key] = valid_embedding`
- Remove the duplicate-filename warning block (it should no longer trigger)

**File**: `src/cogcappro/align/data.py`

In `load_diffusion_embeddings()`, change the lookup logic (~L152–169):

- Add: `embed_key = f"{class_name}/{img_filename}"` (class_name already computed)
- Change primary lookup from `img_filename` to `embed_key`
- Keep a simple basename fallback for backward compatibility with old `.pt` files
- Remove the complex prefix/class-suffix fallback chain (it compensated for the wrong key)

After this fix, re-run diffusion embedding preparation; verify zero "Duplicate filename" warnings in the job log.

---

### Step 2: Restore Uncertainty-Aware Alignment

**File**: `src/cogcappro/align/data.py`

Remove lines 89–94 that force `uncertainty_aware=False` and `blur_type=DirectT`. The config's default value (`uncertainty_aware=True`) will take effect.

---

### Step 3: Increase Training Duration

**New file**: `slurm_scripts/07b_train_retrieval_full_v2.sh`
- Remove `--max_epochs 10` override so the trainer uses the config default (80 epochs)
- Switch partition: `--partition long_gpu` or `emergency_gpua40`
- Set time limit: `--time 24:00:00`

**New file**: `slurm_scripts/08b_align_full_v2.sh`
- Set `--epoch 15`
- Switch partition: `--partition long_gpu`
- Set time limit: `--time 12:00:00`

---

### Step 4: Re-run Full Pipeline

After all code fixes, re-run in this order:

| Step | Script | Wait condition |
|------|--------|----------------|
| 1 | `02b_reprepare_diffusion_embeddings.sh` | Zero duplicate warnings in log |
| 2 | `07b_train_retrieval_full_v2.sh` | 80 epoch rows in metrics.csv |
| 3 | `08b_align_full_v2.sh` | Job exits successfully |
| 4 | `09_generate_full.sh` | Generated images exist in `generated_image/all/` |
| 5 | `10_eval_reconstruction_full.sh` | `reconstruction_metrics.json` updated |
| 6 | `11_multi_seed_summary.sh` | `summary_metrics.json` updated |

---

## Critical Files

| File | Role |
|------|------|
| `src/cogcappro/generate_image/generator.py` | Fix basename→class/basename key (CRITICAL) |
| `src/cogcappro/align/data.py` | Fix key lookup + remove DirectT override |
| `slurm_scripts/07b_train_retrieval_full_v2.sh` | 80-epoch retrieval training |
| `slurm_scripts/08b_align_full_v2.sh` | 15-epoch alignment |
| `slurm_scripts/02b_reprepare_diffusion_embeddings.sh` | Re-generate fixed embeddings |
| `configs/cogcappro.yaml` | Reference for correct stage epoch counts |
| `configs/local.yaml` | Dataset/model paths |

---

## Verification Checklist

1. **Embedding fix**: Job log for `02b` shows **zero** "Duplicate filename" warnings.
2. **Training**: `runs/full/.../lightning_logs/metrics.csv` has 80 epoch rows with decreasing loss.
3. **Alignment**: Alignment job log shows FoveaBlur cache loaded or freshly computed.
4. **Reconstruction metrics**: `reconstruction_metrics.json` shows CLIP > 0.6, PixCorr > 0.05.
5. **Retrieval metrics**: `test_results.json` shows any-modality Top-1 > 30%.

---

## Expected Improvement After All Fixes

| Metric | Before Fix | Expected After Fix |
|--------|------------|--------------------|
| Retrieval Top-1 (any modality) | 23.5% | 40–55% |
| Retrieval Top-5 (any modality) | 55.5% | 75–88% |
| CLIP reconstruction | 0.5 | 0.65–0.80 |
| PixCorr | 0.0 | 0.05–0.15 |
| SSIM | 0.006 | 0.10–0.30 |

Note: Single-subject performance will not reach the paper's 10-subject average (61.2% Top-1), but should be plausible and clearly above chance.

---

## Known Residual Limitations

Even after all fixes, some gap with the paper remains because:
- The course dataset covers only 1 subject; the paper averages 10.
- The course data preprocessing protocol may differ from the original Things-EEG release.
- The paper may use additional training tricks or hyperparameter tuning not reflected in the public configs.

---

## Additional Bugs Discovered During Execution

Two more bugs were found while running the fixed pipeline:

### 5. CRITICAL — VAE Float16 Overflow → All-Black Images

**File**: `src/cogcappro/generate_image/generator.py` — `_init_pipeline()`

The original code called `self.pipe.upcast_vae()` and then immediately forced the VAE back to float16:
```python
if hasattr(self.pipe, "vae") and getattr(self.pipe.vae.config, "force_upcast", False):
    self.pipe.vae.config.force_upcast = False
    self.pipe.vae.to(dtype=torch.float16)
```
This negated the upcast, causing the VAE decoder to run in float16. SDXL-Turbo's VAE is known to overflow in float16, producing NaN values which cast to 0 (all-black pixels). All 200 images became byte-for-byte identical black images.

**Fix**: Replace those 3 lines with `self.pipe.vae.config.force_upcast = True`. The pipeline's own decode logic then temporarily upcasts to float32, preventing NaN.

### 6. HIGH — IP-Adapter Embedding Dimension Error with `guidance_scale=0.0`

**File**: `src/cogcappro/generate_image/generator.py` — `_prepare_embeddings()`

When `guidance_scale=0.0`, `do_classifier_free_guidance=False`. The pipeline's `prepare_ip_adapter_image_embeds` passes the input embeds unchanged (no `chunk(2)` split). However, `_prepare_embeddings` always stacked `[uncond, cond]` → `[2, 1, 1024]`. This 3D tensor was passed as-is, causing the IP-Adapter cross-attention processor to reshape it to `[1, 2, heads, head_dim]` and treat both the zero uncond row AND the real EEG row as conditioning tokens — diluting the signal and, combined with bug #5, producing identical outputs.

**Fix**: When `do_classifier_free_guidance=False`, return `embed.unsqueeze(0)` → shape `[1, 1, 1024]` (cond-only, 3D as required). When `do_cfg=True`, keep the original `[2, 1, 1024]` stacking.

---

## Actual Results Achieved (2026-04-19 to 2026-04-20)

All fixes applied. Pipeline re-run on subject sub-01 with seed 0.

### Retrieval (EEG → CLIP matching)

| Metric | Before Fix | After Fix | Paper |
|--------|------------|-----------|-------|
| Top-1 (any modality) | 23.5% | **61.0%** | 61.2% |
| Top-5 (any modality) | 55.5% | **88.0%** | 90.8% |

Nearly perfect match with the paper on retrieval.

### Reconstruction (IP-Adapter image generation)

Two alignment approaches were attempted. `all_before` uses the raw EEG CLIP embeddings; `all` uses post-alignment embeddings.

| Metric | `all_before` (EEG direct) | `all` (SimpleAlignPipe) | Paper |
|--------|---------------------------|--------------------------|-------|
| CLIP (↑) | **0.707** | 0.659 | 0.830 |
| PixCorr (↑) | 0.130 | **0.133** | 0.163 |
| SSIM (↑) | **0.316** | 0.236 | 0.398 |
| AlexNet-2 (↑) | **0.663** | 0.618 | 0.831 |
| AlexNet-5 (↑) | **0.698** | 0.682 | 0.937 |
| Inception (↑) | 0.597 | **0.607** | 0.720 |

`all_before` achieves **~80% of paper reconstruction scores** and is the best available result. SimpleAlignPipe gives marginal gains on PixCorr and Inception but is worse on CLIP, SSIM, and AlexNet — overall not a clear improvement.

### Summary

The pipeline is now fully functioning. The remaining gap vs. the paper is from:
1. **Alignment**: The EEG retrieval model already outputs CLIP-compatible embeddings, so the alignment stage introduces noise rather than reducing it. The original `all_before` path is optimal.
2. **Single-subject vs. 10-subject average**: Irreducible gap from less training data.

---

## Alignment Investigation (2026-04-20)

Two alignment approaches were explored to fix the initial mode collapse (DiffusionPriorUNet, best cosine sim 0.005 from 15 epochs with broken warmup schedule):

### Attempt 1: DiffusionPriorUNet (30 epochs, fixed warmup)

**Root cause of original collapse**: `num_warmup_steps=100` was hardcoded but total training steps were only ~30 (batch_size=10240 on 16,540 samples). The scheduler never reached the target LR.

**Fixes applied** (`diffusion_pipe.py`, `main.py`):
- Warmup = `max(1, total_steps // 10)` (proportional, no longer hardcoded)
- Train batch_size 10240 → 512 (33 steps/epoch → proper warmup)

**Result**: Still collapsed. Best val cosine sim = 0.009 across 30 epochs. Training loss decreased (1.19 → 0.33), but DDPM inference did not produce aligned embeddings. Likely needs 100+ epochs for UNet-style diffusion prior.

### Attempt 2: SimpleAlignPipe (direct MLP, 100 epochs, fixed loss)

**Root causes fixed** (`diffusion_pipe.py`):
- `SimpleAlignMLP.forward()`: removed L2 normalization that conflicted with unnormalized MSE targets
- `SDEmbeddingLoss`: normalize both pred and target before MSE; removed `loss_reg = u.pow(2).mean()` (meaningless for normalized outputs)
- `OneCycleLR`: lowered `max_lr` from `lr×8` to `lr×3`; re-enabled modality masking

**Result**: val cosine sim reached **0.770** (epoch 31), early-stopped at epoch 51. Dramatic improvement over DiffusionPriorUNet (0.009 → 0.770).

**Reconstruction metrics**: Mixed vs. `all_before`. The aligned embeddings (val-cos 0.770 vs diffusion target) produce slightly better PixCorr/Inception but worse CLIP/SSIM. This indicates the EEG CLIP embeddings are already well-positioned in the IP-Adapter's expected semantic space; projecting them to a different "diffusion embedding" space introduces distortion.
