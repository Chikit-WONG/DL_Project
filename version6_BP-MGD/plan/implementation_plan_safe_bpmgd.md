# Safe BP-MGD Implementation Plan for DSAA2012 Project1 Task 2

## Goal

Implement a leakage-safe EEG-to-image reconstruction pipeline for Project1 Task 2.

Method name:

**Safe BP-MGD: Safe Bio-Perceptual Multi-modal Guided Diffusion**

Main objective:

- Improve Task 2 reconstruction quality.
- Optimize both:
  - SSIM
  - CLIP Score
- Use only allowed information.
- Do not use test images, test labels, or test candidate images during generation, retrieval, hyperparameter tuning, or reranking.

The final system should support:

```text
test EEG -> reconstructed image
```

The test-time input must be only:

```text
1. test EEG
2. trained model weights
3. frozen public pretrained models
4. train-only feature memory bank
5. fixed hyperparameters selected on train/validation
```

Absolutely forbidden:

```text
1. using test ground-truth images for img2img
2. retrieving from test candidate images
3. using test labels/class names as prompt
4. tuning seeds/guidance/denoise strength by comparing with test GT
5. modifying official TA evaluation scripts
```

---

## Reference Repositories

Clone these repositories under `third_party/` for reference only.

```bash
mkdir -p third_party

git clone https://github.com/makeitperfect/VisualEEGDecoding third_party/VisualEEGDecoding
git clone https://github.com/XiaoZhangYES/CognitionCapturerPro third_party/CognitionCapturerPro
git clone https://github.com/ncclab-sustech/EEG_Image_decode third_party/EEG_Image_decode
git clone https://github.com/lucaspiper99/evnet third_party/evnet
```

Use them as implementation references:

```text
VisualEEGDecoding:
- multi-blur feature preprocessing
- blur-level feature selection
- EEG-image contrastive alignment

CognitionCapturerPro:
- multimodal image/text/depth/edge expansion
- uncertainty-weighted masking
- fusion encoder
- asymmetric alignment
- SDXL-Turbo + IP-Adapter generation

EEG_Image_decode:
- ATM EEG encoder
- prior diffusion
- low-level VAE/blurry image pipeline
- reconstruction metrics notebook
- SDXL reconstruction pipeline

evnet:
- EVNet / SubcorticalBlock / VOneBlock
- early-vision structural feature extraction
```

Do not blindly copy all code. Reuse ideas and import small reusable modules only when stable. The final codebase must remain clean, runnable, and reproducible.

---

## Required Final Repository Structure

Create this structure in the current project root.

```text
DL_Project/
├── configs/
│   ├── safe_bpmgd.yaml
│   ├── paths.example.yaml
│   └── ablations/
│       ├── baseline_clip.yaml
│       ├── add_multiblur.yaml
│       ├── add_struct.yaml
│       ├── add_prior.yaml
│       └── full_safe_bpmgd.yaml
│
├── src/
│   └── safe_bpmgd/
│       ├── __init__.py
│       ├── data/
│       │   ├── dataset.py
│       │   ├── splits.py
│       │   └── leakage_guard.py
│       │
│       ├── encoders/
│       │   ├── eeg_atm.py
│       │   ├── eeg_cogcap.py
│       │   ├── heads.py
│       │   └── projection.py
│       │
│       ├── features/
│       │   ├── cache_clip.py
│       │   ├── cache_multiblur.py
│       │   ├── cache_edge_depth.py
│       │   ├── cache_vae.py
│       │   ├── cache_evnet.py
│       │   └── train_memory_bank.py
│       │
│       ├── losses/
│       │   ├── contrastive.py
│       │   ├── multiblur_loss.py
│       │   ├── structure_loss.py
│       │   └── uncertainty_loss.py
│       │
│       ├── prior/
│       │   ├── prior_diffusion.py
│       │   └── train_prior.py
│       │
│       ├── generation/
│       │   ├── generate_candidates.py
│       │   ├── sdxl_ipadapter.py
│       │   ├── control_conditions.py
│       │   └── rerank.py
│       │
│       ├── eval/
│       │   ├── eval_task2.py
│       │   ├── qualitative_grid.py
│       │   └── sanity_checks.py
│       │
│       └── utils/
│           ├── seed.py
│           ├── config.py
│           ├── logging.py
│           └── io.py
│
├── scripts/
│   ├── 00_check_data.py
│   ├── 01_cache_train_features.py
│   ├── 02_train_encoder.py
│   ├── 03_train_prior.py
│   ├── 04_generate_recon.py
│   ├── 05_eval_recon.py
│   └── 06_run_ablation.py
│
├── slurm_scripts/
│   ├── run_cache_features.sh
│   ├── run_train_encoder.sh
│   ├── run_train_prior.sh
│   ├── run_generate_recon.sh
│   └── run_eval.sh
│
├── outputs/
├── checkpoints/
├── feature_cache/
├── third_party/
└── README.md
```

---

## Leakage Guard: Must Implement First

Before implementing the model, implement `src/safe_bpmgd/data/leakage_guard.py`.

It must enforce:

```text
1. test images cannot be loaded during training
2. test images cannot be used to build memory bank
3. test images cannot be used for prototype retrieval
4. test labels/class names cannot be used as prompts
5. official evaluation scripts cannot be modified
6. hyperparameters must be selected only on train/validation
```

Add explicit checks:

```python
assert "test_images" not in train_memory_bank_paths
assert "test" not in prototype_bank_name.lower()
assert config.inference.use_test_candidate_bank is False
assert config.inference.use_test_gt_img2img is False
```

At inference time, allow only:

```text
test EEG
train-only memory bank
frozen pretrained models
fixed config
trained checkpoints
```

Generate a text file for every run:

```text
outputs/<run_name>/leakage_report.txt
```

The leakage report must state:

```text
- train split path
- validation split path
- test EEG path
- feature caches used
- whether any test image path was accessed
- whether any test label/class prompt was used
- memory bank source: train-only
```

---

## Dataset Loading

Use the official/starter Project1 data loader as much as possible.

Requirements:

```text
EEG shape: [63, 250]
testing avg_trials must be true
do not change official train/test split
```

Implement:

```python
load_train_dataset(avg_trials=True or False based on config)
load_val_dataset(from train split only)
load_test_eeg(avg_trials=True)
```

Validation split:

```text
Use only training split.
Default: 95% train / 5% validation.
Seed fixed.
Save val indices to outputs/<run_name>/val_indices.json.
```

Do not create validation from test data.

---

## Feature Caching

All image-derived training targets must be cached from training images only.

Create `scripts/01_cache_train_features.py`.

Cache the following:

### Raw CLIP / DINO Features

From train images:

```text
feature_cache/train/clip_rn50.pt
feature_cache/train/clip_vith14.pt
feature_cache/train/dinov2.pt
```

Use frozen encoders.

### Multi-blur CLIP Features

Reference VisualEEGDecoding.

For each train image, create blur levels:

```text
raw
blur_sigma_1
blur_sigma_2
blur_sigma_4
blur_sigma_8
foveated_blur
```

Cache:

```text
feature_cache/train/multiblur_clip.pt
```

Shape should be:

```text
[num_train_images, num_blur_levels, clip_dim]
```

### Edge and Depth Features

Reference CognitionCapturerPro.

For each train image:

```text
edge map: Canny or HED
depth map: DepthAnything
```

Cache:

```text
feature_cache/train/edge_maps.pt
feature_cache/train/depth_maps.pt
feature_cache/train/edge_clip.pt
feature_cache/train/depth_clip.pt
```

### SDXL VAE Latents / Blurry Image Latents

Reference EEG_Image_decode low-level pipeline.

Cache:

```text
feature_cache/train/vae_latents.pt
feature_cache/train/blurry_latents.pt
```

Use these for structural reconstruction loss.

### EVNet Structural Features

Reference evnet.

Use EVNet/SubcorticalBlock/VOneBlock as a frozen early-vision feature extractor.

Cache:

```text
feature_cache/train/evnet_struct.pt
```

Suggested implementation:

```text
image -> EVNet front-end -> selected activation -> global average pooling -> projection
```

Do not train EVNet. Use it as frozen structural supervision.

---

## Train-only Prototype Memory Bank

Create:

```text
feature_cache/train/prototype_bank.pt
```

It must contain only train image data:

```python
{
    "image_ids": train_image_ids,
    "image_paths": train_image_paths,
    "clip": clip_features,
    "multiblur": multiblur_features,
    "evnet": evnet_struct_features,
    "edge": edge_features,
    "depth": depth_features,
    "vae": vae_latents,
}
```

No test images allowed.

At inference:

```text
test EEG -> predicted semantic embedding -> retrieve top-k prototypes from train bank
```

Use prototypes only as weak priors for generation.

---

##  Model Architecture

Create `src/safe_bpmgd/encoders/eeg_atm.py`.

Backbone:

```text
ATM-style EEG encoder:
- channel-wise attention
- temporal-spatial convolution
- MLP projector
- LayerNorm
```

Input:

```text
x_eeg: [B, 63, 250]
```

Shared hidden:

```text
h: [B, hidden_dim]
```

Create multi-head outputs in `heads.py`:

```python
outputs = {
    "z_sem": semantic_head(h),          # CLIP/DINO semantic embedding
    "z_blur_logits": blur_head(h),      # blur-level selection logits
    "z_struct": struct_head(h),         # EVNet structural embedding
    "z_edge": edge_head(h),             # edge embedding
    "z_depth": depth_head(h),           # depth embedding
    "z_vae": vae_head(h),               # SDXL VAE latent or projected latent
    "uncertainty": uncertainty_head(h), # confidence / uncertainty scalar
}
```

Normalize embeddings consistently.

Important:

```text
Check whether IP-Adapter expects normalized or raw embeddings.
Keep both versions available:
- normalized for contrastive loss
- raw/projected for generation
```

---

##  Training Loss

Create losses:

```text
L_total =
    L_semantic
  + λ_blur * L_multiblur
  + λ_struct * L_evnet_struct
  + λ_edge * L_edge
  + λ_depth * L_depth
  + λ_vae * L_vae
  + λ_unc * L_uncertainty
```

Recommended initial weights:

```yaml
loss:
  lambda_semantic: 1.0
  lambda_blur: 0.5
  lambda_struct: 0.3
  lambda_edge: 0.2
  lambda_depth: 0.2
  lambda_vae: 0.3
  lambda_uncertainty: 0.05
```

### Semantic Loss

Use InfoNCE / CLIP-style contrastive loss:

```text
EEG semantic embedding <-> train image CLIP embedding
```

### Multi-blur Loss

Use predicted blur weights:

```python
w = softmax(z_blur_logits)
target = sum_i w_i * multiblur_clip[:, i, :]
```

Loss:

```text
cosine distance or InfoNCE between z_sem and adaptive multiblur target
```

### Structural Loss

Use EVNet:

```text
SmoothL1(z_struct, evnet_struct_target)
+ cosine loss
```

### Edge / Depth / VAE Loss

Use:

```text
SmoothL1 + cosine
```

For VAE latent, consider projecting high-dimensional latent into a compact latent vector first.

### Uncertainty Loss

Use uncertainty to downweight noisy samples.

Simple version:

```python
weighted_loss = exp(-u) * base_loss + u
```

Do not make uncertainty dominate training.

---

## Training Schedule

Implement `scripts/02_train_encoder.py`.

Use staged training.

### Stage A: Semantic warm-up

Epochs: 20

Enable:

```text
L_semantic
L_multiblur
```

Disable:

```text
L_struct
L_edge
L_depth
L_vae
```

Goal:

```text
stable EEG-to-CLIP alignment
validation retrieval improves
```

### Stage B: Add structural heads

Epochs: 30

Enable:

```text
L_semantic
L_multiblur
L_struct
L_edge
L_depth
L_vae
```

Goal:

```text
keep CLIP alignment while improving structural representation
```

### Stage C: Fine-tuning

Epochs: 20-30

Lower LR.

Enable all losses.

Recommended optimizer:

```yaml
optimizer: AdamW
lr: 1e-4
weight_decay: 0.05
batch_size: 512 or 1024
epochs: 80
scheduler: cosine
mixed_precision: true
```

Save:

```text
checkpoints/<run_name>/encoder_best_val.pt
checkpoints/<run_name>/encoder_last.pt
```

Validation metrics:

```text
val retrieval top-1/top-5
val semantic cosine
val structure cosine
val loss
```

---

## Prior Diffusion

Reference EEG_Image_decode.

Create:

```text
src/safe_bpmgd/prior/prior_diffusion.py
scripts/03_train_prior.py
```

Goal:

```text
z_sem_eeg -> refined image prior z_img_hat
```

Training target:

```text
train image CLIP embedding
```

Condition:

```text
EEG semantic embedding z_sem
```

Use a lightweight MLP/UNet-style diffusion prior.

Fallback:

If prior diffusion is unstable, implement a simpler prior mapper:

```text
z_img_hat = MLP(z_sem)
```

Train with:

```text
MSE + cosine loss
```

The simple prior mapper must be kept as fallback.

---

## Reconstruction Generation

Create:

```text
src/safe_bpmgd/generation/generate_candidates.py
src/safe_bpmgd/generation/sdxl_ipadapter.py
src/safe_bpmgd/generation/control_conditions.py
scripts/04_generate_recon.py
```

Generation input:

```text
1. refined CLIP prior z_img_hat
2. EEG-predicted VAE/blurry latent
3. EEG-predicted edge condition
4. EEG-predicted depth condition
5. train-only prototype image/latent
6. fixed prompt template, no test label
```

Prompt rule:

```text
Do not use test label/class name.
Use neutral prompt only:
"a natural image, high quality, realistic"
or no text prompt if image-conditional generation works.
```

Use:

```text
SDXL-Turbo
IP-Adapter
optional ControlNet edge/depth
img2img or latent initialization
```

Generate candidates:

```yaml
num_candidates_per_eeg: 16 initially
final_num_candidates_per_eeg: 32 or 64 if compute allows
denoise_strength_grid: [0.35, 0.45, 0.55]
guidance_scale_grid: [3.0, 5.0, 7.0]
seed_list: fixed list from config
```

Important:

All grids must be selected on validation before final test generation.

---

## Self-reranking Without Leakage

Create:

```text
src/safe_bpmgd/generation/rerank.py
```

For each candidate image, compute:

```text
CLIP(candidate)
EVNet(candidate)
edge(candidate)
depth(candidate)
VAE(candidate)
```

Rerank using only EEG-predicted targets:

```python
score =
    0.40 * cosine(CLIP(candidate), z_sem)
  + 0.25 * cosine(EVNet(candidate), z_struct)
  + 0.15 * cosine(VAE(candidate), z_vae)
  + 0.10 * cosine(edge(candidate), z_edge)
  + 0.10 * cosine(depth(candidate), z_depth)
```

Do not use ground-truth test image in reranking.

Save:

```text
outputs/<run_name>/candidates/
outputs/<run_name>/final_recon/
outputs/<run_name>/rerank_scores.json
```

---

## Evaluation

Create:

```text
scripts/05_eval_recon.py
```

Use official TA evaluation code or call the official evaluation functions.

Do not modify official evaluation logic.

Report:

```text
SSIM
CLIP Score
optional: PixCorr / Inception / SwAV if available
```

Create qualitative grid:

```text
8-12 examples
GT image only for final report visualization/evaluation stage
reconstructed image
short success/failure note
```

Important:

GT images may be used only for official evaluation/reporting, not for generation, reranking, seed selection, or hyperparameter tuning.

---

## Ablation Plan

Create `scripts/06_run_ablation.py`.

Run these variants:

```text
A0: EEG -> CLIP -> SDXL baseline
A1: A0 + prior mapper/prior diffusion
A2: A1 + multi-blur adaptive alignment
A3: A2 + VAE/blurry low-level branch
A4: A3 + EVNet structural branch
A5: A4 + edge/depth control
A6: A5 + train-only prototype retrieval
A7: A6 + multi-candidate self-reranking
```

For each variant, record:

```text
validation retrieval top-1/top-5
validation SSIM
validation CLIP
test SSIM
test CLIP
leakage report path
```

Expected trend:

```text
multi-blur mainly improves semantic alignment / CLIP
EVNet + VAE + edge/depth mainly improves SSIM
train-only prototype retrieval improves structure but must stay train-only
self-reranking improves final candidate selection without leakage
```

---

## Milestones

### Milestone 1: Baseline running

Target:

```text
data loads correctly
EEG encoder trains
basic CLIP alignment works
one reconstructed image can be generated
```

Deliverables:

```text
encoder checkpoint
sample reconstruction grid
no leakage report
```

### Milestone 2: Multi-blur alignment

Target:

```text
multi-blur feature cache complete
blur-selection head works
validation retrieval improves over baseline
```

### Milestone 3: Structural branch

Target:

```text
EVNet feature cache complete
VAE/edge/depth cache complete
structural losses decrease
generated images have better layout/shape
```

### Milestone 4: Prior + generation

Target:

```text
prior diffusion or prior mapper trained
SDXL/IP-Adapter reconstruction pipeline stable
```

### Milestone 5: Safe prototype + reranking

Target:

```text
train-only memory bank works
candidate generation works
self-reranking works
leakage report confirms no test image usage
```

### Milestone 6: Final run

Target:

```text
final test reconstructions generated
official metrics computed
8-12 qualitative examples prepared
README reproducibility instructions complete
```

---

## Fallback Strategy

If full system is too slow or unstable, use this fallback order:

### Fallback 1: Remove ControlNet

Keep:

```text
CLIP prior
IP-Adapter
VAE/blurry img2img
train-only prototype
```

Remove:

```text
edge/depth ControlNet
```

### Fallback 2: Replace prior diffusion with MLP prior mapper

Use:

```text
z_img_hat = MLP(z_sem)
```

Loss:

```text
MSE + cosine
```

### Fallback 3: Remove EVNet from generation but keep EVNet loss

Use EVNet only for training/reranking, not as direct generation condition.

### Fallback 4: Use train-only prototype img2img

For each test EEG:

```text
retrieve top-1/top-3 train prototype
use prototype as weak img2img init
use EEG CLIP prior as IP-Adapter condition
```

This is still leakage-safe because the prototype bank is train-only.

---

## README Requirements

Write `README.md` with:

```text
1. environment setup
2. data path setup
3. feature caching commands
4. encoder training command
5. prior training command
6. reconstruction generation command
7. evaluation command
8. leakage-safety statement
9. external resources disclosure
10. expected output structure
```

Include this statement:

```text
This project uses a train-only prototype memory bank. No test images, test labels, or test candidate images are used for retrieval, generation, reranking, hyperparameter tuning, or prompt construction. Test ground-truth images are used only by the official evaluation code and for final qualitative visualization.
```

---

## External Resource Disclosure

In the final report and README, disclose:

```text
- OpenCLIP / CLIP
- SDXL-Turbo
- IP-Adapter
- DepthAnything if used
- EVNet
- VisualEEGDecoding reference
- CognitionCapturerPro reference
- EEG_Image_decode reference
```

Also disclose any copied/adapted files.

Do not claim the method is purely original. The contribution is the safe integration:

```text
multi-blur perceptual alignment
+ multimodal asymmetric EEG alignment
+ EVNet early-vision structural supervision
+ prior diffusion / VAE low-level reconstruction
+ train-only prototype retrieval
+ leakage-safe candidate reranking
```

---

## Final Success Criteria

Minimum acceptable:

```text
pipeline runs end-to-end
official evaluation runs
valid reconstruction outputs
leakage report clean
README complete
```

Strong target:

```text
SSIM > CogCapPro-style baseline
CLIP > CogCapPro-style baseline
qualitative examples show better object identity and structure
```

Expected realistic target:

```text
SSIM: 0.44-0.48
CLIP: 0.87-0.90
```

Optimistic target:

```text
SSIM: 0.48-0.50
CLIP: 0.90-0.91
```

Do not promise these numbers in code or README. Report actual measured values honestly.

---

## Immediate Next Actions for Codex

Start implementation in this order:

```text
1. Inspect current project root and existing starter code.
2. Create repository structure.
3. Implement leakage_guard.py.
4. Implement dataset loading wrappers.
5. Implement feature caching for CLIP and multi-blur first.
6. Implement EEG encoder + semantic/multiblur heads.
7. Train baseline and verify validation retrieval.
8. Add VAE/edge/depth/EVNet caches.
9. Add structural heads and losses.
10. Add prior mapper first, prior diffusion second.
11. Add SDXL/IP-Adapter generation.
12. Add train-only prototype retrieval.
13. Add multi-candidate self-reranking.
14. Add official evaluation wrapper.
15. Add ablation runner.
16. Write README and leakage report.
```

Never skip the leakage guard.
Never use test images in memory bank.
Never tune generation parameters using test GT.

## 注意事项

1. 工作区域主要在这个文件夹下/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version6_BP-MGD。但也可能会修改其他文件夹下的文件。
2. 可能要借鉴的论文在这个路径下/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/paper。这些论文对应的github仓库的代码，在这个路径下/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/repository。
3. 更多注意事项，请参考/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/plan/Notes_for_Attention.md。
4. 把所有任务完成，不要停，该等待的作业就等待，要完成的作业完成了，就继续下一个任务，直到所有任务完成，task 2的模型得分出来。
5. 用于slurm集群提交作业的脚本放在这个路径下/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version6_BP-MGD/slurm_scripts。
6. 程序产生的输出都放在这个路径下/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version6_BP-MGD/outputs。
7. 最好能把task2生成的图片和ground truth拼起来，让我更容易的去对比效果。
8. 把plan mode产生的计划，储存为两份markdown，一份英文，一份中文，放在这个路径下/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version6_BP-MGD/plan。
