# VisualEEGDecoding Reproduction Plan (English)

## Context

Reproduce the **VisualEEGDecoding** repository (AAAI 2026 paper: "Leveraging Visual Blur Perception Characteristics for EEG Decoding") to perform **Brain-to-Image Retrieval** on the course dataset (DSAA2012 Project A). The paper reports 80% Top-1 and 96.9% Top-5 accuracy on the original Things-EEG dataset (10 subjects). We have only the course-designated single subject (sub-01) and need to adapt the pipeline accordingly.

**Repo path:** `/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/repository/VisualEEGDecoding`  
(referred to as `$REPO` below)

---

## Key Findings

### Data Format Mismatch
- **Course data** (`image-eeg-data/train.pt`, `test.pt`): preprocessed by teaching team, contains EEG tensors + image IDs (e.g., `"aardvark_01b"`). Shape: `[N_images, N_trials, 63, 250]`.
- **VisualEEGDecoding expected format**: `{'eeg': np.float16 [N, T, 63, 250], 'img': np.array [N, T] of image paths}` where image paths are relative to `Image_set/` (e.g., `"train_images/00001_aircraft_carrier/aircraft_carrier_01b.jpg"`).

### Model Requirements
- Needs **OpenCLIP RN50** pretrained weights to generate 12-level Gaussian-blur image features (1024-dim each).
- Image feature file format: `{blur_key: {image_path: torch.Tensor(1024)}}` for keys `'1','3','9','15','21','27','33','39','45','51','57','63'`.

### Environment
- Reuse **`test` conda env** (Python 3.10, torch 2.10+cu126, open-clip-torch 2.32.0, mne 1.8.0, scipy, numpy — all required packages present).
- The repo's `environment.yml` is for Windows; ignore it.

### Scope
- Only intra-subject training (1 subject = sub-01), not inter-subject (requires 10 subjects from original Things-EEG dataset we don't have).
- No reconstruction task in this repo — only retrieval.

---

## Implementation Steps

### Phase 0: Inspect Course Data Format (login node, no GPU)

**Script:** `$REPO/scripts/inspect_course_data.py`

Run on login node with `conda run -n test` to print exact keys and shapes of `train.pt` and `test.pt`. This informs Step 1 exactly.

```bash
conda run -n test python $REPO/scripts/inspect_course_data.py
```

Critical outputs needed:
- Keys in `train.pt` dict
- Shape of `eeg` tensor
- Format of image ID field (key name, value format)

---

### Phase 1: Download OpenCLIP RN50 Model (login node)

**Model destination:** `/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/`

The `process_image.py` script uses `open_clip.create_model_and_transforms('RN50', pretrained=<path>)`. We use `pretrained='openai'` to auto-download from OpenAI CDN, then save the cached weights.

**Script:** `$REPO/scripts/download_rn50.py`

```python
import open_clip, torch, os, shutil

model, _, _ = open_clip.create_model_and_transforms('RN50', pretrained='openai')
save_dir = '/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai'
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, 'open_clip_pytorch_model.bin'))
print("Saved to", save_dir)
```

Run: `conda run -n test python $REPO/scripts/download_rn50.py`

**Model size:** ~100MB (OpenAI CLIP RN50 weights)

---

### Phase 2: Convert Course Data to VisualEEGDecoding Format (login node)

**Script:** `$REPO/scripts/convert_course_data.py`

Does the following:
1. Loads course `train.pt` and `test.pt`
2. Scans course `training_images/` and `test_images/` directories to build a map from image stem → relative path (e.g., `"aircraft_carrier_01b"` → `"train_images/00001_aircraft_carrier/aircraft_carrier_01b.jpg"`)
3. Converts image IDs in `eeg['img']` to relative paths
4. Ensures EEG shape is `[N, T, 63, 250]` as numpy float16
5. Creates directory structure and saves:
   - `$REPO/data/things-eeg/Preprocessed_data/sub-01/train.pt`
   - `$REPO/data/things-eeg/Preprocessed_data/sub-01/test.pt`
6. Creates symlinks:
   - `$REPO/data/things-eeg/Image_set/train_images` → `image-eeg-data/training_images`
   - `$REPO/data/things-eeg/Image_set/test_images` → `image-eeg-data/test_images`

**Output dict format:**
```python
{
  'eeg': np.array([N_images, N_trials, 63, 250], dtype=np.float16),
  'img': np.array([N_images, N_trials], dtype=object),  # relative paths
  'label': ...,  # preserved from source
}
```

Run: `conda run -n test python $REPO/scripts/convert_course_data.py`

---

### Phase 3: Generate Multi-Blur Image Features (SLURM debug job)

**Modified script:** `$REPO/preprocess/process_image_course.py` (new file, adapted from `process_image.py`)

Changes from original:
- Fix Windows path separators (`\\` → `/`, using `os.path.join` properly)
- Update `base_path` to `$REPO/data/things-eeg/Image_set`
- Update pretrained path to `/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/open_clip_pytorch_model.bin`
- Update save paths to `$REPO/data/things-eeg/Image_feature/`

**SLURM script:** `$REPO/slurm_scripts/02_gen_blur_features.sh`
```bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --time=00:30:00
#SBATCH -o $REPO/slurm_scripts/logs/02_gen_blur_features_%j.out
#SBATCH -e $REPO/slurm_scripts/logs/02_gen_blur_features_%j.err
source ~/miniconda3/etc/profile.d/conda.sh && conda activate test
cd $REPO
python preprocess/process_image_course.py
```

**Output:**
- `$REPO/data/things-eeg/Image_feature/MultiBlur_RN50_train.pt` (~600MB)
- `$REPO/data/things-eeg/Image_feature/MultiBlur_RN50_test.pt` (~15MB)

**Time estimate:** ~20-25 min on 1 GPU for 12 blur levels × ~8000 training images.

---

### Phase 4: Train EEG Encoder (SLURM non-debug job)

**Modified script:** `$REPO/main_eeg_course.py` (new file, adapted from `main_eeg.py`)

Changes from original:
- `data_path` set to `$REPO/data/things-eeg`
- Loop over only `sub=1` (single subject), not sub 1-10
- `cross_subject=False` (intra-subject only)
- Seeds: run with 10 seeds (21-30) to get mean ± std as required by course

**SLURM script:** `$REPO/slurm_scripts/03_train_eeg.sh`
```bash
#SBATCH -p gpu_8h   # or appropriate partition
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --time=02:00:00
#SBATCH -o $REPO/slurm_scripts/logs/03_train_eeg_%j.out
#SBATCH -e $REPO/slurm_scripts/logs/03_train_eeg_%j.err
source ~/miniconda3/etc/profile.d/conda.sh && conda activate test
cd $REPO
python main_eeg_course.py
```

**Time estimate:** ~20 min per subject per seed × 10 seeds ≈ ~3-4 hours (use `gpu` or `gpu_8h` partition)

**Hyperparameters (unchanged from paper):**
- epochs=200, lr=0.001, batch_size=1024, no mixup, no filter
- All 63 channels, full 250 timepoints

---

### Phase 5: Evaluate with Course Metrics

**Script:** `$REPO/scripts/evaluate_course_metrics.py`

Using the trained model, run evaluation with the course evaluation protocol:
- 200-way zero-shot retrieval (200 test images)
- Report Top-1 and Top-5 accuracy
- Repeat with 10 random seeds, report mean ± std

The trained model's `saved_metirc` already contains per-run Top-1/Top-5; we collect across seeds.

---

## Critical Files

| File | Action | Purpose |
|------|---------|---------|
| `$REPO/scripts/inspect_course_data.py` | **Create** | Inspect course data format |
| `$REPO/scripts/download_rn50.py` | **Create** | Download OpenCLIP RN50 |
| `$REPO/scripts/convert_course_data.py` | **Create** | Convert EEG data format |
| `$REPO/preprocess/process_image_course.py` | **Create** | Generate multi-blur features (Linux-fixed) |
| `$REPO/main_eeg_course.py` | **Create** | Train on single subject with 10 seeds |
| `$REPO/slurm_scripts/02_gen_blur_features.sh` | **Create** | SLURM job for feature gen |
| `$REPO/slurm_scripts/03_train_eeg.sh` | **Create** | SLURM job for training |
| `$REPO/scripts/evaluate_course_metrics.py` | **Create** | Final evaluation script |

---

## Models / Data Downloads

| Item | Size | Location | Command |
|------|------|----------|---------|
| OpenCLIP RN50 (openai) | ~100MB | `/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/` | `python scripts/download_rn50.py` |

---

## SLURM Partition Strategy

| Step | Partition | Time | GPUs | Notes |
|------|-----------|------|------|-------|
| Feature generation | `debug` | 30 min | 1 A40 | May fit in debug quota |
| Training (10 seeds) | `gpu` or `gpu_8h` | 3-4 hr | 1 A40 | Exceeds debug 30-min limit |

---

## Verification

1. After Phase 2 conversion: check `$REPO/data/things-eeg/Preprocessed_data/sub-01/train.pt` contains correct keys and shapes
2. After Phase 3: verify `MultiBlur_RN50_train.pt` dict structure — 12 keys, each mapping ~8000 paths to 1024-dim tensors
3. After Phase 4: training log should show increasing Top-1 accuracy toward ~60-80% (single-subject may be lower than multi-subject paper result)
4. Final evaluation: report Top-1 and Top-5 accuracy (mean ± std, 10 seeds, 200-way)

---

## Expected Results & Paper Comparison

| Metric | Paper (10-subject avg, 200 images) | Expected (1 subject) |
|--------|-------------------------------------|----------------------|
| Top-1 Accuracy | ~80% | 60-75% (lower, single subject) |
| Top-5 Accuracy | ~96.9% | 85-95% |

Lower performance expected vs. paper because:
- Paper trains on 10 subjects (more data diversity)
- Paper uses original Things-EEG data (pre-processed with their pipeline)
- Course data may have different preprocessing than Things-EEG

---

## Notes

- The repo does **not** include a Brain-to-Image Reconstruction module — only retrieval
- `main_meg.py` requires MEG data (not available); skip this
- `visualization/brain_area.ipynb` can be run after training for analysis
- If debug partition queue is long, switch to `short` or `gpu_4h` partition
