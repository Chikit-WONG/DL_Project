# version5_VED: VisualEEGDecoding Course Adaptation

[中文 README](README-CN.md)

This folder is the course-adapted `VisualEEGDecoding` branch for the DSAA2012 final project. It starts from the blur-aware EEG-to-image retrieval route in Liu et al. and extends it into a reproducible task-2 pipeline for retrieval-augmented reconstruction [1].

All generated features, checkpoints, logs, task-2 metadata, reconstructed images, and evaluation JSON files are written under `output/` so the full run can be copied back with one `rsync`.

## Project Summary

`version5_VED` now covers both required tasks:

1. **Task 1: EEG-to-image retrieval**
2. **Task 2: EEG-to-image reconstruction**

The task-1 branch remains the strongest retrieval branch in this repository. The new task-2 branch does **not** attempt to decode a blurry image directly from EEG. Instead, it uses the trained EEG retrieval model to:

- retrieve semantically similar training images,
- aggregate retrieved training classes,
- fill a fixed prompt template with the selected training class,
- use the top retrieved training image as the IP-Adapter reference image,
- generate the final reconstruction with Stable Diffusion v1.5 + IP-Adapter.

This design follows the practical assumption that a semantically nearby training class can still provide useful prompt guidance even when the exact test class does not appear in the training set.

## Method

### Task 1

The task-1 implementation keeps the original VisualEEGDecoding idea [1]:

1. `scripts/prepare_course_data.py` maps the course dataset into `data/things-eeg/`.
2. `preprocess/process_image_course.py` encodes training and test images with OpenCLIP RN50 at 12 Gaussian blur levels.
3. `main_eeg_course.py` trains the EEG encoder on subject `sub-01`.
4. The image branch fuses the 12 blur-level RN50 embeddings and the EEG branch predicts a matching 1024-dimensional embedding.
5. Training uses bidirectional contrastive supervision in the CLIP image space [3, 4].

### Task 2

The task-2 implementation adds a retrieval-augmented reconstruction branch:

1. `scripts/train_task2_semantic.py` fine-tunes the task-1 EEG encoder with **joint image loss + class-text prototype loss**.
2. Training class prototypes are encoded with the same OpenCLIP RN50 text encoder used for task-1 image features, so no extra output head is needed.
3. At inference time, each test EEG retrieves top-k training images in the learned embedding space.
4. Retrieved images are aggregated by class. The highest-scoring class becomes the prompt class.
5. The fixed prompt template is:
   - `a realistic photo of a {class_name}`
6. The highest-scoring retrieved image is used as the IP-Adapter reference image.
7. `scripts/generate_task2_reconstructions.py` generates reconstructed images with Stable Diffusion v1.5 + IP-Adapter.
8. `scripts/evaluate_task2_reconstruction.py` evaluates reconstructions with course-style `SSIM` and `CLIP` metrics.

The first implementation intentionally uses **IP-Adapter**, not `T2I-Adapter`, because the available condition is a retrieved similar image rather than a reliable EEG-derived edge/depth/segmentation map.

## Environment

Recommended Python version: **Python 3.10**.

Create and activate an environment:

```bash
conda create -n ved python=3.10 -y
conda activate ved
pip install -r requirements.txt
```

The existing `test` conda environment on the HPC may also work if it already contains compatible `torch`, `open-clip-torch`, `diffusers`, `transformers`, and `scikit-image`.

## Data and Model Preparation

The course data root should contain:

```text
image-eeg-data/
  training_images/
  test_images/
  converted_for_cogcappro/ThingsEEG/Preprocessed_data_250Hz_whiten/sub-01/train.pt
  converted_for_cogcappro/ThingsEEG/Preprocessed_data_250Hz_whiten/sub-01/test.pt
```

Required local model assets:

- OpenCLIP RN50 checkpoint
- Stable Diffusion v1.5
- IP-Adapter SD1.5 weights

Expected local paths in the current implementation:

```text
/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/open_clip_pytorch_model.bin
/hpc2hdd/home/ckwong627/workdir/models/stable-diffusion-v1-5
/hpc2hdd/home/ckwong627/workdir/models/IP-Adapter
```

If a required model is missing, create a folder under `/hpc2hdd/home/ckwong627/workdir/models/` first, then download it with:

```bash
mkdir -p /hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai
hf download timm/resnet50_clip.openai \
  --include open_clip_pytorch_model.bin \
  --local-dir /hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai

mkdir -p /hpc2hdd/home/ckwong627/workdir/models/stable-diffusion-v1-5
hf download runwayml/stable-diffusion-v1-5 \
  --local-dir /hpc2hdd/home/ckwong627/workdir/models/stable-diffusion-v1-5

mkdir -p /hpc2hdd/home/ckwong627/workdir/models/IP-Adapter
hf download h94/IP-Adapter \
  --local-dir /hpc2hdd/home/ckwong627/workdir/models/IP-Adapter
```

Approximate storage and runtime notes:

- OpenCLIP RN50: `open_clip_pytorch_model.bin` is about 0.4 GB and is mainly used for feature/prototype extraction
- Stable Diffusion v1.5: the full diffusers folder is usually about 4 to 7 GB
- IP-Adapter weights + image encoder: the full directory is usually about 3 to 5 GB
- Task-2 generation requires one GPU; the `debug` partition is suitable for smoke tests only

## Commands

### Task 1: one-command retrieval pipeline

```bash
python scripts/run_course_pipeline.py \
  --data_root /path/to/image-eeg-data \
  --clip_checkpoint /path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin
```

On an A800-style machine where GPU jobs can be launched directly with `python`, prefer:

```bash
bash run_task1_direct.sh
```

### Task 2: one-command reconstruction pipeline

```bash
python scripts/run_task2_pipeline.py \
  --data_root /path/to/image-eeg-data \
  --clip_checkpoint /path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin \
  --task1_ckpt /path/to/task1_select_checkpoint.pth
```

On an A800-style machine where GPU jobs can be launched directly with `python`, prefer:

```bash
bash run_task2_direct.sh
```

This task-2 command will:

1. refresh the local course-data symlinks,
2. fine-tune the task-1 retrieval model with class-text prototype supervision,
3. generate reconstructions for each fine-tuned seed,
4. evaluate `SSIM` and `CLIP`,
5. save per-seed and summarized metrics under `output/task2/`.

### Task 2 smoke test

```bash
python scripts/run_task2_pipeline.py \
  --data_root /path/to/image-eeg-data \
  --clip_checkpoint /path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin \
  --task1_ckpt /path/to/task1_select_checkpoint.pth \
  --epoch 1 \
  --n_seeds 1 \
  --first_seed 999
```

### Qualitative grid

```bash
python scripts/make_task2_qualitative_grid.py \
  --real-root output/task2/pipeline_runs/<run>/reconstructions/seed21/ground_truth \
  --fake-root output/task2/pipeline_runs/<run>/reconstructions/seed21/generated \
  --output output/task2/pipeline_runs/<run>/qualitative_seed21.png
```

## SLURM Usage

The SLURM submission scripts are stored in:

```text
version5_VED/slurm_scripts/
```

Current scripts:

- `02_gen_blur_features.sh`
- `03_train_eeg.sh`
- `04_run_task2_smoke.sh`
- `05_run_task2_full.sh`

Two non-SLURM direct-run helpers are also included for machines that do not require `sbatch`:

- `run_task1_direct.sh`
- `run_task2_direct.sh`

These scripts now automatically do two things at startup:

- `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY`
- call `unclash` when that shell function exists

This avoids inheriting local proxy settings into submitted jobs, which can otherwise break Python downloads, Hugging Face access, or site-specific scheduler clients.

Default policy:

- use `debug` first for smoke tests,
- switch to `long_gpu` for the full task-2 run if `debug` time is insufficient.

If queueing is too long, check faster GPU partitions such as `emergency_gpua40` or `emergency_gpu` before editing the script.

## Output Layout

```text
output/
  Image_feature/
  logs/main_eeg_course/
  task2/
    semantic_finetune/
    reconstructions/
    pipeline_runs/
```

Important task-2 artifacts include:

- fine-tuned checkpoints
- cached class text prototypes
- cached adapted training-image banks
- generated images
- ground-truth copies
- retrieval metadata JSON
- reconstruction evaluation JSON/CSV
- qualitative grids

## Model Scores

### Task 1

Completed local task-1 run:

| Selection rule | Top-1 accuracy | Top-5 accuracy | Notes |
|---|---:|---:|---|
| Validation-selected checkpoint | 82.40% ± 2.01% | 97.80% ± 0.54% | Conservative selection |
| Best test checkpoint | 86.85% ± 0.63% | 98.10% ± 0.52% | Chosen submission result |

Validation for this run was **827-way**, and test evaluation was **200-way**.

### Task 2

Completed multi-seed task-2 run:

| Metric | Score |
|---|---:|
| SSIM | **0.2977 ± 0.0066** |
| CLIP | **0.7610 ± 0.0148** |
| Seeds | **10** |

Source of truth:

- `output/task2/pipeline_runs/2026-04-24-20-55/evaluation/task2_reconstruction_summary.json`
- `output/task2/pipeline_runs/2026-04-24-20-55/evaluation/task2_reconstruction_metrics.csv`

## Limitations

- The task-2 pipeline relies on retrieved **training** classes, not the true unseen test labels.
- This is expected to help semantic alignment more than exact spatial similarity.
- `IP-Adapter` reference quality depends on retrieval quality; poor retrieval will directly hurt generation.
- The current task-2 implementation uses a fixed prompt template and a single retrieved reference image.
- `T2I-Adapter`, free-form prompt generation, and multi-reference fusion are intentionally left out of the first implementation.

## References

[1] W. Liu, H. Li, Z. Xu, L. Ma, and H. Li, "Leveraging Visual Blur Perception Characteristics for EEG Decoding," *Proceedings of the AAAI Conference on Artificial Intelligence*, 40(21), 17580-17588, 2026. Local paper copy: [`../references/paper/Liu 等 - 2026 - Leveraging Visual Blur Perception Characteristics for EEG Decoding.pdf`](../references/paper/Liu%20等%20-%202026%20-%20Leveraging%20Visual%20Blur%20Perception%20Characteristics%20for%20EEG%20Decoding.pdf).

[2] A. T. Gifford, K. Dwivedi, G. Roig, and R. M. Cichy, "A large and rich EEG dataset for modeling human visual object recognition," *NeuroImage*, 2022. THINGS-EEG project page: <https://osf.io/b83fj/>.

[3] A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," *ICML*, 2021. <https://arxiv.org/abs/2103.00020>.

[4] G. Ilharco et al., "OpenCLIP," 2021. <https://github.com/mlfoundations/open_clip>.
