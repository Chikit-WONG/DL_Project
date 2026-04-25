# DSAA2012 Deep Learning Final Project: EEG-to-Image Decoding

[中文 README](README-CN.md)

This repository contains the DSAA2012 final project work for EEG-based visual decoding on the THINGS-EEG dataset. The project studies two related tasks:

1. **Image retrieval**: map EEG signals into a visual embedding space and retrieve the correct image from a 200-class candidate pool.
2. **Image reconstruction**: generate an image conditioned on EEG-derived visual features using diffusion models and IP-Adapter-style conditioning.

The repository keeps multiple experimental versions because the project evolved through several different planning and reproduction directions. Version-specific implementation details, commands, and results are documented inside each version folder.

## Repository Layout

```text
DL_Project/
├── Final_Project_Instructions/   # Course project PDFs
├── image-eeg-data/               # Local THINGS-EEG data, ignored by git
├── plan/                         # Project planning notes, Markdown files are tracked
├── references/                   # Papers and local reference material
├── sample_codes/                 # Original sample notebooks
├── version1/                     # Original planned baseline
├── version2/                     # ChatGPT/Claude/Gemini co-designed plan, poor results
├── version3_ATM/                 # Reproduction of EEG_Image_decode (ATM)
├── version4_CCP/                 # Reproduction of CognitionCapturerPro (CCP)
└── version5_VED/                 # Reproduction/adaptation of VisualEEGDecoding
```

## Version Overview

| Version | Main idea | Current role | Key local results |
|---|---|---|---|
| [`version1`](version1/README.md) | The original project plan: custom EEG encoder aligned to CLIP ViT-H/14, then SD v1.5 + IP-Adapter reconstruction | Baseline and reference implementation | Full rerun: Top-1 24.5%, Top-5 53.0%, SSIM 0.2633, CLIP 0.7836 |
| [`version2`](version2/README.md) | A plan designed through cross-discussion among ChatGPT, Claude, and Gemini, with a stronger dual-path encoder and multi-target supervision | Exploratory attempt; the final result was poor and did not meet the intended target | Full rerun: Top-1 20.0%, Top-5 50.5%, SSIM 0.3753, CLIP 0.2755 |
| [`version3_ATM`](version3_ATM/README.md) | Reproduction and adaptation of `EEG_Image_decode`, based on the ATM/ATMS approach | Strong retrieval branch with complete evaluation scripts | Full rerun: Top-1 33.5%, Top-5 63.5%, SSIM 0.2709, CLIP 0.6089 |
| [`version4_CCP`](version4_CCP/README.md) | Reproduction and adaptation of `CognitionCapturerPro` (CCP), including multimodal embeddings, alignment, and SDXL-Turbo generation | Final large-scale CCP reconstruction/adaptation branch | Full rerun: Any-modality Top-1 61.5%, Top-5 89.0%; reconstruction (`all`) SSIM 0.3732, CLIP 0.8981 |
| [`version5_VED`](version5_VED/README.md) | Reproduction and course adaptation of `VisualEEGDecoding`, using multi-blur OpenCLIP RN50 visual features for task 1 and retrieval-augmented IP-Adapter reconstruction for task 2 | Current best retrieval branch, now extended into a complete task1/task2 pipeline for direct Python runs on an A800-style HPC | Task 1: chosen submission score Top-1 86.85% ± 0.63%, Top-5 98.10% ± 0.52%; task 2 pipeline implemented with semantic fine-tuning, fixed prompt class retrieval, and SD v1.5 + IP-Adapter generation |

## Data and Model Artifacts

The raw and converted datasets are intentionally not tracked by git. The expected local data root is:

```text
image-eeg-data/
├── train.pt
├── test.pt
├── EEG_CHANNELS.jsonl
├── training_images/
└── test_images/
```

Some branches also create converted data under `image-eeg-data/converted_for_cogcappro/`.

Pretrained models are expected to be stored outside git, for example under:

```text
/hpc2hdd/home/ckwong627/workdir/models/
```

Typical required model assets include CLIP ViT-H/14, OpenCLIP RN50, Stable Diffusion v1.5, SDXL-Turbo, and IP-Adapter weights. See the version-specific READMEs and config files for exact paths.

## Results and Tracked Outputs

The repository is configured to keep lightweight result summaries and selected visualization images, including:

- metrics JSON/CSV files under version output/result folders;
- selected Task 2 montage or comparison images;
- Markdown result summaries and planning notes.

Large generated artifacts are ignored, including model checkpoints, `.pt` tensors, logs, caches, temporary files, and local datasets.

## Git Hygiene

The top-level `.gitignore` is designed to avoid pushing large files to GitHub while still keeping useful documentation and score summaries:

- ignored: datasets, checkpoints, tensor caches, logs, scheduler output, temporary files, and bulky reference repositories;
- tracked: source code, README files, planning Markdown files under `plan/`, metrics summaries, and selected result images.

Before pushing, it is useful to run:

```bash
git status --short
git add -n README.md README-CN.md .gitignore plan/*.md
```

Use `git add -n` first when checking whether a file would be staged without actually staging it.

## Suggested Reading Order

1. Start with this root README for the project map.
2. Read [`version1/README.md`](version1/README.md) for the original planned baseline.
3. Read [`version3_ATM/README.md`](version3_ATM/README.md) and [`version4_CCP/README.md`](version4_CCP/README.md) for the two reconstruction-oriented reproduction branches.
4. Read [`version5_VED/README.md`](version5_VED/README.md) for the strongest VisualEEGDecoding branch, including the new task-2 retrieval-augmented reconstruction pipeline and the A800/HPC run commands.
5. Check [`plan/`](plan/) for planning history and implementation decisions.
