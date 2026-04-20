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
└── version4_CCP/                 # Reproduction of CognitionCapturerPro (CCP)
```

## Version Overview

| Version | Main idea | Current role | Key local results |
|---|---|---|---|
| [`version1`](version1/README.md) | The original project plan: custom EEG encoder aligned to CLIP ViT-H/14, then SD v1.5 + IP-Adapter reconstruction | Baseline and reference implementation | Joint baseline: Top-1 13.5%, Top-5 36.5%, SSIM 0.276, CLIP 0.708 |
| [`version2`](version2/README.md) | A plan designed through cross-discussion among ChatGPT, Claude, and Gemini, with a stronger dual-path encoder and multi-target supervision | Exploratory attempt; the final result was poor and did not meet the intended target | `v2_final`: Top-1 15.0%, Top-5 35.0%, SSIM 0.3709, CLIP 0.2779 |
| [`version3_ATM`](version3_ATM/README.md) | Reproduction and adaptation of `EEG_Image_decode`, based on the ATM/ATMS approach | Strong retrieval branch with complete evaluation scripts | Top-1 29.0%, Top-5 62.0%, SSIM 0.2852, CLIP 0.6696 |
| [`version4_CCP`](version4_CCP/README.md) | Reproduction and adaptation of `CognitionCapturerPro` (CCP), including multimodal embeddings, alignment, and SDXL-Turbo generation | Final large-scale CCP reproduction/adaptation branch | Fusion Top-1 31.5%, Top-5 64.5%; reconstruction SSIM 0.2254, CLIP 0.5169 |

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

Typical required model assets include CLIP ViT-H/14, Stable Diffusion v1.5, SDXL-Turbo, and IP-Adapter weights. See the version-specific READMEs and config files for exact paths.

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
3. Read [`version3_ATM/README.md`](version3_ATM/README.md) and [`version4_CCP/README.md`](version4_CCP/README.md) for the two reproduction branches.
4. Check [`plan/`](plan/) for planning history and implementation decisions.
