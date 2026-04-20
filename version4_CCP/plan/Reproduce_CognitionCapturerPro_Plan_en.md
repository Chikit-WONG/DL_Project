# CognitionCapturerPro Reproduction Plan for the Course Dataset

## Goal
- Run `CognitionCapturerPro` on the fixed course dataset `image-eeg-data`
- Prioritize `Brain-to-Image Retrieval` and `Brain-to-Image Reconstruction`
- Produce the required course metrics and keep the workflow reproducible with scripts

## Execution Path
1. Reuse the existing `test` conda environment and record the version drift: `Python 3.10 / torch 2.10`
2. Adapt the course dataset into the `ThingsEEG` layout expected by CogCapPro
3. Generate `Image_depth_set_Resize` and `Image_edge_set_Resize` from the released course images
4. Patch the EEG loader so it consumes the real `train.pt/test.pt` fields and infers the trial count from file contents
5. Reuse the existing local `sdxl-turbo`, `IP-Adapter`, and `OpenCLIP` weights for training, alignment, and generation
6. Implement a reconstruction evaluation script that follows the same metric definitions as the course sample code

## Key Defaults
- Fixed subject: `sub-01`
- Trial counts are inferred from file shapes instead of hard-coded constants
- The text branch falls back to the released course `text` field when private BLIP2 files are not available
- The main run uses `ViT-H-14` because its weights already exist locally; `RN50` can be added later as a comparison
- A smoke run is executed first to guarantee end-to-end viability before longer training

## Evaluation
- Retrieval: `Top-1`, `Top-5`
- Reconstruction:
  - Required: `SSIM`, `CLIP`
  - Extended: `PixCorr`, `AlexNet2`, `AlexNet5`, `Inception`, `EffNe0`, `SwAV`

## Deliverables
- `configs/local.yaml`
- `scripts/prepare_course_data.py`
- `scripts/prepare_diffusion_embeddings.py`
- `scripts/evaluate_reconstruction.py`
- `slurm_scripts/*.sh`
- Training, alignment, reconstruction, and evaluation outputs under `runs/...`
