# Version 2 Implementation Plan: THINGS-EEG Retrieval and Reconstruction Upgrade

## Summary
- Build a fully independent `version2` pipeline on top of the working `version1` baseline without changing `version1` behavior or outputs.
- The shared dataset root is fixed to:
  - `/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data`
- The mainline architecture is fixed to: `enhanced EEG encoder + multi-target visual alignment + prior diffusion + single-route IP-Adapter + img2img + SDXL-Turbo`.
- Deliverables include code, SLURM scripts, logs, results, Task 2 montages, and bilingual plan/result reports.

## Key Changes
- `version2/codes/config.py` is the single source of truth and hardcodes the shared dataset directory.
- `version2` never copies the dataset; all training, evaluation, and reconstruction jobs read directly from the shared directory.
- `version2/codes` contains these entry points:
  - `config.py`
  - `data.py`
  - `cache_backbone_features.py`
  - `model.py`
  - `train_encoder.py`
  - `train_prior.py`
  - `reconstruct.py`
  - `evaluate.py`
  - `make_task2_montage.py`
  - `summarize_results.py`
- The model is fixed to:
  - `Semantic head`: primary target `CLIP ViT-H/14`, auxiliary targets `ViT-B/32` and `RN50`
  - `Structural head`: regression target `SD VAE latent`
  - Three-stage schedule:
    - `warmup`: `H14 InfoNCE + 0.5*MSE`
    - `multitarget`: add `B32/RN50 InfoNCE + SmoothL1(VAE latent)`
    - `finetune`: add `hard-negative InfoNCE + supervised contrastive`

## Execution Steps
- Milestone 0: baseline lock-in and plan docs
- Milestone 1: cache `H14/B32/RN50/VAE latent`
- Milestone 2: train the upgraded EEG encoder
- Milestone 3: train prior diffusion
- Milestone 4: connect `Prior + single IP-Adapter + img2img`
- Milestone 5: evaluate, build Task 2 montages, and export bilingual summaries

## Notes
- Default conda environment: `test`
- Default queue for smoke tests: `debug`
- If `SDXL-Turbo` is not stable on day 1-2, the generation branch may temporarily fall back while keeping the encoder/prior mainline moving
