# Safe BP-MGD for Project1 Task 2

Safe BP-MGD is a leakage-safe EEG-to-image reconstruction pipeline for DSAA2012 Project1 Task 2.

## Leakage-Safety Statement

This project uses a train-only prototype memory bank. No test images, test labels, or test candidate images are used for retrieval, generation, reranking, hyperparameter tuning, or prompt construction. Test ground-truth images are used only by the official evaluation code and for final qualitative visualization.

## Environment

Use the existing `test` conda environment first:

```bash
conda activate test
cd /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version6_BP-MGD
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

Required packages include `torch`, `torchvision`, `Pillow`, `numpy`, `pyyaml`, `scikit-image`, `open_clip_torch`, and optionally `diffusers`, `transformers`, `accelerate`.

## Data And Models

Default config points to:

```text
data: /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data
CLIP RN50: /hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/open_clip_pytorch_model.bin
SDXL-Turbo: /hpc2hdd/home/ckwong627/workdir/models/sdxl-turbo
IP-Adapter: /hpc2hdd/home/ckwong627/workdir/models/IP-Adapter
```

## Development Run

Use development mode only for choosing hyperparameters from train-derived validation:

```bash
python scripts/00_check_data.py --mode dev --run-name dev_check
python scripts/01_cache_train_features.py --mode dev --run-name dev_cache --device cuda
python scripts/02_train_encoder.py --mode dev --run-name dev_encoder
python scripts/03_train_prior.py --mode dev --run-name dev_prior --encoder-ckpt checkpoints/dev_encoder/encoder_best_val.pt
```

## Final Full-Train Run

After configuration is fixed, retrain on the complete training split:

```bash
python scripts/00_check_data.py --mode full_train --run-name final_check
python scripts/01_cache_train_features.py --mode full_train --run-name final_cache --device cuda
python scripts/02_train_encoder.py --mode full_train --run-name final_encoder
python scripts/03_train_prior.py --mode full_train --run-name final_prior --encoder-ckpt checkpoints/final_encoder/encoder_final_fulltrain.pt
python scripts/04_generate_recon.py --mode full_train --run-name final_test --encoder-ckpt checkpoints/final_encoder/encoder_final_fulltrain.pt --prior-ckpt checkpoints/final_prior/prior_mapper.pt --backend prototype
python scripts/05_eval_recon.py --run-name final_test
```

Use `--backend sdxl_ipadapter` for SDXL/IP-Adapter generation after validating GPU memory and dependencies.

## Outputs

Important outputs:

```text
feature_cache/final_train/prototype_bank.pt
checkpoints/final_encoder/encoder_final_fulltrain.pt
checkpoints/final_prior/prior_mapper.pt
outputs/final_test/final_recon/
outputs/final_test/metrics.json
outputs/final_test/qualitative_grid.png
outputs/final_test/leakage_report.txt
```

## External Resources

Disclose these resources in the report if used: OpenCLIP/CLIP, SDXL-Turbo, IP-Adapter, DepthAnything, EVNet, VisualEEGDecoding, CognitionCapturerPro, and EEG_Image_decode. This repository integrates ideas from those systems; it should not be described as purely original.
