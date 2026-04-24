# Continue VisualEEGDecoding Reproduction Plan

## Summary

This plan continues from `Handoff_to_Codex.md`. Claude's course-adapted scripts are present, but image feature generation failed because `process_image_course.py` used `pretrained='openai'`, causing OpenCLIP to contact `hf-mirror.com` on the GPU node. The required RN50 weights already exist locally.

Current confirmed state:

- Course EEG and image directories are symlinked into `data/things-eeg/`.
- Local RN50 weights exist at `/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/open_clip_pytorch_model.bin`.
- Job `9703515` produced no feature files; `data/things-eeg/Image_feature/` is empty.
- `open_clip.create_model_and_transforms('RN50', pretrained=<local_bin>)` loads successfully.

## Key Changes

- Update `preprocess/process_image_course.py` to load the local RN50 checkpoint directly, avoiding all external network access during SLURM execution.
- Add `set -eo pipefail` to `slurm_scripts/02_gen_blur_features.sh` and `slurm_scripts/03_train_eeg.sh` so Python failures make the SLURM job fail visibly without breaking conda activation scripts that reference optional environment variables.
- Keep the existing symlink-based data setup; no EEG conversion is required because the `img` paths already match VisualEEGDecoding feature keys.
- Filter training EEG samples to those with precomputed image features, because `train.pt` contains more image references than the provided course image directory.
- Avoid running full `scripts/inspect_course_data.py` on the login node because it loads large tensors and was killed; use lightweight checks or SLURM jobs for heavy inspection.

## Execution Steps

1. Patch the feature-generation script and SLURM scripts.
2. Run a local checkpoint-load smoke test:
   `conda run -n test python -c "import open_clip; open_clip.create_model_and_transforms('RN50', pretrained='<local_bin>')"`
3. Submit image feature generation:
   `sbatch slurm_scripts/02_gen_blur_features.sh`
4. After completion, validate:
   - `data/things-eeg/Image_feature/MultiBlur_RN50_train.pt`
   - `data/things-eeg/Image_feature/MultiBlur_RN50_test.pt`
   - 12 blur keys: `1, 3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63`
   - sample feature tensor shape `[1024]`
5. Submit EEG training only after feature validation:
   `sbatch slurm_scripts/03_train_eeg.sh`
6. Run a debug smoke job with `--epoch 1 --n_seeds 1` if the full job is queued, and confirm the filter reports dropped train samples but no dropped test samples.
7. Monitor training logs until seeds `21-30` complete.
8. Summarize course metrics:
   `conda run -n test python scripts/evaluate_course_metrics.py`

## Test Plan

- Confirm the feature-generation log contains no OpenCLIP download attempts or proxy errors.
- Confirm feature files load with `torch.load(..., weights_only=False)`.
- Confirm train/test feature dictionaries have all 12 blur levels and usable image-path keys.
- Run a smoke training job with `--epoch 1 --n_seeds 1` before relying on the queued full 10-seed job.
- Confirm `all_metrics.csv` has 10 rows and includes `test_top1_acc` and `test_top5_acc`.

## Assumptions

- Use the existing `test` conda environment.
- Use only `sub-01` with intra-subject training.
- Keep paper hyperparameters unless runtime or memory failures require adjustment.
- If the `debug` partition times out during feature generation, switch `02_gen_blur_features.sh` to `i64m1tga40u` and rerun.
- Use `long_gpu` for full training because the original `i64m1tga40u` queue was estimated to start later, while `long_gpu` has sufficient A800 resources and a shorter estimated wait.
