# Full-Training-Set Rerun Plan for `version1`-`version4`

## Summary

The goal is to rerun `version1`, `version2`, `version3_ATM`, and `version4_CCP` with the now-complete training set, then regenerate retrieval and reconstruction metrics under the course protocol. `version5_VED` is out of scope.

This rerun is defined as:

- rebuild any training-dependent caches that may have been created from an incomplete train split
- retrain the intended model pipeline for each version
- rerun reconstruction or generation where required
- rerun official or course-aligned evaluation scripts
- compare the refreshed scores against the previously reported local results

## Implementation Changes

- Add bilingual plan files in `plan/` for this task.
- Normalize the formal rerun entrypoints so they point to the intended working directory and full-run SLURM resources.
- Keep smoke-test and debug entrypoints intact where they are still useful, but stop using them as the default documentation path for final results.
- For `version3_ATM`, refresh stable `LATEST_RETRIEVAL` and `LATEST_RECONSTRUCTION` symlinks after each training job so follow-up evaluation scripts can run without manual timestamp editing.
- For `version4_CCP`, expose a non-debug full evaluation entrypoint for evaluating both `all_before` and `all` generation modes in one run.

## Version-Specific Execution

### `version1`

- Rebuild CLIP caches from the full shared dataset.
- Retrain `phase1_main` and `phase2_main`.
- Re-run reconstruction with all 10 seeds.
- Re-run evaluation to regenerate `metrics_phase2_main_best.json`.
- Use the `version1` directory as the SLURM working directory so `codes/config.py` resolves paths correctly.

### `version2`

- Reuse the existing full scripts for cache, encoder warmup, multitarget, finetune, prior, reconstruction, and evaluation.
- Ensure none of the formal scripts pass `--limit`, `--limit_train`, or `--limit_test`.
- Keep the `compare_v1` summary path so the refreshed `version1` result is incorporated automatically.

### `version3_ATM`

- Re-run retrieval training and reconstruction training with the full course dataset.
- Refresh `LATEST_RETRIEVAL` and `LATEST_RECONSTRUCTION` symlinks after the corresponding jobs.
- Re-run retrieval evaluation, image generation, and reconstruction metric evaluation against those stable symlinks.

### `version4_CCP`

- Re-run diffusion embedding preparation, retrieval training, alignment, image generation, reconstruction evaluation, and summary generation.
- Keep `all_before` as the recommended reconstruction mode, but continue to generate and evaluate both `all_before` and `all`.
- Use the fixed full-generation script and a dedicated full evaluation script rather than the old debug-named evaluation entrypoint.

## Test Plan

- Confirm `train.pt` and `test.pt` are readable and correspond to the expected course dataset.
- Reuse the `test` conda environment unless a concrete dependency failure is observed.
- Run smoke or env checks before long jobs when a version already provides them.
- Verify each version produces fresh checkpoints, evaluation outputs, and final score files without falling back to debug/sample limits.
- Produce a cross-version rerun summary containing at least:
  - Top-1 / Top-5 for retrieval
  - SSIM / CLIP Score for reconstruction
  - output file paths for the refreshed results

## Assumptions

- The task requires full retraining plus reevaluation, not checkpoint-only rescoring.
- Only changes required for full-dataset reruns should be made; no model redesign is part of this task.
- Queue selection may change at submission time if a faster eligible partition is available.
