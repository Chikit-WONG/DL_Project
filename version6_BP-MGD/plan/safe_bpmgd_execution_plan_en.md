# Safe BP-MGD Execution Plan

## Summary

Implement a leakage-safe Task 2 EEG-to-image reconstruction pipeline in `version6_BP-MGD`. Development uses a train-only validation split for all model and generation choices. The final reported test score must come from a model retrained on the complete training split with fixed hyperparameters.

## Implementation

- Build the package, configs, scripts, Slurm jobs, output folders, checkpoint folders, and feature caches.
- Implement `LeakageGuard` first and call it from feature caching, memory bank building, generation, reranking, and evaluation wrappers.
- Load only `image-eeg-data/train.pt` and `image-eeg-data/test.pt`; enforce EEG shape `[63, 250]`; enforce `avg_trials=True` for test.
- Cache CLIP and multiblur features from training images only, then build `prototype_bank.pt` from train-only records.
- Train the ATM-style EEG encoder with semantic and multiblur losses first, then structural fallback losses, then the MLP prior mapper.
- Generate test reconstructions from test EEG, trained checkpoints, frozen pretrained models, and train-only prototypes only.
- Evaluate only after generation; test GT is used only by evaluation and qualitative grid code.

## Final Full-Train Rule

After dev validation selects the final config, rerun cache, encoder training, prior training, and prototype bank construction with `--mode full_train`. Only then generate test images and run Task 2 evaluation.

## Acceptance

- `outputs/<run>/leakage_report.txt` states no test image access during generation or reranking.
- `outputs/<run>/metrics.json` contains SSIM and CLIP score when CLIP dependencies are available.
- `outputs/<run>/qualitative_grid.png` contains 8-12 GT/reconstruction pairs for reporting.
