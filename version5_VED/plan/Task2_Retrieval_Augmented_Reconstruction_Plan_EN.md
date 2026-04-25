# Version5_VED Task 2 Plan: Retrieval-Augmented Reconstruction

## Summary

Implement task 2 on top of the existing `version5_VED` task-1 retrieval model instead of replacing it. The final system should:

1. keep the current blur-aware EEG retrieval backbone,
2. add class-text prototype supervision in the same OpenCLIP RN50 space,
3. retrieve top-k training images for each test EEG,
4. aggregate retrieved classes to choose one prompt class,
5. generate images with a fixed prompt template and IP-Adapter reference image,
6. evaluate outputs with course-style `SSIM` and `CLIP`.

The implementation must be reproducible, output-folder-based, and runnable on the course HPC.

## Core Design

### Retrieval-side changes

- Reuse the current `Brain_Visual_Encoder_EEG` backbone and image-alignment branch.
- Do not add a separate text head; use the same 1024-dimensional OpenCLIP RN50 space.
- Build one text prototype per training class with:
  - `a realistic photo of a {class_name}`
- Fine-tune from a task-1 checkpoint with:
  - `L_total = 0.7 * L_image + 0.3 * L_class`
- Select the best task-2 checkpoint by validation prompt-class retrieval quality.

### Prompt and retrieval logic

- Retrieve top-k training images for each EEG query.
- Default `top-k = 20`.
- Aggregate scores by class.
- Use the highest-scoring class as the prompt class.
- Use the highest-scoring retrieved image as the IP-Adapter reference image.
- Keep the prompt template fixed:
  - `a realistic photo of a {class_name}`

### Generation path

- Use Stable Diffusion v1.5 + IP-Adapter as the first implementation.
- Conditioning:
  - text: selected prompt class
  - image: top-1 retrieved training image
- Do not implement `T2I-Adapter`, free-form prompt generation, or multi-reference fusion in v1.

## Required Outputs

- Fine-tuned task-2 checkpoints
- Cached class text prototypes
- Cached adapted training-image bank
- Generated test images
- Ground-truth image copies
- Per-sample retrieval metadata
- Per-seed reconstruction evaluation JSON
- Mean ± std summary across seeds
- 8–12 qualitative examples or a qualitative grid

All outputs must live under `version5_VED/output/`.

## Test Plan

- Verify class-text prototype generation produces one prototype per training class.
- Verify task-2 fine-tuning loads the task-1 checkpoint without shape mismatch.
- Verify retrieval metadata includes:
  - prompt class,
  - top retrieved image,
  - top retrieved classes,
  - ground-truth image path.
- Verify a smoke run with `--epoch 1 --n_seeds 1` completes.
- Verify the final evaluation JSON includes at least:
  - `eval_ssim`
  - `eval_clip`

## Assumptions

- Subject remains `sub-01`.
- Task 1 checkpoint is already available.
- The first implementation prioritizes semantic improvement and course reproducibility over architectural novelty.
- `IP-Adapter` is the primary adapter because the available conditioning signal is a retrieved similar image, not a reliable EEG-derived control map.
