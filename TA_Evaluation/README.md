# TA Evaluation Code

These codes are for evaluation.

## Task 1

The "**task1_eval.py**" takes in generated similarity matrices and calculate the metrics.  

For fast evaluation:  
```cmd
python ./TA_Evaluation/task1_eval.py --run_dir ./task1/output/main_results

python ./TA_Evaluation/task1_eval.py --run_dir ./task1/output/main_results --checkpoint best
```

The parameter "**--checkpoint**" chooses which class of result to use, either "**select**" (default, use the final epoch or validation selected (if capable) to give the result) or "**best**" (use the test-selected matrices to give the result, **for reference only**).  

The parameter "**--run_dir**" is where the matrix results are located. Usually in "../task1/output/",  
but you can directly find the reported results in "./result_task1/"   
Do mind that it may load matrices from other experiments, if multiple experiments' results are in the same folder. **We recommend pointing out the specific folder of the experiment in the "--run_dir"**  

## Task 2

The "**task2_eval.ipynb**" takes in generated (reconstructed) images and evaluates reconstruction metrics using TA's official `eval_images(...)` function.  

This notebook does **NOT** run the Task 2 pipeline — it only evaluates pre-generated images.

### Prerequisites

1. **Generated images**: Must already exist under `task2/runs/{RUN_TAG}/{EXP_NAME}/{SUBJECT}_seed{seed}/generated_image/{MODE}/`.  
   These are produced by running the Task 2 pipeline (`task2/scripts/run_full_experiment.sh`).

2. **Real test images**: Expected at `image-eeg-data/test_images/` (or `image-eeg-data/converted_for_cogcappro/ThingsEEG/Image_set_Resize/test_images/`).  
   Paired with generated images by filename.

### Configuration (Cell 5)

Modify the following variables in the 5th code cell to match your setup:

| Variable | Description | Default |
|----------|-------------|---------|
| `TASK2_RUN_TAG` | Run tag used when running the pipeline | `"full_tune"` |
| `TASK2_EXP_NAME` | Experiment name from your pipeline config | `"intra-subject_cogcappro_..."` |
| `TASK2_SUBJECT` | Subject ID | `"sub-01"` |
| `TASK2_SEEDS` | List of seeds to evaluate | `list(range(10))` |
| `TASK2_GENERATED_MODE` | Subfolder under `generated_image/` | `"all"` |
| `REAL_IMAGE_ROOT` | Path to ground-truth test images | `image-eeg-data/test_images/` |

### Output

The notebook calls TA's `eval_images(...)` for each seed and reports **mean ± std** over seeds for:

- `eval_pixcorr` — Pixel-wise correlation
- `eval_ssim` — Structural Similarity
- `eval_alex2` / `eval_alex5` — AlexNet layer 2/5 identification
- `eval_inception` — Inception-v3 identification
- `eval_clip` — CLIP (ViT-L/14) identification
- `eval_effnet` — EfficientNet-B1 correlation
- `eval_swav` — SwAV correlation