# version5_VED: VisualEEGDecoding Course Adaptation

[中文 README](README-CN.md)

This folder is the course-adapted `VisualEEGDecoding` branch for the DSAA2012 final project. It adapts the visual-blur EEG decoding idea from Liu et al. to the local course data format and focuses on the image retrieval task rather than diffusion-based reconstruction [1].

The implementation is intended to be portable to another GPU server, including an A800 machine where jobs can be launched directly with `python` instead of `sbatch`. All generated features, logs, checkpoints, and metric files are written under `output/` so the run artifacts can be copied back with one `rsync` command.

## Project Summary

The task is EEG-to-image retrieval on THINGS-style visual EEG data. Given one EEG response, the model predicts an embedding that should match the corresponding image embedding and rank the true image highly among 200 test candidates. The dataset and task follow the general visual decoding setting used by THINGS-EEG work [2], while the image embedding space uses CLIP/OpenCLIP-style visual representations [3, 4].

This version is not an image generation pipeline. It does not run Stable Diffusion, SDXL-Turbo, or IP-Adapter. It is a retrieval-only branch built to reproduce and adapt the strongest VisualEEGDecoding result path within the course project constraints.

## Method

The original VisualEEGDecoding paper argues that blur-aware visual features provide useful supervision for EEG decoding [1]. This course branch keeps that core idea and makes the following local adaptations:

1. `scripts/prepare_course_data.py` maps the course dataset into the expected `data/things-eeg/` structure by creating symbolic links.
2. `preprocess/process_image_course.py` encodes training and test images with OpenCLIP RN50 and 12 Gaussian blur levels.
3. `main_eeg_course.py` trains the EEG encoder on subject `sub-01` using the course-provided 250 Hz whitened EEG tensors.
4. The EEG branch maps 63-channel EEG segments to a 1024-dimensional representation. The image branch learns a weighted fusion over the 12 blur-level RN50 features.
5. Training uses a bidirectional CLIP-style contrastive loss so the correct EEG-image pair has higher similarity than mismatched pairs [3].
6. Validation is run as a full validation-set retrieval task. In the completed local run the split was `train=15713`, `val=827`, `test=200`, so each validation epoch was explicitly **827-way** and each test evaluation was **200-way**. Future runs print the current `VAL=<N>-way` and `TEST=<N>-way` values in the log because these numbers depend on the available image-feature matches.

The official course metric reported below uses the validation-selected checkpoint for each seed. The `best_test` numbers are included only as a reference because selecting by test accuracy is optimistic.

## Environment

Recommended Python version: **Python 3.10**.

Create and activate an environment:

```bash
conda create -n ved python=3.10 -y
conda activate ved
pip install -r requirements.txt
```

If the target HPC has a site-specific CUDA/PyTorch installation command, install `torch` and `torchvision` according to that machine first, then run:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains the project-level Python dependencies, including `open-clip-torch`, `opencv-python-headless`, `numpy`, `pandas`, `scipy`, `einops`, `scikit-learn`, `mne`, and `matplotlib`.

## Data and Model Preparation

Do not commit data, model weights, generated `.pt` features, logs, or checkpoints to GitHub.

The course data root passed to the script should contain:

```text
image-eeg-data/
  training_images/
  test_images/
  converted_for_cogcappro/ThingsEEG/Preprocessed_data_250Hz_whiten/sub-01/train.pt
  converted_for_cogcappro/ThingsEEG/Preprocessed_data_250Hz_whiten/sub-01/test.pt
```

Download the OpenCLIP RN50 checkpoint on a machine with internet access:

```bash
python scripts/download_rn50.py --save_dir /path/to/CLIP-RN50-openai
```

The command saves:

```text
/path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin
```

If the target HPC cannot access the internet, download the file elsewhere and copy the whole `CLIP-RN50-openai/` folder to the target machine.

## One-Command Run

From this folder:

```bash
cd /path/to/version5_VED
```

Run the full pipeline:

```bash
python scripts/run_course_pipeline.py --data_root /path/to/image-eeg-data --clip_checkpoint /path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin
```

This command prepares the local data links, creates multi-blur RN50 image features, trains 10 seeds by default, and writes all outputs under `output/`.

For a quick smoke test:

```bash
python scripts/run_course_pipeline.py --data_root /path/to/image-eeg-data --clip_checkpoint /path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin --epoch 1 --n_seeds 1 --first_seed 999
```

If image features already exist:

```bash
python scripts/run_course_pipeline.py --data_root /path/to/image-eeg-data --clip_checkpoint /path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin --skip_features
```

During training, the log lines explicitly show the retrieval-way setting, for example `VAL (827-way)` and `TEST (200-way)` for the completed local run.

After training, summarize the course metrics:

```bash
python scripts/evaluate_course_metrics.py
```

## Output Layout

All runtime artifacts are grouped under `output/`:

```text
output/
  Image_feature/
    MultiBlur_RN50_train.pt
    MultiBlur_RN50_test.pt
  logs/main_eeg_course/Brain_Visual_Encoder_EEG/<timestamp>/
    all_metrics.csv
    *.log
    *.pth
```

To copy results back from the target HPC:

```bash
rsync -avP /path/to/version5_VED/output/ your_hpc:/path/to/save/version5_VED_output/
```

## Model Scores

Local run: 10 seeds, `21` to `30`; validation checkpoint selection used 827-way retrieval, and final reporting used 200-way retrieval on the course test split.

| Selection rule | Top-1 accuracy | Top-5 accuracy | Notes |
|---|---:|---:|---|
| Validation-selected checkpoint | 82.40% ± 2.01% | 97.80% ± 0.54% | Main course result |
| Best test checkpoint | 86.85% ± 0.63% | 98.10% ± 0.52% | Reference only; optimistic selection |

The metric summary was generated at:

```text
output/course_metrics_summary.json
```

## Limitations

- This branch performs retrieval only; it does not reconstruct images.
- The local run uses the course `sub-01` data only, so it is not a full 10-subject reproduction of the original paper [1].
- The result depends on complete availability of the training and test images. Missing image files will remove samples during feature matching and can make the run invalid.
- The `best_test` numbers should not be used as the primary score because they select checkpoints using test performance.
- The OpenCLIP RN50 feature generation step is storage- and GPU-heavy; generated `.pt` feature files are intentionally ignored by git.

## References

[1] W. Liu, H. Li, Z. Xu, L. Ma, and H. Li, "Leveraging Visual Blur Perception Characteristics for EEG Decoding," *Proceedings of the AAAI Conference on Artificial Intelligence*, 40(21), 17580-17588, 2026. Local paper copy: [`../references/paper/Liu 等 - 2026 - Leveraging Visual Blur Perception Characteristics for EEG Decoding.pdf`](../references/paper/Liu%20等%20-%202026%20-%20Leveraging%20Visual%20Blur%20Perception%20Characteristics%20for%20EEG%20Decoding.pdf).

[2] A. T. Gifford, K. Dwivedi, G. Roig, and R. M. Cichy, "A large and rich EEG dataset for modeling human visual object recognition," *NeuroImage*, 2022. THINGS-EEG project page: <https://osf.io/b83fj/>.

[3] A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," *ICML*, 2021. <https://arxiv.org/abs/2103.00020>.

[4] G. Ilharco et al., "OpenCLIP," 2021. <https://github.com/mlfoundations/open_clip>.
