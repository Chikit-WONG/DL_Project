# version5_VED：VisualEEGDecoding 课程适配版

[English README](README.md)

本目录是 DSAA2012 期末项目中的 `VisualEEGDecoding` 课程适配分支。它基于 Liu 等人在 VisualEEGDecoding 中提出的视觉模糊感知 EEG 解码思路，并将代码改成适配本课程数据格式的版本 [1]。

这个版本的目标是方便在另一台 GPU 服务器上直接运行，例如同学的 A800 HPC。该机器不需要 `sbatch .sh` 脚本时，可以直接用 `python` 启动程序。所有生成的图像特征、日志、checkpoint 和指标都会写入 `output/`，运行结束后可以直接整体 `rsync` 回来。

## 项目介绍

本版本解决的是 EEG-to-image retrieval，也就是根据一段 EEG 信号，在 200 张候选测试图像中检索出对应图像。任务背景与 THINGS-EEG 的视觉 EEG 解码设定一致 [2]，视觉 embedding 使用 CLIP/OpenCLIP 系列模型提供的图像表征 [3, 4]。

这个版本不是图像生成管线，不会运行 Stable Diffusion、SDXL-Turbo 或 IP-Adapter。它是一个只做检索的分支，主要用于在课程数据限制下复现和适配 VisualEEGDecoding 中表现较强的路线。

## 方法介绍

VisualEEGDecoding 原论文认为，不同模糊程度的视觉特征可以为 EEG 解码提供更有效的监督 [1]。本课程版保留这个核心思路，并做了以下适配：

1. `scripts/prepare_course_data.py` 将课程数据通过符号链接整理成 `data/things-eeg/` 下的期望结构。
2. `preprocess/process_image_course.py` 使用 OpenCLIP RN50 对训练图像和测试图像提取 12 个 Gaussian blur level 的图像特征。
3. `main_eeg_course.py` 使用课程提供的 250 Hz whitened EEG tensor，在 `sub-01` 上训练 EEG encoder。
4. EEG 分支将 63 通道 EEG 片段映射到 1024 维 embedding；图像分支学习融合 12 个 blur level 的 RN50 图像特征。
5. 训练目标是 CLIP 风格的双向对比学习，让正确 EEG-image pair 的相似度高于错误配对 [3]。
6. Validation 使用完整 validation split 做检索。在已经完成的本地运行中，数据划分为 `train=15713`、`val=827`、`test=200`，所以每个 epoch 的 validation 明确是 **827-way**，test evaluation 是 **200-way**。后续运行会在日志中打印当前 `VAL=<N>-way` 和 `TEST=<N>-way`，因为这个数值会随可匹配到图像特征的样本数变化。

下面报告的正式课程指标使用每个 seed 的 validation-selected checkpoint。`best_test` 指标只作为参考，因为它使用测试集表现选 checkpoint，结果会偏乐观。

## 环境配置

建议 Python 版本：**Python 3.10**。

创建环境并安装依赖：

```bash
conda create -n ved python=3.10 -y
conda activate ved
pip install -r requirements.txt
```

如果目标 HPC 对 CUDA/PyTorch 有专门安装方式，请先按该机器的官方说明安装 `torch` 和 `torchvision`，然后再运行：

```bash
pip install -r requirements.txt
```

`requirements.txt` 中包含项目级依赖，包括 `open-clip-torch`、`opencv-python-headless`、`numpy`、`pandas`、`scipy`、`einops`、`scikit-learn`、`mne` 和 `matplotlib`。

## 数据和模型准备

不要把数据、模型权重、生成的 `.pt` 特征、日志或 checkpoint 提交到 GitHub。

传给脚本的课程数据目录需要包含：

```text
image-eeg-data/
  training_images/
  test_images/
  converted_for_cogcappro/ThingsEEG/Preprocessed_data_250Hz_whiten/sub-01/train.pt
  converted_for_cogcappro/ThingsEEG/Preprocessed_data_250Hz_whiten/sub-01/test.pt
```

在有网络的机器上下载 OpenCLIP RN50 权重：

```bash
python scripts/download_rn50.py --save_dir /path/to/CLIP-RN50-openai
```

下载后会得到：

```text
/path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin
```

如果目标 HPC 不能联网，就在其他机器下载后把整个 `CLIP-RN50-openai/` 文件夹传过去。

## 一行命令运行

进入本目录：

```bash
cd /path/to/version5_VED
```

运行完整流程：

```bash
python scripts/run_course_pipeline.py --data_root /path/to/image-eeg-data --clip_checkpoint /path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin
```

这条命令会自动准备数据链接、生成 multi-blur RN50 图像特征、默认训练 10 个随机种子，并将全部输出写入 `output/`。

第一次建议先做一个快速 smoke test：

```bash
python scripts/run_course_pipeline.py --data_root /path/to/image-eeg-data --clip_checkpoint /path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin --epoch 1 --n_seeds 1 --first_seed 999
```

如果图像特征已经生成，只想重新训练：

```bash
python scripts/run_course_pipeline.py --data_root /path/to/image-eeg-data --clip_checkpoint /path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin --skip_features
```

训练时日志会明确显示 retrieval-way 设置，例如已经完成的本地运行会显示 `VAL (827-way)` 和 `TEST (200-way)`。

训练结束后汇总课程指标：

```bash
python scripts/evaluate_course_metrics.py
```

## 输出目录

所有运行产物都会集中放在 `output/`：

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

从目标 HPC 拷贝结果时，可以直接同步整个输出目录：

```bash
rsync -avP /path/to/version5_VED/output/ your_hpc:/path/to/save/version5_VED_output/
```

## 模型得分

本地运行设置：10 个 seed，范围为 `21` 到 `30`；validation checkpoint selection 使用 827-way retrieval，最终报告使用课程测试集 200-way retrieval。

| 选择规则 | Top-1 accuracy | Top-5 accuracy | 说明 |
|---|---:|---:|---|
| Validation-selected checkpoint | 82.40% ± 2.01% | 97.80% ± 0.54% | 正式课程结果 |
| Best test checkpoint | 86.85% ± 0.63% | 98.10% ± 0.52% | 仅作参考；偏乐观 |

指标汇总文件位于：

```text
output/course_metrics_summary.json
```

## 局限性

- 本分支只做图像检索，不做图像重建。
- 本地运行只使用课程 `sub-01` 数据，不是原论文中的 10-subject 完整复现 [1]。
- 结果依赖训练图像和测试图像完整存在。如果图片缺失，特征匹配阶段会丢弃样本，训练和评估结果会不可靠。
- `best_test` 指标不能作为主要成绩，因为它根据测试集表现选 checkpoint。
- OpenCLIP RN50 图像特征生成会占用较多存储和 GPU 时间，生成的 `.pt` 特征文件已被 `.gitignore` 忽略。

## 参考文献

[1] W. Liu, H. Li, Z. Xu, L. Ma, and H. Li, "Leveraging Visual Blur Perception Characteristics for EEG Decoding," *Proceedings of the AAAI Conference on Artificial Intelligence*, 40(21), 17580-17588, 2026. 本地论文副本：[`../references/paper/Liu 等 - 2026 - Leveraging Visual Blur Perception Characteristics for EEG Decoding.pdf`](../references/paper/Liu%20等%20-%202026%20-%20Leveraging%20Visual%20Blur%20Perception%20Characteristics%20for%20EEG%20Decoding.pdf)。

[2] A. T. Gifford, K. Dwivedi, G. Roig, and R. M. Cichy, "A large and rich EEG dataset for modeling human visual object recognition," *NeuroImage*, 2022. THINGS-EEG 项目页：<https://osf.io/b83fj/>。

[3] A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," *ICML*, 2021. <https://arxiv.org/abs/2103.00020>。

[4] G. Ilharco et al., "OpenCLIP," 2021. <https://github.com/mlfoundations/open_clip>。
