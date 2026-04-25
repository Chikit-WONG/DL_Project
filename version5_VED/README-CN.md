# version5_VED：VisualEEGDecoding 课程适配版

[English README](README.md)

本目录是 DSAA2012 期末项目中的 `VisualEEGDecoding` 课程适配分支。它以 Liu 等人的模糊感知 EEG 检索路线为起点，并在此基础上扩展出一个可复现的 task 2 检索增强重建流程 [1]。

所有图像特征、日志、checkpoint、task 2 元数据、生成图像和评估结果都会统一写入 `output/`，方便在 HPC 上跑完后整体 `rsync` 回来。

## 项目介绍

`version5_VED` 现在覆盖课程要求的两个任务：

1. **Task 1：EEG-to-image retrieval**
2. **Task 2：EEG-to-image reconstruction**

其中 task 1 仍然是本仓库当前最强的检索分支。新的 task 2 分支**不会**直接从 EEG 解出一张模糊图，而是采用下面这条更务实的路线：

- 用 EEG 检索训练集中语义相近的图片
- 对 top-k 检索结果按类别聚合
- 用聚合后选出的训练类别填固定模板 prompt
- 用 top-1 检索图作为 IP-Adapter 的参考图
- 用 Stable Diffusion v1.5 + IP-Adapter 生成最终重建图像

这样做的核心假设是：虽然测试类别不在训练类别中，但在 CLIP 语义空间里，语义相近的训练类别仍然可以为生成模型提供有效提示。

## 方法介绍

### Task 1

task 1 保留了 VisualEEGDecoding 的核心思路 [1]：

1. `scripts/prepare_course_data.py` 把课程数据映射到 `data/things-eeg/`
2. `preprocess/process_image_course.py` 用 OpenCLIP RN50 在 12 个 Gaussian blur level 上提取训练图像和测试图像特征
3. `main_eeg_course.py` 在 `sub-01` 上训练 EEG encoder
4. 图像分支学习融合 12 个 blur-level RN50 特征，EEG 分支输出匹配的 1024 维 embedding
5. 训练目标是 CLIP 风格的双向对比学习 [3, 4]

### Task 2

task 2 新增了一个检索增强重建分支：

1. `scripts/train_task2_semantic.py` 在 task 1 checkpoint 基础上做语义微调
2. 训练类别 prototype 直接用同一个 OpenCLIP RN50 text encoder 编码，所以不需要额外加输出头
3. 测试时，每个 EEG 会先在训练图像库中做 top-k 检索
4. 对 top-k 检索图像按类别聚合，得分最高的训练类别作为 prompt 类别
5. 固定模板 prompt 为：
   - `a realistic photo of a {class_name}`
6. 得分最高的检索图像作为 IP-Adapter 参考图
7. `scripts/generate_task2_reconstructions.py` 用 Stable Diffusion v1.5 + IP-Adapter 生成重建图像
8. `scripts/evaluate_task2_reconstruction.py` 按课程口径评估 `SSIM` 和 `CLIP`

第一版实现明确优先使用 **IP-Adapter**，不先做 `T2I-Adapter`，因为我们当前拥有的是“检索到的相似参考图”，而不是可靠的 EEG 推断 edge/depth/control map。

## 环境配置

建议 Python 版本：**Python 3.10**。

创建并激活环境：

```bash
conda create -n ved python=3.10 -y
conda activate ved
pip install -r requirements.txt
```

如果 HPC 上已有依赖比较全的 `test` 环境，也可以优先尝试复用，只要其中已经有兼容版本的 `torch`、`open-clip-torch`、`diffusers`、`transformers` 和 `scikit-image`。

## 数据和模型准备

课程数据目录需要包含：

```text
image-eeg-data/
  training_images/
  test_images/
  converted_for_cogcappro/ThingsEEG/Preprocessed_data_250Hz_whiten/sub-01/train.pt
  converted_for_cogcappro/ThingsEEG/Preprocessed_data_250Hz_whiten/sub-01/test.pt
```

当前实现依赖的本地模型：

- OpenCLIP RN50 checkpoint
- Stable Diffusion v1.5
- IP-Adapter SD1.5 权重

当前默认路径：

```text
/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/open_clip_pytorch_model.bin
/hpc2hdd/home/ckwong627/workdir/models/stable-diffusion-v1-5
/hpc2hdd/home/ckwong627/workdir/models/IP-Adapter
```

如果模型不存在，请先在 `/hpc2hdd/home/ckwong627/workdir/models/` 下创建对应文件夹，再下载：

```bash
mkdir -p /hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai
hf download timm/resnet50_clip.openai \
  --include open_clip_pytorch_model.bin \
  --local-dir /hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai

mkdir -p /hpc2hdd/home/ckwong627/workdir/models/stable-diffusion-v1-5
hf download runwayml/stable-diffusion-v1-5 \
  --local-dir /hpc2hdd/home/ckwong627/workdir/models/stable-diffusion-v1-5

mkdir -p /hpc2hdd/home/ckwong627/workdir/models/IP-Adapter
hf download h94/IP-Adapter \
  --local-dir /hpc2hdd/home/ckwong627/workdir/models/IP-Adapter
```

大致资源说明：

- OpenCLIP RN50：`open_clip_pytorch_model.bin` 约 0.4 GB，主要用于图像特征和文本 prototype 提取
- Stable Diffusion v1.5：完整 diffusers 目录通常约 4 到 7 GB
- IP-Adapter 权重和 image encoder：完整目录通常约 3 到 5 GB
- task 2 生成阶段需要 1 张 GPU；`debug` 分区适合 smoke test，不适合完整长时间重建

## 命令

### Task 1：一行命令跑检索流程

```bash
python scripts/run_course_pipeline.py \
  --data_root /path/to/image-eeg-data \
  --clip_checkpoint /path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin
```

如果是在你同学那台可以直接 `python` 用 GPU 的 A800 机器上，优先直接运行：

```bash
bash run_task1_direct.sh
```

### Task 2：一行命令跑重建流程

```bash
python scripts/run_task2_pipeline.py \
  --data_root /path/to/image-eeg-data \
  --clip_checkpoint /path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin \
  --task1_ckpt /path/to/task1_select_checkpoint.pth
```

如果是在你同学那台可以直接 `python` 用 GPU 的 A800 机器上，优先直接运行：

```bash
bash run_task2_direct.sh
```

这条 task 2 命令会完成：

1. 更新课程数据符号链接
2. 在 task 1 checkpoint 上做 class-text prototype 语义微调
3. 为每个 seed 生成重建图像
4. 评估 `SSIM` 和 `CLIP`
5. 在 `output/task2/` 下保存逐 seed 和汇总指标

### Task 2 smoke test

```bash
python scripts/run_task2_pipeline.py \
  --data_root /path/to/image-eeg-data \
  --clip_checkpoint /path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin \
  --task1_ckpt /path/to/task1_select_checkpoint.pth \
  --epoch 1 \
  --n_seeds 1 \
  --first_seed 999
```

### 生成定性结果拼图

```bash
python scripts/make_task2_qualitative_grid.py \
  --real-root output/task2/pipeline_runs/<run>/reconstructions/seed21/ground_truth \
  --fake-root output/task2/pipeline_runs/<run>/reconstructions/seed21/generated \
  --output output/task2/pipeline_runs/<run>/qualitative_seed21.png
```

## SLURM 使用方式

SLURM 提交脚本放在：

```text
version5_VED/slurm_scripts/
```

当前已有：

- `02_gen_blur_features.sh`
- `03_train_eeg.sh`
- `04_run_task2_smoke.sh`
- `05_run_task2_full.sh`

另外也补了两份**非 SLURM** 直接运行脚本，方便在不用 `sbatch` 的 GPU 机器上直接跑：

- `run_task1_direct.sh`
- `run_task2_direct.sh`

这些脚本开头都会自动执行两件事：

- `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY`
- 如果 shell 里有 `unclash` 函数，就额外调用一次 `unclash`

这样可以避免代理环境被继承到作业里，导致 `sbatch` 后的 Python/Hugging Face/学校调度客户端异常。

当前默认策略：

- smoke test 先用 `debug`
- 完整 task 2 运行如果 `debug` 时间不够，则切到 `long_gpu`

如果排队太久，再考虑改到 `emergency_gpua40` 或 `emergency_gpu` 等更快分区。

## 输出目录

```text
output/
  Image_feature/
  logs/main_eeg_course/
  task2/
    semantic_finetune/
    reconstructions/
    pipeline_runs/
```

task 2 重点产物包括：

- 语义微调 checkpoint
- 类别文本 prototype cache
- 训练图像 bank cache
- 生成图像
- ground-truth 拷贝
- 检索元数据 JSON
- 重建评估 JSON/CSV
- 定性拼图

## 模型得分

### Task 1

已完成的本地 task 1 运行：

| 选择规则 | Top-1 accuracy | Top-5 accuracy | 说明 |
|---|---:|---:|---|
| Validation-selected checkpoint | 82.40% ± 2.01% | 97.80% ± 0.54% | 更保守的选点 |
| Best test checkpoint | 86.85% ± 0.63% | 98.10% ± 0.52% | 当前选定的提交结果 |

该次运行的 validation 是 **827-way**，test 是 **200-way**。

### Task 2

task 2 的代码链路已经实现，但在完整多 seed GPU 运行完成前，这里不预填新的最终重建分数。真正的结果以 `output/task2/` 下生成的评估 JSON 为准，重点看：

- `eval_ssim`
- `eval_clip`

## 局限性

- task 2 的 prompt 类别来自**训练集**检索结果，而不是不可见测试类的真实标签
- 这条路线更可能先提升语义一致性，不一定同等提升空间结构一致性
- IP-Adapter 的参考图质量直接受检索质量影响
- 当前第一版只使用固定模板 prompt 和单张 top-1 参考图
- `T2I-Adapter`、自由 prompt 生成、多参考图融合都还没有放进第一版实现

## 参考文献

[1] W. Liu, H. Li, Z. Xu, L. Ma, and H. Li, "Leveraging Visual Blur Perception Characteristics for EEG Decoding," *Proceedings of the AAAI Conference on Artificial Intelligence*, 40(21), 17580-17588, 2026. 本地论文副本：[`../references/paper/Liu 等 - 2026 - Leveraging Visual Blur Perception Characteristics for EEG Decoding.pdf`](../references/paper/Liu%20等%20-%202026%20-%20Leveraging%20Visual%20Blur%20Perception%20Characteristics%20for%20EEG%20Decoding.pdf)。

[2] A. T. Gifford, K. Dwivedi, G. Roig, and R. M. Cichy, "A large and rich EEG dataset for modeling human visual object recognition," *NeuroImage*, 2022. THINGS-EEG 项目页：<https://osf.io/b83fj/>。

[3] A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," *ICML*, 2021. <https://arxiv.org/abs/2103.00020>。

[4] G. Ilharco et al., "OpenCLIP," 2021. <https://github.com/mlfoundations/open_clip>。
