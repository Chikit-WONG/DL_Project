# THINGS-EEG 检索与重建 Version 2

> [English Documentation](README.md)

DSAA2012 深度学习课程项目

`version2` 是在 `version1` 基线之上独立实现的一套升级版 EEG 到图像流水线。但效果差，决定再换一个路线，开启新的version。 它继续使用同一份共享 THINGS-EEG 数据，但在方法上加入了：

- 更强的 EEG 编码器与双路径时序建模
- 电极位置编码与脑区感知通道 gating
- `CLIP ViT-H/14`、`ViT-B/32`、`RN50` 与 SD VAE latent 的多目标监督
- 语义先验网络 `Prior`
- 基于 `SDXL-Turbo + IP-Adapter + img2img` 的图像重建

项目目标是在不改动 `version1` 代码的前提下，同时提升检索和重建能力。

## 项目结构

```text
version2/
├── cache/                  # 图像 backbone 缓存特征
├── checkpoints/            # 编码器 / prior checkpoint
├── codes/                  # 主代码
├── logs/                   # SLURM 输出与训练日志
├── plan/                   # 规划文档与参考资料
├── results/                # 指标、重建结果、拼图、总结
├── slurm_scripts/          # HPC 作业脚本
├── README.md
└── README-CN.md
```

## 方法简介

### 1. 共享数据目录

`version2` 不会复制数据集，而是直接读取 [codes/config.py](/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/codes/config.py) 中定义的共享路径：

- EEG 与图像数据：
  `/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data`
- 训练图像：
  `training_images/`
- 测试图像：
  `test_images/`
- 电极坐标：
  `EEG_CHANNELS.jsonl`

数据读取逻辑默认执行 `80-trial averaging`，并保证 train/test 顺序与缓存好的视觉特征严格对齐。

### 2. EEG 编码器

[codes/model.py](/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/codes/model.py) 中的编码器包含：

- 基于电极空间坐标的位置编码
- 脑区感知 gating，后部通道初始权重更高
- 双路径时序卷积模块
- Transformer token mixing
- subject embedding adapter
- 两个输出头：
  - semantic head：预测图像语义嵌入
  - structural head：预测 SD VAE latent

### 3. 多目标监督

在训练编码器前，[codes/cache_backbone_features.py](/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/codes/cache_backbone_features.py) 会先缓存：

- `CLIP ViT-H/14` 图像嵌入
- `CLIP ViT-B/32` 图像嵌入
- `RN50` 图像嵌入
- `Stable Diffusion v1.5 VAE` latent

编码器训练分三阶段：

1. `warmup`
   `ViT-H/14 InfoNCE + 0.5 * MSE`
2. `multitarget`
   加入 `ViT-B/32`、`RN50`、`VAE latent` 监督
3. `finetune`
   再加入 hard-negative 与 supervised contrastive 项

### 4. Prior 与重建

[codes/train_prior.py](/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/codes/train_prior.py) 训练一个轻量语义先验网络。  
[codes/reconstruct.py](/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/codes/reconstruct.py) 的重建链路为：

- 编码器 semantic head
- 可选 prior 采样
- structural latent 解码成模糊底图
- `SDXL-Turbo` 的 img2img 精修
- `IP-Adapter SDXL` 用 EEG 预测到的语义嵌入做条件控制

## 环境配置

本实现是在 HPC 集群上的 `test` conda 环境中开发和运行的。

### 基本运行依赖

- Python 3.10
- 带 CUDA 的 PyTorch
- `transformers`
- `diffusers`
- `accelerate`
- `torchvision`
- `scikit-image`
- `numpy`
- `Pillow`
- 可选 `clip` 包，用于 `RN50` 缓存回退

作业脚本默认会执行：

```bash
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
module load cuda/12.6
```

### 所需模型目录

在 [codes/config.py](/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/codes/config.py) 中配置为：

- `/hpc2hdd/home/ckwong627/workdir/models/CLIP-ViT-H-14-laion2B-s32B-b79K`
- `/hpc2hdd/home/ckwong627/workdir/models/CLIP-ViT-B-32-laion2B-s34B-b79K`
- `/hpc2hdd/home/ckwong627/workdir/models/stable-diffusion-v1-5`
- `/hpc2hdd/home/ckwong627/workdir/models/IP-Adapter`
- `/hpc2hdd/home/ckwong627/workdir/models/sdxl-turbo`

`IP-Adapter` 相关关键文件：

- `models/ip-adapter_sd15.bin`
- `sdxl_models/ip-adapter_sdxl_vit-h.safetensors`
- `sdxl_models/image_encoder/`

## 运行方式

以下命令默认工作目录为：

```bash
cd /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2
```

### 1. Smoke test

建议先跑这个，用于检查环境、数据路径和最小训练闭环是否正常。

```bash
sbatch slurm_scripts/run_smoke_test.sh
```

### 2. 缓存视觉 backbone 特征

```bash
sbatch slurm_scripts/run_cache_backbone_features.sh
```

输出：

- `cache/backbone_cache_train.pt`
- `cache/backbone_cache_test.pt`

### 3. 训练编码器

Warmup：

```bash
sbatch slurm_scripts/run_train_encoder_warmup.sh
```

Multitarget：

```bash
sbatch slurm_scripts/run_train_encoder_multitarget.sh
```

Finetune：

```bash
sbatch slurm_scripts/run_train_encoder_finetune.sh
```

### 4. 训练 Prior

```bash
sbatch slurm_scripts/run_train_prior.sh
```

### 5. 生成重建图像

```bash
sbatch slurm_scripts/run_reconstruct_all.sh
```

### 6. 评估并生成总结

```bash
sbatch slurm_scripts/run_evaluate.sh
```

输出：

- `results/metrics_v2_final.json`
- `results/task2_montage_v2_final_s00.png`
- `results/results_summary_en.md`
- `results/results_summary_zh.md`

## 手动 CLI 入口

缓存：

```bash
python -u codes/cache_backbone_features.py --split all --batch_size 32
```

编码器三阶段：

```bash
python -u codes/train_encoder.py --stage warmup --tag v2_warmup
python -u codes/train_encoder.py --stage multitarget --tag v2_multitarget --resume checkpoints/v2_warmup_best.pt
python -u codes/train_encoder.py --stage finetune --tag v2_final --resume checkpoints/v2_multitarget_best.pt
```

Prior：

```bash
python -u codes/train_prior.py --encoder_ckpt checkpoints/v2_final_best.pt --tag v2_prior
```

重建：

```bash
python -u codes/reconstruct.py \
  --encoder_ckpt checkpoints/v2_final_best.pt \
  --prior_ckpt checkpoints/v2_prior_best.pt \
  --tag v2_final \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

评估：

```bash
python -u codes/evaluate.py \
  --tag v2_final \
  --encoder_ckpt checkpoints/v2_final_best.pt \
  --compare_v1
python -u codes/make_task2_montage.py --tag v2_final --seed_index 0 --num_samples 20
python -u codes/summarize_results.py --tag v2_final
```

## 当前结果

最近一次完整跑数结果保存在：

- [results/metrics_v2_final.json](/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/results/metrics_v2_final.json)
- [results/results_summary_zh.md](/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/results/results_summary_zh.md)
- [results/task2_montage_v2_final_s00.png](/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/results/task2_montage_v2_final_s00.png)

### 检索结果

| 模型 | Top-1 | Top-5 |
|---|---:|---:|
| Version2 `v2_final` | 15.00% | 35.00% |
| Version1 Joint | 13.50% | 36.50% |
| Version1 Retrieval-only | 14.50% | 34.50% |

### 重建结果

| 模型 | PixCorr | SSIM | 类 CLIP 分数 |
|---|---:|---:|---:|
| Version2 `v2_final` | 0.2754 | 0.3709 | 0.2779 |
| Version1 Joint | 0.0628 | 0.2762 | 0.7081 |

### 重要评估说明

`version2` 的重建 `CLIP` 分数目前不能和 `version1` 直接横向比较。

- `version1` 使用的是 TA 风格的 `two-way identification`，基于 `openai/CLIP ViT-L/14`
- `version2` 当前实现的是与缓存 `ViT-H/14` 图像嵌入的平均余弦相似度

因此，`0.2779` 不能直接解释成“明显差于 0.7081”，除非先统一评估协议。

## 当前已知问题

- `v2_final_best.pt` 目前是按 `Top-1` 选出来的，不是按 `Top-5` 或重建指标选出来的。
- finetune 中途出现过更高的 `Top-5`，但没有被当前的 checkpoint 选择规则保留下来。
- 当前 finetune 阶段可能会让 retrieval 相比 multitarget 最优点略有下降。
- 重建指标与 `version1` 并未完全同口径，比较时需要格外谨慎。

## 下一步建议

- 分别保存 `best_top1`、`best_top5` 与按重建指标选出的 checkpoint
- 重新单独评估 `v2_multitarget_best.pt` 作为 retrieval-oriented checkpoint
- 让 `version2` 的重建评估复用 `version1` 的官方 `eval_images` 指标口径
- 重新调整 finetune 阶段，避免 multitarget 之后 retrieval 退化
