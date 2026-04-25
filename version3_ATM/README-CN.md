# EEG 脑电图到图像检索与重建（ATMS）

[English](README.md) | **[中文说明]**

---

DSAA2012 深度学习期末项目 — 陈亮教授  
数据集：THINGS-EEG（sub-01，63 个 EEG 通道 × 250 个时间点，1654 个训练类别，200 个测试类别）

## 项目简介

本项目使用 **ATMS**（Attention-based Time-series to Multi-modal Space，基于注意力机制的时序到多模态空间编码器）实现 EEG 脑电信号到图像的解码。

整体流程分为两个分支：

| 分支 | 说明 |
|------|------|
| **检索（Retrieval）** | 训练 ATMS 将 EEG 嵌入与 CLIP ViT-H-14 图像特征对齐；评估 200-way Top-1/Top-5 准确率 |
| **重建（Reconstruction）** | 对 ATMS 进行回归微调；将 EEG 嵌入送入 SD v1.5 + IP-Adapter 生成图像 |

### 架构图

```
EEG [B, 63, 250]
  → iTransformer 编码器（主体嵌入 + 自注意力）
  → PatchEmbedding（时域 CNN）→ 展平 [B, 1440]
  → 线性投影 → 1024 维 CLIP ViT-H-14 特征空间
  → [检索] 与 200 个 CLIP 图像特征计算余弦相似度
  → [生成] IP-Adapter → Stable Diffusion v1.5 → 512×512 图像
```

---

## 实验结果（sub-01，训练 40 个 epoch）

### 检索（200-way）

| 指标 | 分数 |
|------|------|
| Top-1 准确率 | **33.50%** |
| Top-5 准确率 | **63.50%** |

### 重建

| 指标 | 分数 |
|------|------|
| CLIP Score | **0.6089 ± 0.0123** |
| SSIM | **0.2709 ± 0.0052** |
| PixCorr | **0.0500 ± 0.0093** |
| AlexNet-2 | 0.6994 ± 0.0149 |
| AlexNet-5 | 0.7047 ± 0.0175 |
| Inception | 0.5765 ± 0.0242 |
| EffNet（↓ 越低越好） | 0.9581 ± 0.0041 |
| SwAV（↓ 越低越好） | 0.6493 ± 0.0032 |

详细结果 CSV：[`outputs/retrieval_eval_run01.csv`](outputs/retrieval_eval_run01.csv) · [`outputs/reconstruction_eval_run02_multiseed.csv`](outputs/reconstruction_eval_run02_multiseed.csv)

其中 retrieval 的 CSV 虽然按标准 10 个随机 200-way seed 输出，但因为候选集本来就是全部 200 个测试类别，所以每一行结果都相同；reconstruction 的 CSV 现在已经是真正的 10-seed 生成与评估结果。

---

## 环境配置

### 前提条件

- Python 3.10
- CUDA 12.6
- Conda（推荐）

### 安装步骤

```bash
# 创建并激活 conda 环境
conda create -n eeg_atm python=3.10 -y
conda activate eeg_atm

# 安装 PyTorch（根据实际 CUDA 版本调整）
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 安装其余依赖
pip install -r requirements.txt
```

### 所需预训练模型

| 模型 | 集群路径 |
|------|----------|
| CLIP ViT-H-14（OpenCLIP） | `/hpc2hdd/home/ckwong627/workdir/models/CLIP-ViT-H-14-laion2B-s32B-b79K/` |
| Stable Diffusion v1.5 | `/hpc2hdd/home/ckwong627/workdir/models/stable-diffusion-v1-5/` |
| IP-Adapter SD1.5 ViT-H | `/hpc2hdd/home/ckwong627/workdir/models/IP-Adapter/models/ip-adapter_sd15.bin` |

### 数据集

将 THINGS-EEG 预处理后的数据放置于：

```
/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/
  ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data/
    train.pt
    test.pt
    test_images/
      <类别名称>/
        *.jpg
```

### Smoke Test（仅 CPU，无需 GPU）

```bash
cd version3_ATM
python smoke_test.py
```

所有 12 项检查均应输出 `[OK]`，最后一行显示 `SMOKE TEST PASSED`。

---

## 完整流程

### 第一步 — 训练检索模型

```bash
sbatch slurm_scripts/run_train_retrieval.sh
# 检查点保存到：models/contrast/ATMS/sub-01/<时间戳>/{5,10,...,40}.pth
```

### 第二步 — 训练重建模型

```bash
sbatch slurm_scripts/run_train_reconstruction.sh
# 检查点保存到：models/contrast/ATMS/sub-01/<时间戳>/{5,10,...,40}.pth
```

### 第三步 — 评估检索性能

```bash
sbatch slurm_scripts/run_eval_retrieval.sh \
  ./models/contrast/ATMS/sub-01/<时间戳>/40.pth run01
# 输出：outputs/retrieval_eval_run01.csv
```

### 第四步 — 生成重建图像

```bash
sbatch slurm_scripts/run_generate_recon.sh \
  ./models/contrast/ATMS/sub-01/<时间戳>/40.pth run01
# 输出：outputs/reconstructions/run01/{ground_truth/, generated/, recon_tensors.pt}
```

### 第五步 — 评估重建指标

```bash
sbatch slurm_scripts/run_eval_reconstruction.sh \
  ./outputs/reconstructions/run01/recon_tensors.pt run01
# 输出：outputs/reconstruction_eval_run01.csv
```

---

## 输出文件位置

| 输出内容 | 路径 |
|----------|------|
| 训练检查点 | `models/contrast/ATMS/sub-01/<时间戳>/<epoch>.pth` |
| 训练损失曲线 | `outputs/contrast/ATMS/sub-01/<时间戳>/ATMS_sub-01.csv` |
| 检索评估 CSV | `outputs/retrieval_eval_<run>.csv` |
| 真实图像（256×256） | `outputs/reconstructions/<run>/ground_truth/<idx>.png` |
| 生成图像（256×256） | `outputs/reconstructions/<run>/generated/<idx>.png` |
| 图像张量（用于评估） | `outputs/reconstructions/<run>/recon_tensors.pt` |
| 重建指标 CSV | `outputs/reconstruction_eval_<run>.csv` |
| SLURM 作业日志 | `logs/<任务类型>_<jobid>.{out,err}` |

---

## 修复的关键 Bug

本项目在开发过程中发现并修复了三个非显而易见的 Bug：

1. **`num_subjects` 不匹配** — 训练脚本以 `num_subjects=10` 初始化 iTransformer，生成 [10, 250] 的主体嵌入矩阵。评估和生成脚本误用默认值 `num_subjects=2`，导致加载检查点时出现 `size mismatch` 错误。修复方式：显式传入 `num_subjects=10`。

2. **导入 diffusers 时发生段错误** — 在函数体内部（CUDA 上下文已由 EEG 编码器建立之后）延迟导入 `diffusers` 会引发 C 扩展冲突，导致段错误。修复方式：将 `diffusers` 的导入移至模块顶层，并在加载 SD 之前释放 EEG 模型（`del model; torch.cuda.empty_cache()`）。

3. **IP-Adapter 双重投影** — 原始脚本通过 `ImageProjection` 预先将 EEG 嵌入从 1024 维投影为 4×768，再传给 `pipe(ip_adapter_image_embeds=...)`。但 diffusers 的 UNet 在内部同样会对 `ip_adapter_image_embeds` 执行 `encoder_hid_proj`（即同一个 `ImageProjection`），导致二次投影和矩阵维度不匹配（`8×768` × `1024×3072`）。修复方式：直接传入原始 [N, 1024] 嵌入，由 UNet 内部完成投影。

---

## 局限性

- **单一被试**：所有结果仅针对 `sub-01`，未评估跨被试泛化能力。
- **训练轮数较少**：在单张 A40 GPU 上训练 40 个 epoch；原始 ATMS 论文使用更长的训练周期。
- **重建质量有限**：即使完成了多 seed 重跑，CLIP Score 也只有 0.61，仍明显低于本仓库最强分支。ATMS 重建分支使用回归损失而非对比对齐，限制了嵌入质量。
- **EffNet / SwAV 距离较高**：说明生成图像的纹理和底层特征与真实图像差异较大。
- **无数据增强或多试验集成**：课程协议要求对多次试验取均值，这会损失时间维度的变异信息。

---

## 项目结构

```
version3_ATM/
├── EEG-preprocessing/              # 原始 EEG 预处理工具
├── eval/
│   ├── eval_retrieval_200way.py    # 200-way 检索评估
│   └── eval_reconstruction_metrics.py  # SSIM / CLIP / AlexNet 等指标
├── Generation/
│   ├── ATMS_reconstruction.py      # 重建模型训练脚本
│   └── generate_reconstructions.py # SD v1.5 + IP-Adapter 图像生成
├── Retrieval/
│   └── ATMS_retrieval.py           # 检索模型训练脚本
├── models/
│   ├── data_bridge.py              # 统一数据加载器（train/test .pt）
│   ├── clip_bridge.py              # OpenCLIP ViT-H-14 辅助工具
│   ├── loss.py                     # CLIP 对比损失
│   └── subject_layers/             # iTransformer、注意力机制、嵌入层
├── slurm_scripts/                  # SBATCH 提交脚本
├── outputs/                        # 评估 CSV 和生成图像
├── smoke_test.py                   # 仅 CPU 的快速验证脚本
├── requirements.txt
└── README.md / README-CN.md
```

---

## 许可证

原始代码库：[EEG_Image_decode](https://github.com/eegatlas/EEG_Image_decode) — 详见 [LICENSE](LICENSE)。  
修改与新增内容：Chi Kit Wong，2026 年 4 月。
