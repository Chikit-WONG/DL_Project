# CognitionCapturerPro — 脑电图脑-图像检索与重建

[English README](./README.md)

本仓库为 DSAA2012 期末项目（陈亮教授）的 **CognitionCapturerPro** 分支。  
在课程 Things-EEG 数据集（单被试 sub-01）上复现并适配了 [CognitionCapturerPro 论文](https://arxiv.org/abs/2401.07935)。

整个流程将观看自然图片时的脑电信号（EEG）解码为：
1. **图像检索** — 给定脑电信号，从 200 张候选图像中排序，返回最可能匹配的图像。
2. **图像重建** — 以脑电信号为条件，使用 SDXL-Turbo + IP-Adapter 生成逼真图像。

---

## 流程概览

```
脑电信号（EEG）
    ↓
EEG 编码器（EEGProjectLayer，80 轮训练）
    ↓
CLIP 兼容嵌入（ViT-H-14 空间）
    ↓
┌──────────────────────────────────┐
│ 检索（余弦相似度）              │ → Top-1 / Top-5 准确率
└──────────────────────────────────┘
    ↓（可选对齐步骤）
SimpleAlignPipe MLP（100 轮训练）
    ↓
┌──────────────────────────────────┐
│ 图像生成（SDXL-Turbo           │ → 重建图像
│ + IP-Adapter ViT-H-14）        │
└──────────────────────────────────┘
```

---

## 环境配置

### 前置要求

- Python 3.10
- CUDA 12.6
- Conda（推荐）

### 安装依赖

```bash
conda create -n cogcap python=3.10 -y
conda activate cogcap
pip install -r requirements.txt
```

### 配置路径

复制示例配置并填入本地路径：

```bash
cp configs/local.example.yaml configs/local.yaml
```

编辑 `configs/local.yaml`：

```yaml
paths:
  data_root: /path/to/image-eeg-data/converted_for_cogcappro   # Things-EEG 课程数据集
  weights_root: /path/to/models                                  # CLIP、SDXL-Turbo、IP-Adapter
  runs_root: /path/to/version4_CCP/runs                         # 训练输出目录
  sdxl_root: /path/to/models/sdxl-turbo
  ip_adapter_root: /path/to/models/IP-Adapter
  clip_weights_rel:
    ViT-H-14: CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin
```

### 验证安装

```bash
python smoke_test.py
```

全部 13 项检查应通过（smoke test 无需 GPU）。

---

## 复现步骤

### 第 1 步 — 准备扩散嵌入

```bash
sbatch slurm_scripts/02b_reprepare_diffusion_embeddings.sh
```

### 第 2 步 — 训练 EEG 检索模型（80 轮）

```bash
sbatch slurm_scripts/07b_train_retrieval_full_v2.sh
```

### 第 3 步 — 训练对齐模型（SimpleAlignPipe，100 轮）

```bash
sbatch slurm_scripts/08d_simple_align.sh
```

### 第 4 步 — 生成图像

```bash
sbatch slurm_scripts/09e_generate_debug.sh   # 同时生成 all_before 和 all 两种模式
```

### 第 5 步 — 评估

```bash
sbatch slurm_scripts/10d_eval_debug.sh
sbatch slurm_scripts/11b_summary_v2.sh
```

---

## 模型得分（sub-01，seed 0）

### 检索（200-way，任意模态融合）

| 指标 | 本项目 | 论文（10 被试均值） |
|------|--------|---------------------|
| **Top-1** | **61.0%** | 61.2% |
| **Top-5** | **88.0%** | 90.8% |

### 重建

提供两种生成模式：

| 指标 | `all_before`（EEG → IP-Adapter，**最佳**） | `all`（SimpleAlignPipe） | 论文 |
|------|--------------------------------------------|--------------------------|------|
| CLIP (↑) | **0.707** | 0.659 | 0.830 |
| PixCorr (↑) | 0.130 | **0.133** | 0.163 |
| SSIM (↑) | **0.316** | 0.236 | 0.398 |
| AlexNet-2 (↑) | **0.663** | 0.618 | 0.831 |
| AlexNet-5 (↑) | **0.698** | 0.682 | 0.937 |
| Inception (↑) | 0.597 | **0.607** | 0.720 |

**推荐使用 `all_before` 模式**：将 EEG CLIP 嵌入直接送入 IP-Adapter，无需中间对齐步骤，整体质量更好。

---

## 输出位置

| 路径 | 内容 |
|------|------|
| `outputs/generated_all_before/` | 200 张生成图像（直接 EEG → IP-Adapter，最优质量） |
| `outputs/generated_all/` | 200 张生成图像（SimpleAlignPipe 对齐后） |
| `outputs/comparison/` | 三联对比网格（Ground Truth \| all_before \| all） |
| `outputs/comparison/grid_all200.png` | 全部 200 组三联图的总览网格 |
| `outputs/comparison/comparison_page01-10.png` | 分页对比图（每页 20 组） |
| `outputs/comparison/single/` | 200 张独立三联对比图 |
| `results/reconstruction_metrics_all_before.json` | `all_before` 重建指标 |
| `results/reconstruction_metrics_all.json` | `all` 重建指标 |
| `results/retrieval_test_results.json` | 检索准确率及各模态明细 |
| `results/summary_metrics.json` | 汇总指标 |

---

## 已修复的 Bug

相比原始仓库，共发现并修复了 6 个 Bug：

1. **嵌入键名碰撞**（`generator.py`）：不同类别目录下同名文件的扩散嵌入互相覆盖。改为以 `类别/文件名` 作为键名后修复。
2. **训练轮数严重不足**：检索模型实际只训练了 10 轮（配置要求 80 轮），对齐只训练了 1 轮。已在 `07b`/`08b` 脚本中修正。
3. **不确定性感知掩码被绕过**（`align/data.py`）：原代码硬编码 `DirectT` 覆盖，已移除，恢复使用 UM 模块。
4. **VAE float16 NaN**（`generator.py`）：`force_upcast=False` 导致 VAE 在 float16 下溢出，输出全黑图像。改为 `vae.config.force_upcast = True` 后修复。
5. **IP-Adapter 嵌入维度错误**（`generator.py`）：`guidance_scale=0.0` 时传入了 `[2,1,1024]` 而非 `[1,1,1024]`。加入 `embed.unsqueeze(0)` 后修复。
6. **PyTorch 2.6 `weights_only` 默认值变更**（`align/main.py`）：新版本默认 `True`，导致自定义数据集类无法反序列化。加入 `weights_only=False` 后修复。

---

## 局限性

- **单被试**：论文报告的是 10 个被试的平均值。仅用 1 个被试所得的脑电嵌入质量受限，存在不可消除的性能差距（约 10–20%）。
- **脑电噪声大**：原始脑电信号信噪比低，即使训练 80 轮，嵌入质量仍受信号质量制约。
- **语义鸿沟**：基于 CLIP 的检索/生成能正确识别高层类别（Top-1 达 61%），但仅凭脑电信号难以编码细粒度视觉细节（视角、纹理、光照等）。
- **对齐阶段引入失真**：EEG 检索模型输出的嵌入已在 IP-Adapter 期望的 CLIP 空间中；额外映射到"扩散嵌入空间"反而引入失真（CLIP 从 0.707 降至 0.659）。

---

## 关键文件

| 文件 | 作用 |
|------|------|
| `src/cogcappro/models/brain_backbone.py` | EEG 编码器（EEGProjectLayer） |
| `src/cogcappro/models/fusion_backbone.py` | 多模态融合骨干网络 |
| `src/cogcappro/training/module.py` | PyTorch Lightning 训练模块 |
| `src/cogcappro/align/diffusion_pipe.py` | SimpleAlignPipe + DiffusionPriorUNet |
| `src/cogcappro/generate_image/generator.py` | SDXL-Turbo + IP-Adapter 生成 |
| `configs/cogcappro.yaml` | 主要模型/训练配置 |
| `configs/local.yaml` | 本地路径配置（不提交至仓库） |
| `plan/Reproduce_CognitionCapturerPro_Fix_Plan_en.md` | 完整 Bug 修复与调优记录（英文） |
| `plan/Reproduce_CognitionCapturerPro_Fix_Plan_zh.md` | 完整 Bug 修复与调优记录（中文） |
