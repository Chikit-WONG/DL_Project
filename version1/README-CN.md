# 脑电信号图像检索与重建（THINGS-EEG）

DSAA2012 深度学习 — 项目 A

本项目基于 **THINGS-EEG** 数据集实现了一套完整的 EEG 到图像的处理流水线。
给定一段人类观看图像时产生的 63 通道 EEG 信号，模型需完成以下两个任务：

1. **图像检索**：从 200 个候选图像中找到与 EEG 对应的正确图像（零样本分类）
2. **图像重建**：通过 IP-Adapter + Stable Diffusion v1.5 生成逼真的图像

---

## 项目结构

```
DL_Project/
├── codes/
│   ├── config.py               # 所有超参数与路径配置
│   ├── data.py                 # 数据集、数据增强、DataLoader
│   ├── model.py                # EEG 编码器 + 统一模型
│   ├── utils.py                # 指标计算：检索、eval_images、工具函数
│   ├── cache_clip_features.py  # 预计算 CLIP 图像嵌入并缓存
│   ├── train.py                # 训练脚本（Phase 1 & 2，支持所有架构）
│   ├── reconstruct.py          # IP-Adapter 图像生成
│   ├── evaluate.py             # 端到端评估
│   └── run_all.ipynb           # Notebook：完整流水线演示
├── slurm_scripts/              # HPC 的 SLURM 作业脚本
├── checkpoints/                # 保存的模型权重（不纳入 git 追踪）
├── clip_cache/                 # 预计算的 CLIP 特征（不纳入 git 追踪）
├── outputs/                    # 重建图像与评估指标 JSON
├── image-eeg-data/             # THINGS-EEG 数据集
└── plan/                       # 实现方案（中文版）
```

---

## 模型架构

### EEG 编码器（约 570 万参数）

```
输入：[B, 63, T]
  ↓ 空间 Conv1d（63 → 128，k=1）× 2  +  BN + GELU
  ↓ 时序 Conv1d × 3（stride=2，k=15）→ [B, 320, T/8]
  ↓ 位置编码 + Transformer（3 层，d=320，heads=8，FFN=640）
  ↓ 全局平均池化 → [B, 320]
  ↓ MLP 头（320 → 640 → 1024）
输出：[B, 1024]（与 CLIP ViT-H-14 维度相同）
```

### 统一损失函数

```
L = α × L_InfoNCE  +  β × L_MSE

L_InfoNCE：对称 CLIP 风格对比损失（作用于 L2 归一化的嵌入）
L_MSE：    均方误差（作用于原始未归一化嵌入）
           → 强制 EEG 嵌入落入 CLIP 图像空间，兼容 IP-Adapter 重建
```

| 架构 | α | β | 说明 |
|---|---|---|---|
| **Arch A（联合训练）** | 1.0 | 1.0 | 同时优化检索与重建 |
| Arch B（仅检索）| 1.0 | 0.0 | 仅检索，无 MSE 监督 |
| Arch B（仅重建）| 0.0 | 1.0 | 仅重建，无 InfoNCE |

---

## 依赖环境

需要配置好 `test` conda 环境，并加载 CUDA 12.6：

```bash
# 核心依赖包
torch==2.10.0+cu126
torchvision==0.25.0+cu126
transformers==4.49.0
diffusers==0.37.1
accelerate==1.13.0
datasets==4.8.2
scikit-image==0.25.2
scipy==1.15.3
timm==1.0.26
clip==1.0
numpy==1.26.4
```

### 所需预训练模型权重

以下模型需存放在 `MODELS_ROOT` 目录下（在 `codes/config.py` 中设置路径）：

| 模型 | 用途 |
|---|---|
| `IP-Adapter/models/image_encoder/` | CLIP ViT-H-14 编码器（1024 维） |
| `CLIP-ViT-H-14-laion2B-s32B-b79K/` | 图像处理器配置文件 |
| `stable-diffusion-v1-5/` | SD 基础生成模型 |
| `IP-Adapter/models/ip-adapter_sd15.bin` | IP-Adapter 权重 |

评估阶段还需要以下权重（HPC 计算节点无法访问外网，需在登录节点提前下载）：

```bash
# 在登录节点（有代理网络）执行一次即可
python -c "
import torch, clip
clip.load('ViT-L/14')
torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
torch.hub.load('pytorch/vision', 'inception_v3', pretrained=True)
torch.hub.load('pytorch/vision', 'efficientnet_b1', pretrained=True)
torch.hub.load('facebookresearch/swav', 'resnet50', pretrained=True)
"
```

---

## 运行流程

所有脚本均以 `DL_Project/` 为工作目录，请在该目录下提交作业。

### 第 1 步 — 缓存 CLIP 图像特征

```bash
sbatch slurm_scripts/run_cache_clip.sh
# 输出：clip_cache/clip_train_features.pt（7968 张图 × 1024 维）
#       clip_cache/clip_test_features.pt （200  张图 × 1024 维）
```

### 第 2 步 — Phase 1 训练（检索预热，α=1，β=0）

```bash
sbatch slurm_scripts/run_train_phase1.sh
# 保存：checkpoints/phase1_main_best.pt
# 约 6 分钟（A100），50 轮，batch=128，lr=3e-4
```

### 第 3 步 — Phase 2 训练（联合微调，α=1，β=1）

```bash
sbatch slurm_scripts/run_train_phase2.sh
# 从 phase1_main_best.pt 继续训练
# 保存：checkpoints/phase2_main_best.pt
```

### 第 4 步 — 图像重建（10 个随机种子）

```bash
for i in {0..9}; do
  sbatch slurm_scripts/run_reconstruct_s${i}.sh
done
# 输出：outputs/recon_images_phase2_main_s0{0-9}.pt（每个约 157 MB）
# 注意：受 QOS 限制，最多同时提交 8 个作业
```

### 第 5 步 — 评估

```bash
sbatch slurm_scripts/run_evaluate.sh
# 输出：outputs/metrics_phase2_main_best.json
```

### Arch B 消融实验

```bash
# 训练
sbatch slurm_scripts/run_train_retrieval_only.sh   # α=1, β=0
sbatch slurm_scripts/run_train_recon_only.sh       # α=0, β=1

# 重建（同样 10 个种子，标签分别替换为 archB_retrieval / archB_reconstruction）
for i in {0..9}; do
  sbatch slurm_scripts/run_reconstruct_archB_ret_s${i}.sh
  sbatch slurm_scripts/run_reconstruct_archB_rec_s${i}.sh
done

# 评估
sbatch slurm_scripts/run_evaluate_archB_ret_full.sh
sbatch slurm_scripts/run_evaluate_archB_rec_full.sh
```

### 不使用 SLURM 手动运行

```bash
conda activate test
module load cuda/12.6

# 缓存特征
python codes/cache_clip_features.py

# 训练（所有参数说明）
python codes/train.py --phase 1 --tag my_run --epochs 50 --alpha 1.0 --beta 0.0
python codes/train.py --phase 2 --tag my_run --epochs 30 --alpha 1.0 --beta 1.0 \
                      --resume checkpoints/my_run_best.pt

# 重建
python codes/reconstruct.py --ckpt checkpoints/my_run_best.pt \
                             --seeds 0 1 2 --tag my_run --num_inference_steps 20

# 评估
python codes/evaluate.py --ckpt checkpoints/my_run_best.pt --recon_tag my_run
```

---

## 实验结果

### 任务一：图像检索（200-way 零样本分类）

| 模型 | α | β | Top-1 准确率 | Top-5 准确率 |
|---|---|---|---|---|
| Arch B（仅检索） | 1.0 | 0.0 | **14.5%** | 34.5% |
| **Arch A（联合训练）** | **1.0** | **1.0** | **13.5%** | **36.5%** |
| Arch B（仅重建） | 0.0 | 1.0 | 9.0% | 24.0% |
| 随机基准线 | — | — | 0.5% | 2.5% |

> Top-1 是随机基准的 27～29 倍；Top-5 是随机基准的 14～15 倍。

### 任务二：图像重建（10 个种子的均值 ± 标准差，200 张测试图像）

| 评估指标 | Arch B（仅检索） | **Arch A（联合训练）** | Arch B（仅重建） |
|---|---|---|---|
| PixCorr | 0.0328 ± 0.0063 | **0.0628 ± 0.0043** | 0.0709 ± 0.0067 |
| SSIM | 0.1981 ± 0.0042 | **0.2762 ± 0.0040** | 0.2749 ± 0.0030 |
| AlexNet（第 2 层） | 0.6482 ± 0.0178 | **0.7022 ± 0.0162** | 0.7124 ± 0.0124 |
| AlexNet（第 5 层） | 0.7416 ± 0.0138 | 0.7903 ± 0.0100 | **0.8223 ± 0.0076** |
| Inception | 0.6291 ± 0.0213 | 0.6732 ± 0.0122 | **0.7273 ± 0.0150** |
| CLIP | 0.6577 ± 0.0083 | 0.7081 ± 0.0088 | **0.7526 ± 0.0062** |
| EffNet | **0.9433 ± 0.0027** | 0.9267 ± 0.0025 | 0.8846 ± 0.0027 |
| SwAV | 0.6690 ± 0.0041 | 0.6282 ± 0.0026 | **0.5744 ± 0.0030** |

**加粗** = 三个模型中该指标最优。

---

## 最终建议

**推荐使用 Arch A（联合训练，α=1，β=1）作为最终方案。**

| 评估维度 | Arch B（仅检索） | **Arch A（联合训练）** | Arch B（仅重建） |
|---|---|---|---|
| 检索 Top-1 | 🥇 14.5% | 🥈 13.5% | 🥉 9.0% |
| 重建质量（8 指标中最优数量） | 🥉 0/8 | 🥈 4/8 | 🥇 4/8 |
| 是否同时支持两个任务 | 否 | **是** | 否 |

**推荐理由：**

- Arch A 的检索准确率（13.5%）仅比检索专项模型低 1%，代价极小。
- Arch A 的重建质量均衡——8 个指标中有 4 个达到最优，整体表现最稳定。
- Arch B（仅重建）虽有 4 个重建指标最优，但检索 Top-1 损失高达 5.5%，不适合作为通用模型。
- Arch B（仅检索）的重建质量在所有 8 个指标上均为三者中最差——单靠 InfoNCE 损失无法将嵌入向量充分对齐至 CLIP 图像空间，不足以驱动 IP-Adapter 重建。
- 联合训练验证了 **一个编码器同时服务两个任务** 的可行性，无需额外架构开销。
