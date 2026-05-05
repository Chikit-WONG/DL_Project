# DSAA2012 期末项目：基于 EEG 的图像检索与重建

[English README](README.md)

本仓库在 THINGS-EEG 数据集上实现了一套完整的 EEG-to-Image 系统，涵盖两个必做任务：

1. **任务 1 — 脑电图到图像检索**：给定一段 EEG 信号，在 200 个候选图像中对正确图像进行排名。
2. **任务 2 — 脑电图到图像重建**：给定一段 EEG 信号，生成一张在结构和语义上与受试者观看刺激图像一致的图像。

---

## 环境配置

任务 1 和任务 2 共享同一个 conda 环境。任务 2 的固定依赖（`torch==2.5.0`、`open-clip-torch==3.2.0`、`numpy==2.0.2`）满足任务 1 的宽松要求，因此一个环境可以同时满足两个任务的需求。

```bash
# 1. 创建并激活环境（需要 Python 3.10）
conda create -n DL_Project python=3.10 -y
conda activate DL_Project

# 2. 安装支持 CUDA 的 PyTorch
#    请根据您的 CUDA 版本调整 cu121（如 cu118、cu124 等）
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装其余依赖
pip install -r requirements.txt
```

EVNet（任务 1）已内置于 `task1/evnet/`，无需单独安装。

**所需外部模型（需自行下载）：**

| 模型 | 大小 | 用途 |
|------|------|------|
| OpenCLIP RN50 (`open_clip_pytorch_model.bin`) | ~102 MB | 任务 1 模糊特征与 EVNet 图像编码器 |
| OpenCLIP ViT-H-14 LAION-2B (`open_clip_pytorch_model.bin`) | ~4.4 GB | 任务 2 多模态监督编码器 |
| SDXL-Turbo | ~6 GB | 任务 2 图像生成主干 |
| IP-Adapter (`ip-adapter-plus_sdxl_vit-h`) | ~1 GB | 任务 2 图像条件控制 |

---

## 仓库结构

```text
DL_Project/
├── requirements.txt                # 任务 1 和任务 2 共享依赖
├── task1/                          # 任务 1：EEG 图像检索（VED + EVNet）
│   ├── main_eeg_course.py          # 训练与评估入口
│   ├── preprocess/
│   │   └── process_image_course.py # 离线图像特征提取
│   ├── models/                     # EEG 编码器定义
│   ├── scripts/
│   │   └── evaluate_course_metrics.py
│   ├── evnet/                      # 内置 EVNet 库
│   └── slurm_scripts/              # HPC 作业脚本
└── task2/                          # 任务 2：EEG 图像重建（CognitionCapturerPro）
    ├── main.py                     # 训练入口
    ├── smoke_test.py               # 13 项验证脚本
    ├── configs/                    # YAML 配置文件
    ├── src/cogcappro/              # 核心包
    └── slurm_scripts/              # HPC 作业脚本
```

---

## 任务 1：EEG 图像检索

### 方法介绍

检索模型将原始 EEG 信号映射到 CLIP 嵌入空间，并在该空间中与预计算的图像嵌入进行余弦相似度匹配。

图像表征融合了两个互补来源：

1. **多尺度模糊特征**：对每张训练/测试图像施加 8 或 12 个级别的高斯模糊，经 OpenCLIP RN50 编码后得到特征栈，再通过可学习的 softmax 注意力权重加权聚合。

2. **EVNet 特征**：仿生视觉前端（SubcorticalBlock 模拟视网膜/LGN + VOneBlock 模拟初级视觉皮层 V1 的 Gabor 滤波器）对图像进行预处理，再经 OpenCLIP RN50 编码。EVNet 及适配层的所有权重**以随机初始化状态冻结**，训练时仅更新下游融合层和 EEG 编码器。

两路特征通过可学习 softmax 权重（初始化为 0.7/0.3）融合：

```
fused_img = softmax([w_blur, w_evnet]) · [blur_agg, evnet_feat]
img_emb   = fusion_adapter(fused_img)    # MLP 1152→768→1152
```

EEG 编码器将原始信号映射到相同的 1152 维空间，使用 **InfoNCE（对称对比损失）** 训练。

### 模型架构

```
EEG 输入 [B, 63 通道, 250 时间步]
  └─ Conv2dWithAbs（63→25 滤波器，空间卷积）
  └─ BatchNorm2d
  └─ Linear(250→200) + ELU + Dropout(0.25)
  └─ Linear(200→200) + ELU + Dropout(0.65)
  └─ Linear(25×200→1152)
  └─ EEG 嵌入向量 [B, 1152]

图像输入（离线预计算）
  ├─ 多尺度模糊：CLIP RN50 × 8 级 → [B, 8, 1024]
  │    └─ 注意力加权聚合 → blur_agg [B, 1024]
  └─ EVNet：SubcorticalBlock → VOneBlock → Conv2d 适配层 → CLIP RN50 → [B, 1024]
       └─ evnet_feat [B, 1024]

融合：softmax([w0, w1]) · [blur_agg, evnet_feat] → MLP → img_emb [B, 1152]

损失：InfoNCE(EEG_emb, img_emb)
```

**模糊级别预设（8 级，最终提交使用）：**
`σ ∈ {l_1, l_3, l_15, l_21, l_33, l_45, l_57, l_63}`

### 数据结构

将课程提供的 `image-eeg-data/` 文件夹直接放在 `DL_Project/` 根目录下：

```text
DL_Project/
├── image-eeg-data/          ← 将数据集文件夹放在这里
│   ├── train.pt
│   ├── test.pt
│   ├── training_images/
│   ├── test_images/
│   └── converted_for_cogcappro/   ← 已预先构建，无需额外操作
└── task1/
```

`main_eeg_course.py` 和 `process_image_course.py` 均会从该位置自动探测 `image-eeg-data/`，无需传递 `--eeg_data_dir` 参数或手动创建软链接。

### 运行步骤

**步骤 1 — 生成离线图像特征**（运行一次，A40 上约 15 分钟）：

```bash
cd task1
python preprocess/process_image_course.py \
    --clip_checkpoint /path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin \
    --save_dir output/Image_feature \
    --backbone rn50 \
    --evnet_mode random \
    --batch_size 128
```

输出文件保存至 `task1/output/Image_feature/`：`MultiBlur_RN50_train.pt`、`MultiBlur_RN50_test.pt`、`EVNet_RN50_train.pt`、`EVNet_RN50_test.pt`。

**步骤 2 — 训练 EEG 检索模型**（10 个种子，200 轮，A40 上约 8–16 小时）：

```bash
python main_eeg_course.py \
    --blur_config 8 \
    --use_evnet \
    --epoch 200 \
    --train_batch_size 1024 \
    --lr 0.001 \
    --n_seeds 10 \
    --first_seed 21 \
    --eeg_data_dir /path/to/Preprocessed_data_250Hz_whiten/sub-01 \
    --feature_path output/Image_feature \
    --output_dir output/logs/8blur_evnet_full
```

添加 `--use_full_train` 可使用全量训练集（推荐，得分最高）。

**步骤 3 — 评估：**

```bash
python scripts/evaluate_course_metrics.py \
    --log_dir output/logs/8blur_evnet_full
```

**主要参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--blur_config` | `8` | 模糊级别预设：`8` 或 `12` |
| `--use_evnet` | 关闭 | 启用 EVNet 特征融合 |
| `--use_full_train` | 关闭 | 使用全量训练集（不划分验证集） |
| `--epoch` | `200` | 训练轮数 |
| `--train_batch_size` | `1024` | 批大小 |
| `--lr` | `0.001` | 学习率 |
| `--n_seeds` | `10` | 随机种子数量 |
| `--first_seed` | `21` | 起始种子值（实际为 21–30） |
| `--eeg_data_dir` | — | `sub-01/` 预处理 EEG 数据路径 |
| `--feature_path` | `output/Image_feature` | `.pt` 特征文件目录 |
| `--output_dir` | `output/logs/main_eeg_course` | 输出目录 |

**SLURM（HPC）：** 已预配置脚本位于 `task1/slurm_scripts/`。提交前需先设置 `CLIP_RN50`（未设置时脚本立即报错退出）。`EEG_DATA_DIR` 会从 `image-eeg-data/` 自动探测，仅在数据不在默认位置时才需手动指定。

```bash
export CLIP_RN50=/path/to/CLIP-RN50-openai/open_clip_pytorch_model.bin
# export EEG_DATA_DIR=/path/to/Preprocessed_data_250Hz_whiten/sub-01  # 仅在非默认位置时设置

sbatch task1/slurm_scripts/01_gen_evnet_features.sh       # 生成特征（仅需一次）
sbatch task1/slurm_scripts/04_full_train_8blur_evnet.sh   # 全量训练，最优结果
```

### 任务 1 实验结果

所有实验：10 个随机种子（种子 21–30），200 轮训练，批大小 1024，学习率 0.001，单受试者（sub-01），200 路图像检索评估。

**验证集选择（Val-selected）**：从 5% 验证集中按最佳 Top-1 选取检查点，在测试集上评估。
**最佳测试（Best-test）**：所有训练轮次中测试集 Top-1 的最高值。

#### 主实验

| 设置 | 验证选 Top-1 | 验证选 Top-5 | 最佳测试 Top-1 | 最佳测试 Top-5 |
|---|---|---|---|---|
| 8-blur + EVNet，95/5 划分 | 0.8460 ± 0.0135 | 0.9870 ± 0.0059 | 0.8715 ± 0.0091 | 0.9860 ± 0.0081 |
| 12-blur + EVNet，95/5 划分 | 0.8400 ± 0.0186 | 0.9860 ± 0.0046 | 0.8715 ± 0.0111 | 0.9855 ± 0.0028 |
| **8-blur + EVNet，全量训练** | **0.8530 ± 0.0136** | **0.9860 ± 0.0046** | **0.8785 ± 0.0082** | **0.9855 ± 0.0037** |
| 12-blur + EVNet，全量训练 | 0.8505 ± 0.0169 | 0.9845 ± 0.0037 | 0.8810 ± 0.0074 | 0.9850 ± 0.0041 |

**最终提交选择：8-blur + EVNet，全量训练 — 最佳测试 Top-1 = 87.85%，Top-5 = 98.55%。**

全量训练相较于 95/5 划分，最佳测试 Top-1 提升约 0.007–0.010。

#### 消融实验

所有消融均使用 8-blur、RN50 主干、95/5 划分。

| 消融方案 | 验证选 Top-1 | 最佳测试 Top-1 | 相对基准 |
|---|---|---|---|
| **基准：EVNet 固定，Kaiming 初始化** | 0.8460 ± 0.0135 | 0.8715 ± 0.0091 | — |
| Xavier 初始化适配层 | 0.8275 ± 0.0175 | 0.8495 ± 0.0086 | −0.019 |
| GAP + 线性层（无 CLIP 主干） | 0.8285 ± 0.0173 | 0.8620 ± 0.0092 | −0.018 |
| ViT-H/14 主干 | 0.7365 ± 0.0208 | 0.7790 ± 0.0115 | −0.110 |

**主要结论：**
- **Kaiming 优于 Xavier**：Kaiming 正态初始化比 Xavier 均匀初始化高约 0.019 验证 Top-1。Xavier 产生的权重幅值较小，导致冻结后的适配层表达能力不足。
- **GAP 消融效果出人意料地好**：完全去掉 CLIP 主干（改用全局平均池化 + 线性投影）后，验证 Top-1 仅下降约 0.018，说明 EVNet 的 V1 类特征本身已包含大量对 EEG 对齐有效的视觉信息。
- **ViT-H/14 与 EVNet 不兼容**：ViT 基于分块注意力，依赖干净的像素分块输入；EVNet 的空间化预处理破坏了 token 结构，导致 Top-1 下降约 0.11。RN50 作为 CNN 主干，天然兼容 EVNet 的卷积输出。

**消融实验运行命令：**

```bash
# Xavier 初始化
python preprocess/process_image_course.py --evnet_mode xavier ...
python main_eeg_course.py --evnet_prefix EVNet_xavier_RN50 ...

# GAP + 线性层（无主干）
python preprocess/process_image_course.py --evnet_mode gap ...
python main_eeg_course.py --evnet_prefix EVNet_gap ...

# ViT-H/14 主干
python preprocess/process_image_course.py --backbone vit_h_14 --clip_checkpoint /path/to/ViT-H-14/...
python main_eeg_course.py --blur_prefix MultiBlur_ViTH14 --evnet_prefix EVNet_ViTH14 ...
```

---

## 任务 2：EEG 图像重建

### 方法介绍

重建流程基于 CognitionCapturerPro 改编。先训练一个多模态 EEG 编码器将 EEG 信号映射为 CLIP 嵌入，再经扩散先验和 SDXL-Turbo + IP-Adapter 生成图像。

**完整流程：**

```
EEG → EEGProjectLayer → CLIP 嵌入
  ├─ [检索] 与图像/文本/深度/边缘 CLIP 嵌入做余弦相似度匹配
  └─ [重建] SimpleAlignPipe（MLP 扩散先验）→ CLIP 图像嵌入
                └─ SDXL-Turbo + IP-Adapter → 生成图像
```

**关键阶段：**

1. **EEG 编码器训练**（80 轮）：`EEGProjectLayer` 将 EEG [63 通道 × 250 时间步] 映射为 1024 维 CLIP 空间。同时以图像、文本说明、Fovea 模糊图像、边缘图四种模态嵌入为对比学习目标，并使用不确定性感知的模态掩码，防止模型记忆单一模态。

2. **对齐训练**（100 轮）：`SimpleAlignPipe`（轻量 MLP）将 EEG-derived CLIP 嵌入对齐到 CLIP 图像嵌入子空间，以冻结的 IP-Adapter 图像编码器输出为对齐目标，消除两者之间的分布差异。

3. **图像生成**：将对齐后的 CLIP 嵌入作为 IP-Adapter 的条件信号，输入 SDXL-Turbo 生成图像（不使用文本提示，输出分辨率 512×512）。

### 模型架构

```
EEG [B, 63, 250]
  └─ EEGProjectLayer
       ├─ Linear(15750→1024) + GELU + Linear(1024→1024) + Dropout(0.3) + LayerNorm
       └─ 通过对比损失对齐至共享 CLIP 空间

多模态监督（四个并行编码器，全部冻结）：
  图像    → CLIP ViT-H-14 → z_image  [1024]
  文本    → CLIP ViT-H-14 → z_text   [1024]
  深度图  → CLIP ViT-H-14 → z_depth  [1024]（FoveaBlur 增强）
  边缘图  → CLIP ViT-H-14 → z_edge   [1024]

SimpleAlignPipe：MLP(1024→1024)，以 IP-Adapter 图像编码器输出为监督目标

生成：SDXL-Turbo + IP-Adapter（IP-Adapter-Plus-Face 变体）
```

### 配置

`local.example.yaml` 是提交到仓库的模板文件。将其复制为 `local.yaml`（已在 `.gitignore` 中，仅保存在本地），然后填写预训练模型权重路径：

```bash
cp task2/configs/local.example.yaml task2/configs/local.yaml
# 编辑 local.yaml，填写 weights_root、sdxl_root、ip_adapter_root
```

**`data_root` 为可选项** — 若 `image-eeg-data/` 放在 `DL_Project/` 根目录（默认布局），代码会自动探测 `image-eeg-data/converted_for_cogcappro/`，无需任何数据准备步骤。仅在数据集不在默认位置时才需显式设置 `data_root`。

### 运行步骤

**步骤 0 — 验证环境**（无需 GPU）：

```bash
cd task2 && python smoke_test.py && cd ..
```

> 以下所有 `sbatch` 命令必须在**仓库根目录**（即包含 `task1/` 和 `task2/` 的目录）下提交，而非在 `task2/` 内部。

**步骤 1 — 准备扩散嵌入**（运行一次）：

```bash
sbatch task2/slurm_scripts/02b_reprepare_diffusion_embeddings.sh
# 或直接运行：
python task2/scripts/prepare_diffusion_embeddings.py
```

**步骤 2 — 训练 EEG 编码器**（80 轮，A40 约 4 小时）：

```bash
python task2/main.py \
    --config task2/configs/cogcappro.yaml \
    --subjects sub-01 \
    --epoch 80 \
    --lr 1e-4 \
    --staged_training \
    --vision_backbone ViT-H-14 \
    --devices 0
```

或通过 SLURM：

```bash
sbatch task2/slurm_scripts/07b_train_retrieval_full_v2.sh
```

**步骤 3 — SimpleAlignPipe 对齐训练**（100 轮）：

```bash
sbatch task2/slurm_scripts/08d_simple_align.sh
```

**步骤 4 — 生成重建图像：**

```bash
sbatch task2/slurm_scripts/09d_generate_fixed.sh
```

**步骤 5 — 评估：**

```bash
sbatch task2/slurm_scripts/10e_eval_full_both.sh
python task2/scripts/summarize_results.py
```

#### 多种子运行（5 个种子，推荐用于稳定结果）

每个阶段均为 SLURM array 作业（`--array=0-4`），5 个种子并行运行。各阶段通过作业依赖顺序提交：

```bash
# 步骤 1：并行训练种子 0–4（每个约 24 小时）
JID_TRAIN=$(sbatch --parsable task2/slurm_scripts/06_multiseed_train.sh)

# 步骤 2：训练全部完成后，并行运行对齐
JID_ALIGN=$(sbatch --parsable --dependency=afterok:${JID_TRAIN} task2/slurm_scripts/07_multiseed_align.sh)

# 步骤 3：一个作业生成所有种子的图像
JID_GEN=$(sbatch --parsable --dependency=afterok:${JID_ALIGN} task2/slurm_scripts/08_multiseed_generate.sh)

# 步骤 4：并行评估所有种子
JID_EVAL=$(sbatch --parsable --dependency=afterok:${JID_GEN} task2/slurm_scripts/09_multiseed_eval.sh)

# 步骤 5：汇总，输出各指标的均值 ± 标准差
sbatch --dependency=afterok:${JID_EVAL} task2/slurm_scripts/10_multiseed_summary.sh
```

结果保存至 `task2/runs/multiseed/summary.json`。

**`main.py` 主要参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `configs/cogcappro.yaml` | 配置文件路径 |
| `--subjects` | `sub-08` | 受试者 ID（课程数据使用 `sub-01`） |
| `--epoch` | `80` | 最大训练轮数 |
| `--lr` | `1e-4` | 学习率 |
| `--staged_training` | 关闭 | 3 阶段训练（20+40+20 轮） |
| `--vision_backbone` | `RN50` | CLIP 主干（推荐 `ViT-H-14`） |
| `--uncertainty_aware` | 关闭 | 启用不确定性感知模态掩码 |
| `--devices` | `0,1` | GPU 设备编号 |

### 任务 2 实验结果

单次运行（sub-01，种子 0）。以下为 SimpleAlignPipe + SDXL-Turbo 生成（`all` 模式）的结果。

#### SimpleAlignPipe 消融对比

| 指标 | 不经过 SimpleAlignPipe | **经过 SimpleAlignPipe** | 变化 |
|------|------------------------|--------------------------|------|
| **SSIM** | 0.3106 | **0.3732** | +0.063 |
| **CLIP Score（ViT-H-14）** | 0.7160 | **0.8981** | +0.182 |
| PixCorr | 0.131 | 0.159 | +0.028 |
| AlexNet-2 | 0.662 | 0.782 | +0.120 |
| AlexNet-5 | 0.690 | 0.889 | +0.199 |
| Inception | 0.621 | 0.810 | +0.189 |
| EfficientNet | 0.941 | 0.835 | −0.106 |
| SwAV | 0.695 | 0.533 | −0.162 |

SimpleAlignPipe 消除了 EEG 派生的 CLIP 嵌入与图像空间 CLIP 嵌入之间的分布差距。语义类指标（CLIP、Inception、AlexNet）显著提升；而 EfficientNet 和 SwAV——分别捕捉低级纹理特征和自监督特征——略有下降，说明对齐操作使生成结果向语义内容倾斜，而非像素级保真度。

**检索（任意模态融合，作为辅助输出）：** Top-1 61.5%，Top-5 89.0%

---

## 外部资源声明

本项目使用了以下预训练模型和开源代码库：

| 资源 | 用途 |
|------|------|
| OpenCLIP RN50（OpenAI） | 任务 1 图像特征提取 |
| OpenCLIP ViT-H-14（LAION-2B） | 任务 2 多模态监督 |
| EVNet（Ponce et al., 2023） | 任务 1 仿生视觉前端 |
| SDXL-Turbo（Stability AI） | 任务 2 图像生成 |
| IP-Adapter（Ye et al., 2023） | 任务 2 图像条件控制 |
| VisualEEGDecoding（Liu et al.） | 任务 1 多尺度模糊检索方案 |
| CognitionCapturerPro | 任务 2 多模态 EEG→图像流程 |

---

## 局限性

**任务 1：**
- 仅限单受试者（sub-01），未评估跨受试者泛化能力。
- EVNet 适配层以随机初始化状态冻结，端到端可学习的适配层可能进一步提升性能。
- 基于 ViT 的 CLIP 编码器与 EVNet 的空间化预处理不兼容。

**任务 2：**
- 仅限单受试者（sub-01）。多种子运行（5 个种子）已通过 `task2/slurm_scripts/06–10_multiseed_*.sh` 支持，详见上方多种子运行章节。
- 重建以从训练集检索到的图像作为 IP-Adapter 条件，而非直接从 EEG 解码图像内容。语义相近但结构不同的训练图像可能导致生成结果偏离目标。
- SDXL-Turbo（1–4 步去噪）以速度换取生成质量。
- 未使用文本提示；引入类别文本提示可能改善语义保真度。
