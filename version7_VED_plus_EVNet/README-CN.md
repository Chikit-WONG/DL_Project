# 版本 7：VED + EVNet — 脑电图到图像检索

[← 返回项目根目录](../README-CN.md) | [English README](README.md)

本版本在 [`version5_VED`](../version5_VED/README-CN.md) 的多尺度模糊 CLIP 流程基础上，引入仿生 EVNet 视觉前端进行扩展。**仅涵盖任务 1（EEG 图像检索）**；任务 2 重建请参阅 `version5_VED` 或 `version6_BP-MGD`。

## 目录

- [方法介绍](#方法介绍)
- [模型架构](#模型架构)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [运行流程](#运行流程)
- [实验结果](#实验结果)
- [消融实验](#消融实验)
- [局限性](#局限性)

---

## 方法介绍

本项目的核心任务是 **EEG 到图像检索**：给定受试者观看某张图像时采集的 EEG 信号，从 200 张候选图像中识别出对应图像。

**第 7 版**在基础 VED 框架的基础上引入了 **EVNet 固定前端**。EVNet 模拟灵长类动物的早期视觉通路——具体为视网膜/外侧膝状核（皮层下处理，SubcorticalBlock）以及初级视觉皮层 V1（VOneBlock，含 Gabor 滤波器），并以**全参数冻结**的方式作为图像特征提取器使用。其输出与多尺度高斯模糊 CLIP 特征进行加权融合，构成更丰富的图像表征，供 EEG 编码器与之对齐。

融合方式：

```
fused = w₀ · blur_agg + w₁ · evnet_feat
```

其中 `w₀`、`w₁` 是经 softmax 归一化的可学习标量（初始化为 0.7 / 0.3），`blur_agg` 是多尺度模糊特征栈经注意力加权后的聚合向量。

---

## 模型架构

### 图像特征提取（离线预计算）

```
输入图像 (224×224)
       │
       ├──── 多尺度高斯模糊（8 或 12 个级别）
       │            └── CLIP 编码器（RN50 或 ViT-H/14）
       │                   └── 模糊特征栈  [num_levels × 1024维]
       │
       └──── EVNet 前端（全参数冻结）
                  ├── SubcorticalBlock   （视网膜 / LGN）
                  ├── VOneBlock          （V1 Gabor 滤波器）
                  ├── Conv2d 适配层      （512→3 通道，Kaiming 初始化，冻结）
                  └── CLIP 编码器（RN50 或 ViT-H/14）
                         └── EVNet 特征  [1024维]
```

**模糊级别预设：**

| 配置 | 级别 |
|------|------|
| 8-blur | `l_1, l_3, l_15, l_21, l_33, l_45, l_57, l_63` |
| 12-blur | `l_1, l_3, l_9, l_15, l_21, l_27, l_33, l_39, l_45, l_51, l_57, l_63` |

### EEG 编码器（`Brain_Visual_Encoder_EEG`）

```
EEG 输入 [B, 63通道, 250时间步]
       │
       ├── Conv2dWithAbs  （空间卷积：63ch → 25 个滤波器）
       ├── BatchNorm2d
       ├── Linear(250→200) + ELU + Dropout(0.25)
       ├── Linear(200→200) + ELU + Dropout(0.65)
       └── Linear(25×200 → 1152维)   ← EEG 嵌入向量
```

### 融合与损失函数

训练时模型接收预计算的 `blur_stack` 和 `evnet_feat` 张量，图像分支计算如下：

```
blur_agg = Σ softmax(learned_scale) · blur_stack    # 对模糊级别做注意力加权
fused    = softmax(fusion_logits) · [blur_agg, evnet_feat]
img_emb  = fusion_adapter(fused)                    # MLP: 1152→768→1152
```

损失函数：**InfoNCE**（对称对比损失），在 EEG 嵌入与图像嵌入之间计算。

---

## 环境配置

### 依赖要求

- Python 3.9+
- PyTorch ≥ 2.0（含 CUDA 支持）
- `open-clip-torch`
- `numpy`、`scipy`、`pandas`、`tqdm`、`opencv-python`、`Pillow`
- EVNet（已内置于 `evnet/` 目录）

### 安装步骤

```bash
# 1. 激活环境
conda activate test   # 或新建一个环境

# 2. 安装 Python 依赖
pip install torch torchvision open-clip-torch numpy scipy pandas tqdm opencv-python Pillow

# 3. EVNet 通过相对路径直接导入，process_image_course.py 会自动将 evnet/ 加入 sys.path，
#    无需单独安装。
```

### CLIP 模型权重

| 主干网络 | 文件名 | 大小 |
|----------|--------|------|
| RN50（OpenAI） | `open_clip_pytorch_model.bin` | ~102 MB |
| ViT-H/14（LAION-2B） | `open_clip_pytorch_model.bin` | ~3.9 GB |

将权重文件放置在任意目录，并通过 `--clip_checkpoint` 参数指定路径。

---

## 数据准备

### EEG 数据

默认目录结构（也可通过 `--eeg_data_dir` 直接指定）：

```
Preprocessed_data_250Hz_whiten/
└── sub-01/
    ├── train.pt    # 字典：{'eeg': Tensor[N,1,63,250], 'img': array[N,k,路径]}
    └── test.pt
```

图像目录软链接已配置完毕：

```
data/things-eeg/Image_set/train_images -> .../image-eeg-data/training_images
data/things-eeg/Image_set/test_images  -> .../image-eeg-data/test_images
```

### 图像特征文件

预计算特征存储在 `output/Image_feature/`，文件大小参考：

| 文件 | 说明 | 大小 |
|------|------|------|
| `MultiBlur_RN50_train.pt` | 8/12 级模糊，RN50（软链接至 v5） | — |
| `MultiBlur_RN50_test.pt` | | — |
| `EVNet_RN50_train.pt` | EVNet + RN50 特征 | ~67 MB |
| `EVNet_RN50_test.pt` | | ~896 KB |
| `MultiBlur_ViTH14_train.pt` | 8/12 级模糊，ViT-H/14 | ~791 MB |
| `MultiBlur_ViTH14_test.pt` | | ~9.7 MB |
| `EVNet_ViTH14_train.pt` | EVNet + ViT-H/14 特征 | ~67 MB |
| `EVNet_ViTH14_test.pt` | | ~896 KB |
| `EVNet_xavier_RN50_*.pt` | Xavier 初始化适配层变体 | 训练集 ~67 MB |
| `EVNet_gap_*.pt` | GAP + 线性层（无 CLIP 主干） | 训练集 ~67 MB |

---

## 运行流程

### 第一步：生成图像特征

```bash
# RN50 + EVNet（随机/Kaiming 初始化适配层）
python preprocess/process_image_course.py \
    --clip_checkpoint /path/to/CLIP-RN50/open_clip_pytorch_model.bin \
    --save_dir output/Image_feature \
    --backbone rn50 \
    --evnet_mode random \
    --batch_size 128

# ViT-H/14 + EVNet
python preprocess/process_image_course.py \
    --clip_checkpoint /path/to/CLIP-ViT-H-14/open_clip_pytorch_model.bin \
    --save_dir output/Image_feature \
    --backbone vit_h_14 \
    --evnet_mode random \
    --batch_size 64
```

`--evnet_mode` 选项说明：

| 模式 | 说明 |
|------|------|
| `random` | Kaiming 正态初始化 Conv2d 适配层 |
| `xavier` | Xavier 均匀初始化 Conv2d 适配层 |
| `gap` | 全局平均池化 + 线性层，不使用 CLIP 主干 |

输出文件命名规则：

| `--backbone` | `--evnet_mode` | 模糊特征前缀 | EVNet 特征前缀 |
|---|---|---|---|
| `rn50` | `random` | `MultiBlur_RN50` | `EVNet_RN50` |
| `rn50` | `xavier` | `MultiBlur_RN50` | `EVNet_xavier_RN50` |
| `rn50` | `gap` | `MultiBlur_RN50` | `EVNet_gap` |
| `vit_h_14` | `random` | `MultiBlur_ViTH14` | `EVNet_ViTH14` |

### 第二步：训练

```bash
# 8-blur + EVNet 固定，RN50，95/5 划分
python main_eeg_course.py \
    --blur_config 8 \
    --use_evnet \
    --epoch 200 \
    --train_batch_size 1024 \
    --lr 0.001 \
    --n_seeds 10 \
    --first_seed 21 \
    --feature_path output/Image_feature \
    --output_dir output/logs/8blur_evnet_split

# 全量训练（不划分验证集）
python main_eeg_course.py \
    --blur_config 8 \
    --use_evnet \
    --use_full_train \
    --epoch 200 \
    --train_batch_size 1024 \
    --lr 0.001 \
    --n_seeds 10 \
    --first_seed 21 \
    --feature_path output/Image_feature \
    --output_dir output/logs/8blur_evnet_full

# ViT-H/14 主干变体
python main_eeg_course.py \
    --blur_config 8 \
    --use_evnet \
    --blur_prefix MultiBlur_ViTH14 \
    --evnet_prefix EVNet_ViTH14 \
    --epoch 200 \
    --train_batch_size 1024 \
    --lr 0.001 \
    --n_seeds 10 \
    --first_seed 21 \
    --feature_path output/Image_feature \
    --output_dir output/logs/8blur_evnet_vith14_split
```

主要参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--blur_config` | `8` | 模糊级别预设：`8` 或 `12` |
| `--use_evnet` | 关闭 | 启用 EVNet 特征融合 |
| `--blur_prefix` | `MultiBlur_RN50` | 模糊特征 `.pt` 文件前缀 |
| `--evnet_prefix` | `EVNet_RN50` | EVNet 特征 `.pt` 文件前缀 |
| `--use_full_train` | 关闭 | 使用全量训练集（不划分验证集） |
| `--epoch` | 200 | 训练轮数 |
| `--train_batch_size` | 1024 | 批大小 |
| `--lr` | 0.001 | 学习率 |
| `--n_seeds` | 10 | 随机种子数量 |
| `--first_seed` | 21 | 起始种子值（实际种子为 first_seed … first_seed+n_seeds-1） |
| `--eeg_data_dir` | — | 覆盖 EEG 数据路径 |

### SLURM 脚本

预配置的 SLURM 脚本位于 `slurm_scripts/`：

| 脚本 | 说明 |
|------|------|
| `01_gen_evnet_features.sh` | 生成 RN50 + EVNet 特征 |
| `02_train_8blur_evnet_split.sh` | 8-blur + EVNet，RN50，95/5 划分 |
| `03_train_12blur_evnet_split.sh` | 12-blur + EVNet，RN50，95/5 划分 |
| `04_full_train_8blur_evnet.sh` | 8-blur + EVNet，RN50，全量训练 |
| `05_full_train_12blur_evnet.sh` | 12-blur + EVNet，RN50，全量训练 |
| `06_gen_evnet_xavier_features.sh` | 生成 Xavier 初始化适配层特征 |
| `07_gen_evnet_gap_features.sh` | 生成 GAP + 线性层特征 |
| `08_train_8blur_evnet_xavier_split.sh` | 消融：Xavier 初始化 |
| `09_train_8blur_evnet_gap_split.sh` | 消融：GAP + 线性层（无主干） |
| `10_gen_vith14_features.sh` | 生成 ViT-H/14 特征 |
| `11_train_8blur_evnet_vith14_split.sh` | 消融：ViT-H/14 主干 |

---

## 实验结果

所有实验：10 个随机种子（种子 21–30），200 轮训练，批大小 1024，学习率 0.001，单受试者（sub-01），在 200 路图像检索任务上评估。

**Val-selected（验证集选择）**：按最佳验证集 Top-1 选取检查点，在测试集上评估。  
**Best-test（最佳测试）**：所有轮次中测试集 Top-1 的最高值。

### 主实验（95/5 训练/验证划分）

| 实验 | 验证选 Top-1 | 验证选 Top-5 | 最佳测试 Top-1 | 最佳测试 Top-5 |
|---|---|---|---|---|
| 8-blur + EVNet 固定（RN50） | 0.8460 ± 0.0135 | 0.9870 ± 0.0059 | 0.8715 ± 0.0091 | 0.9860 ± 0.0081 |
| 12-blur + EVNet 固定（RN50） | 0.8400 ± 0.0186 | 0.9860 ± 0.0046 | 0.8715 ± 0.0111 | 0.9855 ± 0.0028 |

### 全量训练集（不划分验证集）

| 实验 | 验证选 Top-1 | 验证选 Top-5 | 最佳测试 Top-1 | 最佳测试 Top-5 |
|---|---|---|---|---|
| 8-blur + EVNet 固定（RN50） | 0.8530 ± 0.0136 | 0.9860 ± 0.0046 | 0.8785 ± 0.0082 | 0.9855 ± 0.0037 |
| 12-blur + EVNet 固定（RN50） | 0.8505 ± 0.0169 | 0.9845 ± 0.0037 | 0.8810 ± 0.0074 | 0.9850 ± 0.0041 |

全量训练集相较于 95/5 划分，最佳测试 Top-1 提升约 0.007–0.010。

---

## 消融实验

所有消融均以"8-blur + EVNet 固定（RN50，95/5 划分）"为基准。

| 消融方案 | 验证选 Top-1 | 最佳测试 Top-1 | 相对基准（验证选） |
|---|---|---|---|
| **基准**：EVNet 固定，Kaiming 初始化（RN50） | 0.8460 ± 0.0135 | 0.8715 ± 0.0091 | — |
| Xavier 初始化适配层（RN50） | 0.8275 ± 0.0175 | 0.8495 ± 0.0086 | −0.019 |
| GAP + 线性层（无 CLIP 主干） | 0.8285 ± 0.0173 | 0.8620 ± 0.0092 | −0.018 |
| EVNet 固定，Kaiming 初始化（ViT-H/14） | 0.7365 ± 0.0208 | 0.7790 ± 0.0115 | −0.110 |

**主要结论：**

- **Kaiming 优于 Xavier**：Kaiming 正态初始化的冻结 Conv2d 适配层比 Xavier 均匀初始化高约 0.019 的验证 Top-1。Xavier 初始化产生的权重幅值较小，可能导致冻结后的适配层表达能力不足。

- **GAP 消融**：完全去除 CLIP 主干（改用 AdaptiveAvgPool2d + 线性层）后，验证 Top-1 仅下降约 0.018，说明 EVNet 的 V1 类特征本身已包含大量有效信息。最佳测试 Top-1 的差距仅为 0.009。

- **ViT-H/14 明显劣于 RN50**（验证 Top-1 下降 0.110）。ViT-H/14 采用基于分块（patch）的自注意力机制，期望输入为干净的像素图像块；EVNet 的卷积适配层输出的是经过空间变换的特征图，破坏了 ViT 赖以工作的 token 结构。RN50 作为 CNN 主干网络，天然兼容 EVNet 的空间化输出。

---

## 局限性

1. **仅限单受试者。** 所有实验均使用受试者 sub-01，未评估跨受试者的泛化能力。

2. **ViT-H/14 与 EVNet 不兼容。** EVNet 适配层（SubcorticalBlock → VOneBlock → Conv2d）产生的图像适合 CNN 主干处理，而 Transformer 类 CLIP 编码器（ViT-*）将图像视为固定大小的分块序列，无法有效处理 EVNet 预处理后的输入，导致 Top-1 准确率下降约 11 个百分点。

3. **EVNet 适配层在随机初始化后即冻结。** Conv2d 适配层权重以 Kaiming 正态（默认）或 Xavier 均匀方式初始化后立即冻结，不参与训练。若采用端到端可学习的适配层，有望进一步提升性能。

4. **无跨受试者或跨时段验证。** 95/5 划分与全量训练均属于受试者内场景，未涉及时间泛化（不同采集时段）的评估。

5. **单模态。** 模型仅接受 EEG 信号作为输入，未探索多模态脑信号（如 fMRI）或更丰富的 EEG 采集范式。

6. **仅使用高斯模糊作为图像退化方式。** 多尺度特征仅基于高斯模糊，其他感知相关的变换（频率掩蔽、空间相位打乱等）尚未探索，可能具有互补价值。

7. **评估仅限于检索任务。** 评测指标为 200 路强制选择检索的 Top-1/3/5 准确率，未涉及图像生成或语义相似度等更广泛的评估维度。
