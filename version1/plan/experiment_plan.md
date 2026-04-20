# Project 1: Brain-to-Image — 实验方案与消融实验计划

## 项目概述

DSAA2012 课程 Project A：基于 EEG 脑电信号的图像检索与重建。

- **Task 1 (Retrieval):** 给定 EEG 片段，在 200 张候选图片中找到正确的刺激图片
- **Task 2 (Reconstruction):** 给定 EEG 片段，生成与刺激图片在结构和语义上一致的图片

**数据集:** 63 通道 EEG，250 时间步，训练集 ~1654 类 (80 trials)，测试集 200 类

**评估指标:**
- Retrieval: Top-1 Accuracy, Top-5 Accuracy
- Reconstruction: PixCorr, SSIM, AlexNet(2), AlexNet(5), Inception, CLIP, EfficientNet, SwAV

---

## 整体实验框架

```
                    ┌──────────────────────────────────────────┐
                    │           大消融实验 (Macro Ablation)       │
                    ├────────────────────┬─────────────────────┤
                    │   架构 A: 共享编码器  │  架构 B: 独立编码器   │
                    │  (Shared Encoder)  │ (Separate Encoders) │
                    └────────┬───────────┴──────────┬──────────┘
                             │                      │
                    ┌────────▼──────────┐  ┌────────▼──────────┐
                    │  小消融实验 (Micro) │  │  小消融实验 (Micro) │
                    │  - EEG Encoder    │  │  - EEG Encoder    │
                    │  - Image Encoder  │  │  - Image Encoder  │
                    │  - Retrieval 模型  │  │  - Retrieval 模型  │
                    │  - Generation 模型 │  │  - Generation 模型 │
                    └───────────────────┘  └───────────────────┘
```

---

## 一、大消融实验：共享编码器 vs 独立编码器

### 架构 A：共享编码器 (Shared Encoder)

一个 EEG Encoder + 一个 Image Encoder 同时服务 Task 1 和 Task 2，用两个任务 loss 的加权和进行联合训练。

```
EEG [B, 63, 250]
      │
  EEG Encoder (共享)
      │
  EEG Embedding [B, D]
      │
      ├──→ Task 1: Retrieval ──→ L_retrieval
      │
      └──→ Task 2: Reconstruction ──→ L_reconstruction
      
Total Loss = α × L_retrieval + β × L_reconstruction
```

**优势:** 两个任务共享表征，可能互相增强；参数量更少，训练效率高

**风险:** 两个任务可能存在目标冲突，需要仔细调节 α 和 β

**超参消融:** 尝试不同的 α:β 比例（如 1:1, 1:0.5, 0.5:1 等）

### 架构 B：独立编码器 (Separate Encoders)

两个独立的 EEG Encoder + Image Encoder 分别处理两个任务，各自用独立的 loss 进行反向传播。

```
EEG [B, 63, 250]          EEG [B, 63, 250]
      │                         │
  EEG Encoder 1             EEG Encoder 2
      │                         │
  Embedding 1 [B, D]       Embedding 2 [B, D]
      │                         │
  Task 1: Retrieval         Task 2: Reconstruction
      │                         │
  L_retrieval               L_reconstruction
  (独立反向传播)              (独立反向传播)
```

**优势:** 每个编码器可以针对各自任务充分优化，不存在任务冲突

**风险:** 参数量翻倍；无法利用任务间的互补信息

### 大消融对比维度

| 对比维度 | 架构 A (共享) | 架构 B (独立) |
|---------|-------------|-------------|
| 参数量 | 1x | 2x |
| 训练方式 | 联合 loss | 分别训练 |
| Task 1 指标 | Top-1, Top-5 | Top-1, Top-5 |
| Task 2 指标 | SSIM, CLIP 等 | SSIM, CLIP 等 |
| 训练时间 | 较短 | 较长 |
| 是否存在任务冲突 | 可能 | 无 |

---

## 二、小消融实验：模块级对比

以下小消融实验需要在**两种架构下都进行**，确保对比的完整性。

### 2.1 EEG Encoder 消融

对 EEG 信号编码部分尝试不同的模型架构：

| 编号 | 模型 | 说明 |
|------|------|------|
| E1 | CNN + Transformer | 卷积提取局部时空特征，Transformer 建模全局依赖 |
| E2 | Pure CNN (ResNet-style) | 全卷积网络，深层残差连接 |
| E3 | Pure Transformer | 直接将 EEG patch 作为 token 序列输入 Transformer |
| E4 | LSTM / GRU | 经典 RNN 方案，适合时间序列 |
| E5 | EEGNet / ShallowConvNet | EEG 领域的经典轻量模型 |
| E6 | MoE + MLA Transformer | Mixture of Experts 让不同专家网络处理不同 EEG 模式；Multi-head Latent Attention (DeepSeek-V2 风格) 将 KV 压缩到低维潜空间，提升参数效率。适合在有限数据上用更大容量但更稀疏的模型 |

### 2.2 Image Encoder 消融 (用于提取图片 embedding 作为对比学习 target)

| 编号 | 模型 | Embedding 维度 | 说明 |
|------|------|---------------|------|
| I1 | CLIP ViT-L/14 | 768 | 语义对齐能力强，社区广泛使用 |
| I2 | CLIP ViT-B/32 | 512 | 更轻量的 CLIP 版本 |
| I3 | DINOv2 ViT-L | 1024 | 自监督视觉特征，结构信息更丰富 |
| I4 | ImageNet-pretrained ViT | 768/1024 | 分类预训练的视觉 Transformer |
| I5 | ResNet-50 (ImageNet) | 2048 | 经典 CNN baseline |

### 2.3 Task 1 Retrieval 方法消融

| 编号 | 方法 | 说明 |
|------|------|------|
| R1 | CLIP 对齐 + Cosine Similarity | EEG embedding 对齐到 CLIP 空间，余弦相似度排序 |
| R2 | 可学习度量网络 (Metric Learning) | 用 MLP 计算 EEG-Image pair 的匹配分数 |
| R3 | Cross-Attention Fusion | EEG 和 Image 特征通过 cross-attention 交互后打分 |
| R4 | KNN with learned features | 使用 k-近邻在特征空间中检索 |

### 2.4 Task 2 Reconstruction 方法消融

| 编号 | 方法 | 类型 | 说明 |
|------|------|------|------|
| G1 | IP-Adapter + Stable Diffusion | Diffusion (UNet) | EEG embedding 作为 IP-Adapter 输入引导扩散模型 |
| G2 | ControlNet + Stable Diffusion | Diffusion (UNet) | 用 ControlNet 条件控制生成 |
| G3 | FLUX | Diffusion (DiT / Flow Matching) | Black Forest Labs 的 rectified flow 模型，使用 DiT 架构，生成质量极高；通过条件注入 EEG embedding 引导生成 |
| G4 | GAN (条件 GAN / StyleGAN) | GAN | 经典对抗生成网络 |
| G5 | Bagel | Autoregressive | ByteDance 的统一多模态模型，自回归方式生成图像 token；将 EEG embedding 作为条件 prefix 输入 |
| G6 | VQGAN + Transformer | Autoregressive | 先将图像量化为离散 token，再用 Transformer 自回归预测 |
| G7 | VAE-based (DALL-E decoder style) | VAE | 变分自编码器方案 |
| G8 | Versatile Diffusion | Diffusion | 多条件扩散模型，支持 image/text 条件 |

---

## 三、实验矩阵总览

```
                               架构 A (共享)    架构 B (独立)
                               ───────────    ───────────
EEG Encoder:
  E1 CNN+Transformer              ✓              ✓
  E2 Pure CNN                     ✓              ✓
  E3 Pure Transformer             ✓              ✓
  E6 MoE+MLA Transformer         ✓              ✓
  ...                             ...            ...

Image Encoder:
  I1 CLIP ViT-L/14               ✓              ✓
  I2 DINOv2                      ✓              ✓
  ...                             ...            ...

Retrieval:
  R1 Cosine Similarity            ✓              ✓
  R2 Metric Learning              ✓              ✓
  ...                             ...            ...

Reconstruction (按类型):
  [Diffusion]  G1 IP-Adapter+SD   ✓              ✓
  [Diffusion]  G3 FLUX            ✓              ✓
  [GAN]        G4 cGAN/StyleGAN   ✓              ✓
  [AR]         G5 Bagel           ✓              ✓
  [AR]         G6 VQGAN+Trans     ✓              ✓
  ...                             ...            ...
```

**注意:** 小消融实验应在两种架构下都进行，最终在报告中统一对比，以验证结论的一致性。

---

## 四、分工方案

### Person A：架构 A — 共享编码器，完成两个任务

**职责:**
- 实现共享 EEG Encoder + Image Encoder 的联合训练框架
- 设计联合 loss: `L_total = α × L_retrieval + β × L_reconstruction`
- 在共享编码器架构下完成 Task 1 (Retrieval) 和 Task 2 (Reconstruction)
- 调节 loss 权重 α 和 β

**产出:**
- 架构 A 的完整训练 + 推理代码
- 架构 A 在两个任务上的完整评估指标
- 报告中架构 A 的方法描述和实验分析

### Person B：架构 B 中的 Encoder 1 — 完成 Task 1 (Retrieval)

**职责:**
- 实现架构 B 中专用于 Retrieval 的 EEG Encoder
- 完成 Task 1 的训练和评估（Top-1, Top-5）
- 在 Task 1 范围内进行小消融实验：
  - EEG Encoder 消融 (E1-E5)
  - Image Encoder 消融 (I1-I5)
  - Retrieval 方法消融 (R1-R4)

**产出:**
- 架构 B 中 Task 1 的完整代码
- Retrieval 相关的全部消融实验结果
- 报告中 Retrieval 部分的方法和实验

### Person C：架构 B 中的 Encoder 2 — 完成 Task 2 (Reconstruction)

**职责:**
- 实现架构 B 中专用于 Reconstruction 的 EEG Encoder
- 完成 Task 2 的训练和图像生成
- 在 Task 2 范围内进行小消融实验：
  - EEG Encoder 消融 (E1-E5)
  - Image Encoder 消融 (I1-I5)
  - 生成模型消融 (G1-G8: Diffusion / GAN / Autoregressive / VAE)

**产出:**
- 架构 B 中 Task 2 的完整代码
- Reconstruction 相关的全部消融实验结果
- 报告中 Reconstruction 部分的方法和实验

### 协作流程

```
Phase 1: 各自开发 (并行)
─────────────────────────
Person A ──→ 架构 A 共享编码器 + 两任务
Person B ──→ 架构 B Encoder 1 + Retrieval + 小消融
Person C ──→ 架构 B Encoder 2 + Reconstruction + 小消融

Phase 2: 交叉消融 (需要协调)
─────────────────────────
Person B 的小消融实验 ──→ 也在架构 A 下跑一遍
Person C 的小消融实验 ──→ 也在架构 A 下跑一遍
Person A 提供架构 A 的接口供 B/C 调用

Phase 3: 代码重构 + 统一 (合作)
─────────────────────────
- 统一代码接口和目录结构
- 确保所有小消融在两种架构下结果完整
- 合并评估结果，完成报告
```

---

## 五、最终代码结构 (重构后)

```
DL_Project/
├── image-eeg-data/                    # 原始数据 (不修改)
│
├── src/
│   ├── data.py                        # 数据加载 + 增强 (共用)
│   ├── utils.py                       # 公共工具函数 (共用)
│   │
│   ├── encoders/                      # EEG Encoder 模块
│   │   ├── cnn_transformer.py         # E1: CNN + Transformer
│   │   ├── pure_cnn.py               # E2: Pure CNN
│   │   ├── pure_transformer.py        # E3: Pure Transformer
│   │   ├── rnn.py                     # E4: LSTM / GRU
│   │   ├── eegnet.py                 # E5: EEGNet
│   │   └── moe_mla_transformer.py    # E6: MoE + MLA Transformer
│   │
│   ├── image_encoders/                # Image Encoder 模块
│   │   └── clip_features.py           # I1-I5: 各种 Image Encoder 特征缓存
│   │
│   ├── retrieval/                     # Task 1 方法
│   │   ├── cosine_retrieval.py        # R1: Cosine Similarity
│   │   ├── metric_learning.py         # R2: Metric Learning
│   │   └── cross_attention.py         # R3: Cross-Attention
│   │
│   ├── reconstruction/                # Task 2 方法
│   │   ├── ip_adapter_sd.py           # G1: IP-Adapter + SD
│   │   ├── controlnet_sd.py           # G2: ControlNet + SD
│   │   ├── flux.py                    # G3: FLUX (DiT / Flow Matching)
│   │   ├── cgan.py                    # G4: Conditional GAN
│   │   ├── bagel.py                   # G5: Bagel (Autoregressive)
│   │   ├── vqgan_transformer.py       # G6: VQGAN + Transformer (AR)
│   │   ├── vae_decoder.py            # G7: VAE-based
│   │   └── versatile_diffusion.py    # G8: Versatile Diffusion
│   │
│   ├── architectures/                 # 两种大架构
│   │   ├── shared_encoder.py          # 架构 A: 共享编码器 + 联合 loss
│   │   └── separate_encoders.py       # 架构 B: 独立编码器 + 分别 loss
│   │
│   ├── train.py                       # 统一训练入口
│   └── evaluate.py                    # 统一评估入口
│
├── configs/                           # 实验配置文件
│   ├── shared_e1_r1_g1.yaml           # 架构A + CNN-Trans + Cosine + IP-Adapter
│   ├── separate_e1_r1_g1.yaml         # 架构B + CNN-Trans + Cosine + IP-Adapter
│   └── ...                            # 各种消融组合
│
├── checkpoints/                       # 模型权重
├── outputs/                           # 生成图片和评估结果
└── scripts/                           # HPC 提交脚本
    ├── run_shared.sh
    └── run_separate.sh
```

---

## 六、实验优先级

鉴于时间有限，建议按以下优先级推进：

| 优先级 | 实验 | 说明 |
|--------|------|------|
| P0 (必须完成) | 架构 A 共享编码器 + 默认配置 (E1+I1+R1+G1) | 保证两个任务都有有效产出 |
| P0 (必须完成) | 架构 B 独立编码器 + 默认配置 (E1+I1+R1+G1) | 完成大消融对比 |
| P1 (高优先) | EEG Encoder 消融 (E1-E3) | 核心模块，影响最大 |
| P1 (高优先) | Reconstruction 方法消融: Diffusion 类 (G1-G3) | G1 IP-Adapter + G3 FLUX 对比，直接影响 25 分 |
| P2 (中优先) | Image Encoder 消融 (I1-I3) | 可能有显著提升 |
| P2 (中优先) | Retrieval 方法消融 (R1-R2) | Cosine baseline 已经很强 |
| P2 (中优先) | Reconstruction 方法消融: GAN vs AR (G4-G6) | G4 cGAN + G5 Bagel + G6 VQGAN，跨范式对比 |
| P3 (时间允许) | EEG Encoder: MoE+MLA (E6) 和经典模型 (E4-E5) | 丰富报告，展示架构探索深度 |
| P3 (时间允许) | 更多 Generation 模型 (G7-G8) | VAE 和 Versatile Diffusion 补充 |

---

## 七、时间线

| 日期 | 里程碑 |
|------|--------|
| 4月第1周 | 环境搭建 + 数据管道 + CLIP 特征缓存 |
| 4月第2周 | 各自完成默认配置 (P0)，确保两个任务都能跑通 |
| 4月第3周 | 推进 P1/P2 小消融实验 |
| 4月第4周 (4.28 前) | 汇总结果，准备 Presentation |
| 5月第1周 | 补充 P3 实验 + 交叉消融 + 代码重构 |
| 5月10日 | 提交最终报告 + 代码 |
