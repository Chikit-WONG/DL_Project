# DSAA2012 Project A: HMEG-Dual-Prior (最终大一统工程白皮书)

**项目目标**：在 THINGS-EEG 数据集上实现零样本检索（Top-1 > 20%）与结构化重建（SSIM > 0.310）的双重 SOTA。
**核心理念**：最强的时空特征提取（硬核物理先验） + 最稳健的图像生成（分层软约束）。
**定稿状态**：经由 Gemini, Claude, ChatGPT 三方交叉核对，架构与依赖 100% 锁定。

---

## 一、 核心架构设计 (Architecture)

### 1. 增强型编码器 (NERV-style Encoder)
* **物理位置注入**：根据 `EEG_CHANNELS.jsonl`，注入 63 个电极的 3D 球面坐标 PE。
* **双路卷积 (Dual-Path Conv)**：并行 ST（空间-时间）和 TS（时间-空间）分支。
* **脑区门控 (Region-aware Gating)**：对 63 通道进行可学习加权，初始权重偏向枕叶与顶叶。
* **个体适配 (Subject Adapter)**：加入轻量级 Token 解决跨受试者差异。

### 2. 任务双头 (Dual-Head Strategy)
* **Semantic Head (语义头)**：
    * **主力对齐**：`CLIP ViT-H/14`（稳住现有检索基线）。
    * **多视角补充**：`CLIP ViT-B/32` & `RN50`（提供额外视觉统计与鲁棒性）。
* **Structural Head (结构头)**：
    * **对齐目标**：图像的 `VAE Latent` (4x64x64)。
    * **损失函数**：仅使用 `Smooth L1 Loss` (防止显存溢出与过拟合)。

---

## 二、 生成管线与分步训练 (Pipeline & Training)

### 1. 两阶段软约束生成 (Two-Stage Generation)
* **Stage 1 (先验提纯)**：用轻量级 Prior U-Net (~20M) 将含噪 EEG 语义特征映射为纯净的 CLIP 图像先验。
* **Stage 2 (SDXL-Turbo 渲染)**：
    * **软结构底图 (img2img)**：Structural Head 输出的 VAE Latent 解码为模糊底图 (Strength=0.5)。
    * **硬语义控制 (IP-Adapter)**：纯净的 CLIP 先验注入单路 IP-Adapter 控制内容与材质。

### 2. 三阶段热身训练策略 (Warm-up Strategy)
* **Epoch 0 - 20 (基线热身)**：仅激活主力语义损失 $L_{InfoNCE}(H14)$ + MSE。
* **Epoch 21 - 40 (分层对齐)**：平滑引入 $L_{InfoNCE}(B32/RN50)$ + $L_{SmoothL1}(VAE)$。
* **Epoch 41+ (精修期)**：加入 $L_{HardNegative}$ 难负样本加权提升区分度。

---

## 三、 敏捷执行甘特图 (7-Day Sprint)

* **Day 1**: 预缓存 H/14, B32, RN50, VAE Latent 特征；打通 SDXL-Turbo 管道。
* **Day 2-3**: 在 `model.py` 中实装 Dual-Path Conv、位置编码与脑区门控。
* **Day 4-5**: 打通双头训练。前 20 epoch 确保 Top-1 检索基线突破 18%。
* **Day 6**: 冻结 Encoder，单独训练 Prior Diffusion。
* **Day 7**: 集成 img2img + IP-Adapter，跑满 80-trial average，定稿最终 SSIM/PixCorr 指标。

---

## 四、 核心参考文献与依赖 (Definitive References)

### 1. 核心理论基石 (Core Literature)
* **ATM (框架基线)**: Li et al. (2024). *Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion*. NeurIPS 2024. [[arXiv]](https://arxiv.org/abs/2403.07721) | [[GitHub]](https://github.com/dongyangli-del/EEG_Image_decode)
* **HMEG (多视角+VAE思想)**: Anonymous. (2026). *Learning Brain Representation with Hierarchical Visual Embeddings*. [[arXiv]](https://arxiv.org/abs/2602.07495)
* **结构化表征对齐**: Anonymous. (2026). *Aligning What EEG Can See: Structural Representations for Brain-Vision Matching*. [[arXiv]](https://arxiv.org/abs/2603.07077) *(已核实存在)*
* **NICE (对比基线)**: Song et al. (2024). *Decoding Natural Images from EEG for Object Recognition*. ICLR 2024. [[arXiv]](https://arxiv.org/abs/2308.13234)

### 2. 启发性架构 (Architectural Inspirations)
* **NECOMIMI (双路时空卷积)**: Chen et al. (2024). [[GitHub]](https://github.com/ChiShengChen/EEG_gen_img_NECOMIMI)
* **CognitionCapturer (多模态联合)**: Zhang et al. [[GitHub]](https://github.com/XiaoZhangYES/CognitionCapturer)

### 3. 工程主线依赖 (Core Repositories)
* **SDXL-Turbo**: [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo) (替换原 SD1.5，极速推理)
* **IP-Adapter**: [TencentARC/IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) (语义硬约束)
* **Diffusers**: [huggingface/diffusers](https://github.com/huggingface/diffusers) (img2img 基础库)
* **MindEyeV2**: [MedARC-AI/MindEyeV2](https://github.com/MedARC-AI/MindEyeV2) (统一的 PixCorr/SSIM 评估脚本)
