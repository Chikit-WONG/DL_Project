# 终极大一统方案 V5：HMEG-Dual-Prior 终局决议 (Final Blueprint)

**项目目标**：在 THINGS-EEG 数据集上实现零样本检索（Top-1 > 20%）与结构化重建（SSIM > 0.310）的双重 SOTA。
**核心理念**：用最强的时空特征压榨信号，用最稳的“软约束”引导生成。

---

## 1. 核心架构设计 (The Final Architecture)

### 1.1 编码器：NERV-style 增强型 Encoder
* **物理位置注入**：根据电极 3D 坐标注入 Position Encoding，赋予模型空间先验。
* **双路卷积 (Dual-Path Conv)**：并行 ST（空间-时间）和 TS（时间-空间）分支，提取互补特征。
* **脑区门控 (Region-aware Gating)**：对 63 通道进行可学习加权，初始权重向枕叶 (Occipital) 和顶叶 (Parietal) 倾斜。
* **Subject Adapter**：针对不同受试者引入轻量级 Token，解决个体脑电差异。

### 1.2 任务双头 (Dual-Head Strategy)
* **语义头 (Semantic Head)**：
    * **主力目标**：CLIP ViT-H/14（确保检索性能的向后兼容）。
    * **辅助目标**：CLIP ViT-B/32 & RN50（提供多视角特征补充）。
* **结构头 (Structural Head)**：
    * **唯一目标**：图像的 VAE Latent（4x64x64）。
    * **损失函数**：Smooth L1 Loss（回归低频物理结构，不加对比损失以防 OOM）。

---

## 2. 生成管线 (The SOTA Pipeline)

放弃高风险的硬约束（T2I-Adapter），采用高容错的“两阶段净化+软约束引导”。

* **Stage 1: 先验提纯 (Prior Diffusion)**
    * 训练一个 ~20M 的 U-Net，将含噪的 EEG 语义特征映射为纯净的 CLIP 图像先验 $Z_{refined}$。
* **Stage 2: 渲染引导 (SDXL-Turbo)**
    * **结构引导 (软约束)**：由 Structural Head 预测的 VAE Latent 解码为模糊底图，作为 **img2img** 的起点（Strength=0.5）。
    * **语义引导 (硬约束)**：将 $Z_{refined}$ 注入单路 **IP-Adapter**。
    * **底座**：SDXL-Turbo（4步推理，极速迭代）。

---

## 3. 训练策略与损失函数 (Training & Loss)

采用**三阶段热身策略**，防止多目标训练初期的梯度混乱。

| 阶段            | 周期 (Epoch) | 激活的 Loss 项                                               | 目的                     |
| :-------------- | :----------- | :----------------------------------------------------------- | :----------------------- |
| **1. 基线热身** | 0 - 20       | $1.0 \times L_{InfoNCE}(H14)$                                | 稳住核心检索精度         |
| **2. 分层对齐** | 21 - 40      | $+ 0.3 \times L_{InfoNCE}(B32/RN50) + 0.3 \times L_{MSE}(VAE)$ | 引入空间结构与多视角语义 |
| **3. 指标精修** | 41+          | $+ 0.1 \times L_{HardNegative} + 0.2 \times L_{Class}$       | 提升 Top-1 的区分度      |

---

## 4. 执行甘特图 (10-Day SOTA Sprint)

* **Day 1-2**: **基础设施升级**。完成 B32/RN50/VAE Latent 的特征预缓存；切换生成底座为 SDXL-Turbo。
* **Day 3-5**: **编码器攻坚**。实装双路卷积与位置编码，跑通阶段 1 & 2 的训练，验证 Top-1 是否突破 18%。
* **Day 6-8**: **先验模型训练**。独立训练 Prior Diffusion，打通从 EEG 到精炼语义的映射。
* **Day 9-10**: **全链路合龙**。集成 img2img + IP-Adapter，微调 Guidance Scale，完成全指标评估（SSIM/CLIP/PixCorr）。

---

## 5. 为什么这是最终方案？(Final Justification)

1.  **关于检索**：保留了 H/14 这个最强基准，同时用双路卷积解决了时间序列的特征损失。
2.  **关于重建**：用 img2img 替代 T2I-Adapter 解决了 EEG 预测不准导致的“图像崩溃”风险。
3.  **关于工程**：分步训练（Warm-up）是应对 6 个不同损失函数同时存在的唯一可行方案。