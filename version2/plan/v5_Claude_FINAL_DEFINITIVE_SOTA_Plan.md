# EEG-to-Image SOTA 冲击方案【最终定稿】
**DSAA2012 Deep Learning — Project A**
*三方（Claude + ChatGPT + Gemini）五轮讨论后完全统一*
*定稿日期：2026-04-16 · 不再产生新版本*

---

## 零、最终裁定：ChatGPT v4 剩余两个问题

ChatGPT v4 提出的两个最后问题，在此一并定稿：

**问题 1：3路 IP-Adapter 是否永久降为"增强实验"？**
> **是。** 三方一致：3路 IP-Adapter 不进主线，归入 Final-Submit-C 可选实验。
> 理由：depth_proj / edge_proj 在训练初期不稳定，多路控制器会放大噪声。
> 后期如果单路稳定后可以尝试。

**问题 2：L_class 是否进入 Final-Submit-B？**
> **是。** L_class 参数极少（只是 supervised contrastive loss），
> 对检索边界精修有帮助，权重设 0.2，epoch 40 之后加入，风险极低。
> Final-Submit-B = 主线 + L_hard + L_class。

**Gemini v4 声明已 100% 收敛。ChatGPT v4 已接受所有核心决定。三方统一，本文件定稿。**

---

## 一、一句话版本

> **ViT-H/14 主对齐 + B32/RN50/VAE 补充 + 双路卷积Encoder + Prior Diffusion + 单路IP-Adapter + img2img + SDXL-Turbo，采用渐进式多目标热身训练。**

---

## 二、完整架构

### 2.1 EEG Encoder

```
EEG [B, 63, T]
  │
  ├─ 电极位置编码
  │    EEG_CHANNELS.jsonl → 球面坐标 → 正弦编码 [63, D]
  │    在 DualPathConv 前注入
  │
  ├─ Region-aware Gating（63个可学习权重）
  │    初始化：枕叶/顶叶电极（index 46-62）权重=2.0
  │    让模型自学区域重要性，近乎零参数代价
  │
  ├─ DualPathConv（NERV风格双路并行）
  │    路径A：先 Spatial（跨电极）→ 再 Temporal（时间轴）
  │    路径B：先 Temporal（时间轴）→ 再 Spatial（跨电极）
  │    两路 concat → 后续层
  │
  ├─ Channel-wise Attention（电极空间关系建模）
  │
  ├─ Post-Transformer TSConv（保留token维度，不早期GAP）
  │
  ├─ Subject Token（per-subject个体适配）
  │
  ├─ Semantic Head → Z_sem [B, 1024]
  │    对齐目标：CLIP ViT-H/14（主力）
  │              CLIP ViT-B/32（补充）
  │              CLIP RN50（补充）
  │
  └─ Structural Head → Z_str [B, 4·64·64]
       对齐目标：VAE Latent（低层轮廓/布局/色块）
```

### 2.2 生成管道

```
Z_sem ────────────────────────────────────────────────┐
  │                                                   │
  ▼                                                   │
Prior Diffusion U-Net（~20M参数）                      │
  输入：Z_sem + 噪声化Z_I + 时间步t                    │
  训练：CFG（10%概率null vector替换Z_sem）              │
  输出：Z_I_refined（精炼后的CLIP ViT-H/14向量）        │
  │                                                   │
  ▼                                                   │
单路 IP-Adapter（语义控制）                             │
  ← Z_I_refined                                       │
  │                                                   │
  └─────────── SDXL-Turbo（4步推理）──────────────────┘
                      ↑
              img2img 结构底图
                      ↑
               VAE 解码器（blurry image）
                      ↑
                Z_str（Structural Head输出）
                strength = 0.5

  最终输出：重建图像
```

---

## 三、损失函数与热身策略

```python
# ════════════════════════════════════════════
# 热身期 epoch 0-19：只用主对齐，防梯度冲突
# ════════════════════════════════════════════
L_total = InfoNCE(eeg_sem, clip_h14) \
        + 0.5 * MSE(eeg_sem, clip_h14)

# ════════════════════════════════════════════
# 主训练期 epoch 20-39：加多视觉目标 + 结构对齐
# ════════════════════════════════════════════
L_total += 0.3 * InfoNCE(eeg_sem, clip_b32)
L_total += 0.3 * InfoNCE(eeg_sem, clip_rn50)
L_total += 0.3 * SmoothL1(eeg_str, vae_latent.flatten(1))

# ════════════════════════════════════════════
# 精修期 epoch 40+：加难负样本 + 类别对比
# ════════════════════════════════════════════
L_total += 0.1 * hard_negative_infonce(eeg_sem, clip_h14)
L_total += 0.2 * supervised_contrastive(eeg_sem, text_labels)

# ════════════════════════════════════════════
# 可选（最后阶段，随时可关闭）
# ════════════════════════════════════════════
# L_total += 0.05 * InfoNCE(eeg_sem, vlm_text_embed)
```

**训练不稳定时的降级策略：**

```
如果 epoch 20+ 训练崩溃：
  → 去掉 L_sem_aux（B32+RN50），只保留 H/14 + VAE
如果仍不稳：
  → 去掉 VAE SmoothL1，只保留 H/14 主对齐
  → 重新热身后再逐步加回
```

---

## 四、执行计划与验收标准

### Milestone 0：基线固化（Day 1 上午）

| 工作 | 验收 |
|---|---|
| 确认 80-trial averaging（`eeg.mean(axis=1)`） | ✓ |
| 固定随机种子，统一评估脚本 | ✓ |
| 记录 Arch A Joint 所有基准指标 | 基准表完整 |

---

### Milestone 1：生成底座升级（Day 1 下午 - Day 2）

| 工作 | 验收标准 |
|---|---|
| SD v1.5 → SDXL-Turbo（`steps=4`） | 重建 CLIP ≥ 0.750 |
| 单路 IP-Adapter 接通 | 重建 SSIM ≥ 0.290 |
| 预计算缓存：H/14, B32, RN50, VAE Latent | 缓存文件完整 |

---

### Milestone 2：Encoder 升级（Day 3-5）

| 工作 | 验收标准 |
|---|---|
| DualPathConv + 位置编码 + Region Gating | — |
| Semantic Head + Structural Head | — |
| 热身训练：前20 epoch 只用 `L_sem_main` | — |
| epoch 20+ 加入 `L_sem_aux + L_struct` | — |
| 只评估检索，不生成图像 | **Top-1 ≥ 18%** |
| — | **Top-5 ≥ 42%** |
| VAE Latent 可视化 | 结构合理（人工确认） |

---

### Milestone 3：Prior Diffusion（Day 5-6）

| 工作 | 验收标准 |
|---|---|
| 实现 PriorUNet（6层，~20M参数） | — |
| CFG 训练（10% null vector） | cosine_sim > 0.85 |
| 串联 Z_sem → Prior → Z_I_refined | CLIP 比 M1 再 +0.03 |

---

### Milestone 4：完整管道评估（Day 7）

| 工作 | 验收标准 |
|---|---|
| 串联：Encoder→Prior→IP-Adapter→img2img→SDXL | — |
| 10个 seed 批量评估 | **Top-1 ≥ 20%** |
| — | **Top-5 ≥ 48%** |
| — | **SSIM ≥ 0.310** |
| — | **CLIP ≥ 0.760** |

---

### Milestone 5：可选增强（Day 8-10）

| 工作 | 验收标准 |
|---|---|
| 加 `L_hard + L_class`（epoch 40+） | Top-1 ≥ 23% |
| Subject Adapter 精调 | — |
| 可选：`L_text`（λ=0.05） | 边际效果验证 |
| 可选：3路 IP-Adapter（仅限结构稳定后） | SSIM ≥ 0.330 |

---

### 甘特图

```
Day:  1am 1pm  2    3    4    5    6    7    8    9   10
M0   [==]
M1       [=========]
M2                 [==============]
M3                          [=========]
M4                                    [====]
M5                                         [============]
```

---

## 五、三档提交模型

### Final-Submit-A（最稳，7天可完成）
```
DualPathConv + 位置编码 + Region Gating
Semantic Head（H/14主 + B32+RN50补）
Structural Head（VAE）
Prior Diffusion
单路 IP-Adapter + img2img + SDXL-Turbo
渐进热身训练
```

### Final-Submit-B（推荐，8-9天可完成）
```
= Final-Submit-A
+ L_hard（难负样本加权，epoch 40+）
+ L_class（类别对比损失，epoch 40+）
```

### Final-Submit-C（探索，10天+）
```
= Final-Submit-B
+ 3路 IP-Adapter（仅限结构预测稳定后）
+ L_text（λ=0.05，边际实验）
```

**建议提交：Final-Submit-B（时间允许则尝试C的部分实验）**

---

## 六、消融实验设计

| 实验 | 改动 | 对比基准 | 主要观测指标 |
|---|---|---|---|
| Exp-0 | Arch A Joint 基准 | — | Top-1, SSIM, CLIP |
| Exp-1 | + SDXL-Turbo | Exp-0 | SSIM, CLIP |
| Exp-2 | + 双路卷积 + 位置编码 + Region Gating | Exp-1 | Top-1, Top-5 |
| Exp-3 | + 多视觉目标（H/14+B32+RN50+VAE） | Exp-2 | Top-1, Top-5 |
| Exp-4 | + Prior Diffusion | Exp-3 | SSIM, CLIP, PixCorr |
| Exp-5 | + img2img 结构底图 | Exp-4 | PixCorr, SSIM |
| Exp-6 | + L_hard + L_class（精修） | Exp-5 | Top-1 边际 |
| Exp-7 | + 3路 IP-Adapter（可选） | Exp-6 | SSIM, 视觉质量 |

---

## 七、明确排除项（永久关闭，不再讨论）

| 方案 | 排除理由 |
|---|---|
| SD v1.5 作为底座 | 2022年模型，落后两代 |
| T2I-Adapter 作为主线核心 | 推理时无像素空间输入，噪声传播 |
| Triplet Loss 替换 InfoNCE | InfoNCE 是多负样本升级版，退步 |
| VLM Caption 进主推理链 | ATM 已证明不稳定，语义漂移 |
| 所有 loss 同时大权重训练 | 梯度冲突，必须热身 |

---

## 八、三方贡献归属

| 模块 | 贡献方 |
|---|---|
| 双路卷积（DualPathConv） | NERV/NECOMIMI + Claude |
| 电极位置编码 | Claude（EEG_CHANNELS.jsonl） |
| Region-aware Gating | ChatGPT HMEG-Prior + Gemini（坚持进主线） |
| 保留token维度 / Post-TSConv | ChatGPT HMEG-Prior |
| 多视觉目标对齐（H/14+B32+RN50+VAE） | **ChatGPT HMEG-Prior（最重要新增）** |
| H/14 主力地位坚守 | **Claude（三方争议，Claude 胜）** |
| Prior Diffusion | ATM NeurIPS 2024 |
| 渐进热身训练策略 | ChatGPT v2/v3 |
| SDXL-Turbo | 三方一致 |
| 1路 IP-Adapter + img2img | **ChatGPT v3 + Gemini v3（保守务实）** |
| 3路 IP-Adapter（Phase 2可选） | Claude + CognitionCapturerPro 2026 |
| 类别对比损失 L_class | EEG-CLIP 2025 + Claude |
| 验收标准数字 | Gemini v2（Claude修正） |
| VLM文字辅助（极低权重可选） | Gemini v1 |
| 100% 收敛宣言 | **Gemini v4** |

---

*本文件为 Claude + ChatGPT + Gemini 五轮讨论后的最终定稿版。*
*三方已完全统一。不再产生新讨论版本。*
*下一步：直接按本文件修改 `model.py`、`train.py`、`reconstruct.py`。*
