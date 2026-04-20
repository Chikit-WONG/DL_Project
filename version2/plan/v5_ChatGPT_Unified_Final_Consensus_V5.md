# 三方统一最终定稿版 V5：EEG-to-Image SOTA 执行计划

**用途：** 这是给 ChatGPT / Claude / Gemini 继续核对的“最终定稿候选版”。  
**目标：** 在不再引入新的大分歧前提下，把当前三方已经基本统一的方案固化成一份可执行计划。  
**当前状态：** 基于 `v4_Claude_SOTA_Proposal_V4_Convergence.md` 与 `v4_Gemini.md` 收敛而成。  
**结论：** 本版默认 **ChatGPT 也正式接受 V4 中的三项待表态问题**，因此该版本可以视为当前最接近最终统一意见的执行方案。

---

## 0. ChatGPT 对 V4 三个待表态问题的正式答复

### 0.1 ViT-H/14 的角色定位

**接受。**

- `CLIP ViT-H/14` 作为 **Semantic Head 的主力对齐目标**，不可删除。
- `CLIP ViT-B/32` 与 `CLIP RN50` 作为 **补充目标**，用于提供额外视觉统计与鲁棒性。
- 统一意见不是“删掉 H/14”，而是把它从“唯一目标”升级为“主目标 + 辅助目标”。

### 0.2 Region-aware Gating 是否进主线
**接受。**

- 这个模块只有 63 个可学习参数，几乎零成本。
- 它不会显著增加训练不稳定性。
- 它与“视觉相关信息在 occipital / parietal 更关键”的经验先验一致，因此适合直接进入主线。

### 0.3 IP-Adapter 路数
**接受分阶段方案。**

- **第一阶段主线：** `1 路 IP-Adapter（语义） + 1 路 img2img（VAE 结构底图）`
- **第二阶段可选增强：** 如果 `depth_proj / edge_proj` 质量通过验证，再升级到 `3 路 IP-Adapter`

> **因此，V4 中剩余的三项分歧，现在全部视为已统一。**

---

## 1. 一句话最终方案

> **DualPathConv + 电极位置编码 + Region-aware Gating + Subject Token/Adapter + Semantic/Structural 双头 + ViT-H/14 主对齐 + B32/RN50/VAE 辅助对齐 + Prior Diffusion + 单路 IP-Adapter + img2img + SDXL-Turbo + 渐进式多目标热身训练。**

---

## 2. 最终主线架构

## 2.1 EEG Encoder

```text
EEG [B, 63, T]
  ↓ 电极位置编码（EEG_CHANNELS.jsonl → 球面坐标 → 正弦编码）
  ↓ Region-aware Gating（63个可学习权重，枕叶/顶叶初始化更高）
  ↓ DualPathConv（NERV风格）
      路径A：先 Spatial → 再 Temporal
      路径B：先 Temporal → 再 Spatial
  ↓ Channel-wise Attention
  ↓ Post-Transformer TSConv（保留 token 维度，不早期 GAP）
  ↓ Subject Token / Subject Adapter
  ├── Semantic Head
  └── Structural Head
```

---

## 2.2 对齐目标

### Semantic Head
- 主对齐：`CLIP ViT-H/14`
- 补充对齐：`CLIP ViT-B/32`
- 补充对齐：`CLIP RN50`

### Structural Head
- 对齐：`VAE Latent`

---

## 3. 最终损失函数与热身策略

### 3.1 损失函数定义

```python
# 主语义对齐（主力）
L_sem_main = InfoNCE(eeg_sem, clip_h14) + 0.5 * MSE(eeg_sem, clip_h14)

# 辅助语义对齐（补充）
L_sem_aux  = 0.3 * InfoNCE(eeg_sem, clip_b32)            + 0.3 * InfoNCE(eeg_sem, clip_rn50)

# 结构对齐
L_struct   = 0.3 * SmoothL1(eeg_str, vae_latent.flatten(1))

# 可选精修项（不是最早阶段必须）
L_class    = 0.2 * supervised_contrastive(eeg_sem, text_labels)
L_hard     = 0.1 * hard_negative_weighted_infonce(eeg_sem, clip_h14)

# 可选极低权重文本辅助（默认关闭）
L_text     = 0.05 * InfoNCE(eeg_sem, vlm_text_embed)
```

### 3.2 热身训练策略

```python
# epoch 0-19
L_total = L_sem_main

# epoch 20-39
L_total = L_sem_main + L_sem_aux + L_struct

# epoch 40-59
L_total = L_sem_main + L_sem_aux + L_struct + L_class + L_hard

# epoch 60+
# 默认不加 L_text
# 若主线稳定且有时间，再单独做边际实验
```

### 3.3 训练原则
1. **绝不一开始就把所有 loss 同时大权重打开。**
2. **先保住主线稳定，再加精修项。**
3. **如果训练不稳，优先保留 `L_sem_main + L_struct`，再逐步恢复其他项。**

---

## 4. 两阶段生成管道（最终统一版）

## 4.1 Stage I：Prior Diffusion
目标：把 noisy 的 EEG semantic embedding 精炼成更接近真实视觉分布的 `Refined CLIP Prior`。

```python
Input:  Z_semantic
Model:  PriorUNet (~20M parameters)
Train:  Classifier-Free Guidance (10% null vector)
Output: Z_I_refined
```

---

## 4.2 Stage II：单路 IP-Adapter + img2img + SDXL-Turbo

### 当前主线（正式定稿）
- **语义控制：** `Z_I_refined -> 单路 IP-Adapter`
- **结构控制：** `Z_structural -> VAE decoder -> blurry image -> img2img`
- **生成底座：** `SDXL-Turbo`
- **推理步数：** `4`
- **img2img strength：** `0.5`

```python
def reconstruct(eeg_signal):
    z_sem, z_str = encoder(eeg_signal)
    z_clip = prior_unet.sample(z_sem, guidance_scale=3.0)
    blurry = vae_decoder(z_str.reshape(B, 4, 64, 64))

    image = sdxl_turbo(
        ip_adapter_embeds=[z_clip],   # 单路语义
        image=blurry,                 # 结构底图
        strength=0.5,
        num_inference_steps=4,
    )
    return image
```

---

## 4.3 第二阶段可选增强（不属于第一轮主线）
只有满足下面条件时，才升级为 3 路 IP-Adapter：

- `depth_proj` 的输出可验证地带来收益
- `edge_proj` 的输出可验证地带来收益
- 不会显著降低生成稳定性

```text
第一轮主线：1路 IP-Adapter + img2img
第二轮增强：1路 → 3路 IP-Adapter（可选）
```

---

## 5. 明确的主线 / 可选 / 暂不建议

## 5.1 主线必做
1. 80-trial averaging 固化
2. SDXL-Turbo
3. DualPathConv
4. 电极位置编码
5. Region-aware Gating
6. Subject Token / Adapter
7. Semantic / Structural 双头
8. H/14 主对齐 + B32 / RN50 / VAE 补充对齐
9. Prior Diffusion
10. 单路 IP-Adapter + img2img
11. 渐进式热身训练

## 5.2 可选增强
1. `L_class`
2. `L_hard`
3. 3 路 IP-Adapter
4. `L_text`（极低权重）
5. 更细的 subject-specific adapter 调优

## 5.3 暂不建议
1. T2I-Adapter 作为当前主线核心
2. ControlNet 作为当前主线核心
3. Triplet Loss 替代 InfoNCE
4. VLM Caption 进入主推理链
5. 第一轮就把所有控制器和所有损失全部打开

---

## 6. 统一执行路线图（7~10 天）

## Day 1
- 固化 baseline
- 确认 80-trial averaging
- 固定随机种子与评估脚本

## Day 2
- 切换到 SDXL-Turbo
- 打通单路 IP-Adapter + SDXL-Turbo 推理
- 记录新的 reconstruction baseline

## Day 3-4
- 实装 DualPathConv
- 加入电极位置编码
- 加入 Region-aware Gating
- 加入 Subject Token / Adapter
- 加双 Head
- 预缓存 H/14、B32、RN50、VAE latent

## Day 5
- 只训练 encoder 侧
- 前 20 epoch 只跑 `L_sem_main`
- 验证 retrieval 是否达到阶段性目标

## Day 6
- 加入 `L_sem_aux + L_struct`
- 检查 Structural Head 输出的 latent / blurry image 是否合理

## Day 7
- 训练 Prior Diffusion
- 打通完整链路：
  `Encoder -> Prior -> IP-Adapter + img2img -> SDXL-Turbo`
- 跑完整评估

## Day 8-10（可选）
- 加 `L_hard`
- 再尝试 `L_class`
- 仅在主线稳定后，才考虑升级到 3 路 IP-Adapter
- 最后才考虑 `L_text`

---

## 7. 验收标准（统一修正版）

### Milestone 1（Day 1-2）
- Recon CLIP ≥ `0.750`
- SSIM ≥ `0.290`

### Milestone 2（Day 3-5）
- Top-1 ≥ `18%`
- Top-5 ≥ `45%`
- Structural Head 输出可视化结构合理

### Milestone 3（Day 5-6）
- Prior 后 `cosine_sim(Z_pred, Z_true)` 明显提升
- Recon CLIP 相比 M1 再提升 ≥ `0.03`

### Milestone 4（Day 7）
- Top-1 ≥ `20%`
- Top-5 ≥ `48%`
- SSIM ≥ `0.310`
- Recon CLIP ≥ `0.760`

### Milestone 5（Day 8-10，可选增强）
- Top-1 ≥ `23%`
- SSIM ≥ `0.330`

> 说明：  
> `全面超越 ATM` 仍然是最终长期目标，**不是 Day 7 的强制要求**。

---

## 8. 消融实验设计（建议）

| 实验编号 | 改动 | 对比基准 |
|---|---|---|
| Exp-0 | 当前 Arch A Joint 基线 | — |
| Exp-1 | + SDXL-Turbo | Exp-0 |
| Exp-2 | + DualPathConv + 位置编码 + Region Gating + Subject Token | Exp-1 |
| Exp-3 | + H/14 主 + B32/RN50/VAE 补充对齐 | Exp-2 |
| Exp-4 | + Prior Diffusion | Exp-3 |
| Exp-5 | + img2img 结构底图 | Exp-4 |
| Exp-6 | + L_class + L_hard | Exp-5 |
| Exp-7（可选） | + 3 路 IP-Adapter | Exp-6 |

**推荐最终提交模型：**
- 时间紧：`Exp-5`
- 时间较充裕：`Exp-6`

---

## 9. 本版与上一版最大的收敛结果

1. **ViT-H/14 主力地位正式固定。**
2. **Region-aware Gating 正式进入主线。**
3. **3 路 IP-Adapter 正式降级为第二阶段增强，而不是第一轮刚需。**
4. **热身训练策略正式固定。**
5. **T2I-Adapter / Triplet 替代 / Caption 主推理链正式排除。**

---

## 10. 最终定稿候选语句

> **最终统一意见候选版：以 ViT-H/14 为主力语义对齐目标，辅以 B32、RN50 与 VAE latent，采用 DualPathConv + Region-aware Gating + Subject Token 的双头 EEG 编码器，并用 Prior Diffusion + 单路 IP-Adapter + img2img + SDXL-Turbo 完成两阶段重建，同时使用渐进式多目标热身训练。**

如果 Claude / Gemini / ChatGPT 都接受这句话，就不再需要产生新的大版本，只剩实验顺序和超参数微调问题。
