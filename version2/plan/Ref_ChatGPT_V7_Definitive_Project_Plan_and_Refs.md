# DSAA2012 Project A：最终总方案与参考链接（V7 定稿候选）

**用途：** 这是结合最新两份文件  
- `Ref_v3_Claude_Reference_Links_DEFINITIVE_FINAL.md`
- `Ref_v3_Gemini.md`  

整理出的**最终总方案与参考链接整合稿**。  
**目标：** 作为 ChatGPT / Claude / Gemini 三方在本轮讨论中的**最终定稿候选版**。  
**状态：** 除非后续发现新的链接错误或仓库状态变化，否则这版可以直接作为项目总纲与参考附录使用。

---

## 0. 本轮整合后的结论

### 0.1 哪份更适合做什么？
- **Claude v3 参考链接终稿**  
  更适合作为 **最终参考附录主体**。  
  原因：条目最全、分类最细、状态标记（✅ / ⚠️）最清楚。  

- **Gemini v3 参考文献与依赖清单**  
  更适合作为 **工程白皮书摘要**。  
  原因：表达更短、更适合在 README、答辩 PPT、项目立项说明里快速说明“我们到底借鉴了什么”。

### 0.2 本轮最重要的统一判断
1. **Claude v3 的链接附录已经很接近最终版。**
2. **Gemini v3 里对工程主线的摘要可以直接吸收进总方案正文。**
3. **`arXiv:2602.07495` 和 `arXiv:2603.07077` 两篇 2026 论文都确实存在，应保留。**
4. **EVA 不进入正式主线参考列表。**
5. **CognitionCapturerPro 的 GitHub 可保留为论文中声明的地址，但仍建议标注“需手动确认公开状态”。**
6. **Gemini v3 中 CognitionCapturerPro 的 GitHub 写成了 `CognitionCapturer` 仓库地址，这里需要修正。**

---

## 1. 最终统一工程主线（保持不变）

> **最终统一意见：**  
> `DualPathConv + 电极位置编码 + Region-aware Gating + Subject Token/Adapter + Semantic/Structural 双头 + ViT-H/14 主对齐 + B32/RN50/VAE 辅助对齐 + Prior Diffusion + 单路 IP-Adapter + img2img + SDXL-Turbo + 渐进式多目标热身训练`

---

## 2. 工程白皮书摘要（吸收 Gemini 风格）

## 2.1 编码器
- **位置先验**：根据 `EEG_CHANNELS.jsonl` 注入 63 个电极的 3D 球面坐标位置编码。
- **Dual-Path Conv**：并行 ST（空间→时间）和 TS（时间→空间）两条卷积分支。
- **Region-aware Gating**：对 63 个 EEG 通道做可学习加权，初始化偏向枕叶 / 顶叶。
- **Subject Token / Adapter**：缓解跨被试差异。
- **双头输出**：
  - `Semantic Head`
  - `Structural Head`

## 2.2 分层对齐目标
### Semantic Head
- 主力：`CLIP ViT-H/14`
- 补充：`CLIP ViT-B/32`
- 补充：`CLIP RN50`

### Structural Head
- 目标：`VAE Latent`
- 损失：`Smooth L1`

## 2.3 两阶段生成
### Stage I：Prior Diffusion
- 输入：`EEG semantic embedding`
- 输出：`Refined CLIP prior`
- 训练方式：Classifier-Free Guidance

### Stage II：软约束渲染
- **结构路**：`VAE latent -> blurry image -> img2img`
- **语义路**：`Refined CLIP prior -> 单路 IP-Adapter`
- **生成底座**：`SDXL-Turbo`
- **推理步数**：`4`

---

## 3. 训练策略（最终版）

### 3.1 热身训练
```python
# epoch 0-19
L_total = L_sem_main

# epoch 20-39
L_total = L_sem_main + L_sem_aux + L_struct

# epoch 40-59
L_total = L_sem_main + L_sem_aux + L_struct + L_class + L_hard

# epoch 60+
# L_text 默认关闭，仅最后做边际实验
```

### 3.2 推荐损失写法
```python
L_sem_main = InfoNCE(eeg_sem, clip_h14) + 0.5 * MSE(eeg_sem, clip_h14)

L_sem_aux  = 0.3 * InfoNCE(eeg_sem, clip_b32)            + 0.3 * InfoNCE(eeg_sem, clip_rn50)

L_struct   = 0.3 * SmoothL1(eeg_str, vae_latent.flatten(1))

# 可选精修
L_class    = 0.2 * supervised_contrastive(eeg_sem, text_labels)
L_hard     = 0.1 * hard_negative_weighted_infonce(eeg_sem, clip_h14)

# 最后才尝试
L_text     = 0.05 * InfoNCE(eeg_sem, vlm_text_embed)
```

---

## 4. 敏捷执行计划（7~10 天）

### Day 1
- 固化 baseline
- 确认 80-trial averaging
- 固定随机种子与评估脚本

### Day 2
- 切换到 SDXL-Turbo
- 打通单路 IP-Adapter + SDXL-Turbo 推理
- 记录新的 reconstruction baseline

### Day 3-4
- 实装 DualPathConv
- 加入电极位置编码
- 加入 Region-aware Gating
- 加入 Subject Token / Adapter
- 加双 Head
- 预缓存 H/14、B32、RN50、VAE latent

### Day 5
- 只训练 encoder 侧
- 前 20 epoch 只跑 `L_sem_main`
- 验证 retrieval 是否达到阶段性目标

### Day 6
- 加入 `L_sem_aux + L_struct`
- 检查 Structural Head 输出的 latent / blurry image 是否合理

### Day 7
- 训练 Prior Diffusion
- 打通完整链路：
  `Encoder -> Prior -> IP-Adapter + img2img -> SDXL-Turbo`
- 跑完整评估

### Day 8-10（可选）
- 加 `L_hard`
- 再尝试 `L_class`
- 仅在主线稳定后，才考虑升级到 3 路 IP-Adapter
- 最后才考虑 `L_text`

---

## 5. 验收标准（统一修正版）

### Milestone 1（Day 1-2）
- Recon CLIP ≥ `0.750`
- SSIM ≥ `0.290`

### Milestone 2（Day 3-5）
- Top-1 ≥ `18%`
- Top-5 ≥ `45%`

### Milestone 3（Day 5-6）
- Prior 后特征相似度显著提升
- Recon CLIP 相比 M1 再提升 ≥ `0.03`

### Milestone 4（Day 7）
- Top-1 ≥ `20%`
- Top-5 ≥ `48%`
- SSIM ≥ `0.310`
- Recon CLIP ≥ `0.760`

### Milestone 5（Day 8-10，可选增强）
- Top-1 ≥ `23%`
- SSIM ≥ `0.330`

---

## 6. 最终参考文献与开源链接（主附录）

## 6.1 主线核心论文（直接引用，优先阅读）

### 1. ATM ✅
**Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion**  
Li et al. — NeurIPS 2024

- arXiv: https://arxiv.org/abs/2403.07721
- NeurIPS PDF: https://proceedings.neurips.cc/paper_files/paper/2024/file/ba5f1233efa77787ff9ec015877dbd1f-Paper-Conference.pdf
- OpenReview: https://openreview.net/forum?id=RxkcroC8qP
- GitHub: https://github.com/dongyangli-del/EEG_Image_decode

### 2. NICE ✅
**Decoding Natural Images from EEG for Object Recognition**  
Song et al. — ICLR 2024

- arXiv: https://arxiv.org/abs/2308.13234
- OpenReview: https://openreview.net/forum?id=dhLIno8FmH
- GitHub: https://github.com/eeyhsong/NICE-EEG

### 3. NECOMIMI / NERV ✅
**NECOMIMI: Neural-Cognitive Multimodal EEG-informed Image Generation with Diffusion Models**  
Chen — 2024/2025

- arXiv: https://arxiv.org/abs/2410.00712
- JMIR: https://medinform.jmir.org/2025/1/e72027
- OpenReview: https://openreview.net/forum?id=ZLZs2QG7vz
- GitHub: https://github.com/ChiShengChen/EEG_gen_img_NECOMIMI

### 4. Learning Brain Representation with Hierarchical Visual Embeddings ✅
- arXiv: https://arxiv.org/abs/2602.07495
- HTML: https://arxiv.org/html/2602.07495v1
- GitHub: 暂未公开

### 5. Aligning What EEG Can See: Structural Representations for Brain–Vision Matching ✅
- arXiv: https://arxiv.org/abs/2603.07077
- HTML: https://arxiv.org/html/2603.07077v1
- GitHub: 暂未公开

### 6. CognitionCapturer ✅
- arXiv: https://arxiv.org/abs/2412.10489
- GitHub: https://github.com/XiaoZhangYES/CognitionCapturer

### 7. CognitionCapturerPro ⚠️
- arXiv: https://arxiv.org/abs/2603.12722
- HTML: https://arxiv.org/html/2603.12722v1
- GitHub（论文中声明的地址，建议手动确认公开状态）:  
  https://github.com/XiaoZhangYES/CognitionCapturerPro

### 8. Perceptogram ✅
- arXiv: https://arxiv.org/abs/2404.01250
- GitHub: https://github.com/desa-lab/Perceptogram
- OpenReview: https://openreview.net/forum?id=IZOeRDS6zU

### 9. EEG-CLIP ⚠️
- ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/S0893608025010470
- GitHub: 暂未确认官方仓库

### 10. DreamDiffusion ✅
- arXiv: https://arxiv.org/abs/2306.16934
- GitHub: https://github.com/bbaaii/DreamDiffusion

### 11. JMVR ⚠️
- arXiv: https://arxiv.org/abs/2603.19667
- GitHub: 暂未公开

### 12. UMind ⚠️
- arXiv: https://arxiv.org/abs/2509.14772
- GitHub: 暂未公开

---

## 6.2 数据集
### THINGS-EEG2 ✅
- DOI: https://doi.org/10.1016/j.neuroimage.2022.119754
- OSF: https://osf.io/3jk45/
- 官网: https://things-initiative.org/

### THINGS-EEG1 ✅
- DOI: https://doi.org/10.1038/s41597-021-01102-7
- OSF: https://osf.io/hd6zk/

### THINGS-data ✅
- DOI: https://doi.org/10.7554/eLife.82580

---

## 6.3 生成模型与核心组件
### IP-Adapter ✅
- arXiv: https://arxiv.org/abs/2308.06721
- GitHub: https://github.com/tencent-ailab/IP-Adapter
- 项目页: https://ip-adapter.github.io/

### SDXL-Turbo ✅
- arXiv: https://arxiv.org/abs/2311.17042
- Hugging Face: https://huggingface.co/stabilityai/sdxl-turbo
- GitHub: https://github.com/Stability-AI/generative-models
- Diffusers 文档: https://huggingface.co/docs/diffusers/using-diffusers/sdxl_turbo

### T2I-Adapter ✅
- arXiv: https://arxiv.org/abs/2302.08453
- GitHub: https://github.com/TencentARC/T2I-Adapter

### DALL·E 2 / unCLIP ✅
- arXiv: https://arxiv.org/abs/2204.06125
- 非官方实现: https://github.com/lucidrains/DALLE2-pytorch

### SDEdit / img2img ✅
- arXiv: https://arxiv.org/abs/2108.01073

---

## 6.4 视觉模型与工具链
### CLIP ✅
- arXiv: https://arxiv.org/abs/2103.00020
- OpenAI CLIP: https://github.com/openai/CLIP
- OpenCLIP: https://github.com/mlfoundations/open_clip

### Diffusers ✅
- GitHub: https://github.com/huggingface/diffusers
- 文档: https://huggingface.co/docs/diffusers

### MindEyeV2 ✅
- arXiv: https://arxiv.org/abs/2305.18274
- GitHub: https://github.com/MedARC-AI/MindEyeV2
- 项目页: https://medarc-ai.github.io/mindeye2/

### MNE-Python ✅
- GitHub: https://github.com/mne-tools/mne-python
- 文档: https://mne.tools/stable/index.html

---

## 6.5 可选阶段参考
### Depth Anything V2 ✅
- GitHub: https://github.com/DepthAnything/Depth-Anything-V2

### Qwen2.5-VL ✅
- 官方博客: https://qwenlm.github.io/blog/qwen2.5-vl/
- GitHub: https://github.com/QwenLM/Qwen2.5-VL

### Brain Decoding / Benchetrit et al. ✅
- arXiv: https://arxiv.org/abs/2310.19812
- OpenReview: https://openreview.net/forum?id=3y1K6buO8c

---

## 7. 明确排除 / 不建议进入主线的参考

### EVA ⛔
- OpenReview 页面存在，但为 **withdrawn submission**
- 不建议进入正式主线参考列表

---

## 8. 本版相对前一版的关键修正

1. **继续保留 Claude 的完整参考附录结构**
2. **吸收 Gemini 的工程白皮书写法，形成统一总纲**
3. **修正 `2603.07077` 的状态：它是真实存在的论文**
4. **指出 Gemini v3 中 CognitionCapturerPro 仓库链接写错，需要改成 Pro 仓库地址**
5. **继续保守处理代码状态：**
   - CognitionCapturerPro：论文声明地址，可保留但需手动确认
   - EEG-CLIP / JMVR / UMind：论文可保留，代码先不写成“已确认官方仓库”

---

## 9. 一句话最终结论

> **如果只保留一份“最终总稿”，就应以 Claude v3 的参考附录为主体，以 Gemini v3 的工程白皮书风格为正文表达，再加上本文件中的纠错与状态说明。**

