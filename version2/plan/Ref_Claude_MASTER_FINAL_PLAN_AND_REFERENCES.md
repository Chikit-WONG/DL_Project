# DSAA2012 Project A：SOTA 冲击执行方案【终极定稿】
**三方（Claude + ChatGPT + Gemini）三轮核对后完全统一**
**定稿日期：2026-04-16 · 此为唯一权威版本，不再产生新版本**

---

## 一、一句话方案

> **DualPathConv + 电极位置编码 + Region-aware Gating + Subject Token + Semantic/Structural 双头 + ViT-H/14 主对齐 + B32/RN50/VAE 辅助对齐 + Prior Diffusion + 单路 IP-Adapter + img2img + SDXL-Turbo + 渐进式多目标热身训练**

---

## 二、完整架构

### 2.1 EEG Encoder

```
EEG [B, 63, T]
  ↓ 电极位置编码（EEG_CHANNELS.jsonl → 球面坐标 → 正弦编码）
  ↓ Region-aware Gating（63个可学习权重，枕叶/顶叶初始权重=2.0）
  ↓ DualPathConv（NERV风格）
      路径A：先 Spatial（跨电极） → 再 Temporal（时间轴）
      路径B：先 Temporal（时间轴） → 再 Spatial（跨电极）
  ↓ Channel-wise Attention
  ↓ Post-Transformer TSConv（保留 token 维度，不早期 GAP）
  ↓ Subject Token / Subject Adapter
  ├── Semantic Head  → 对齐 ViT-H/14（主）+ B32 + RN50（补）
  └── Structural Head → 对齐 VAE Latent（Smooth L1）
```

### 2.2 损失函数与热身策略

```python
# ═══════════════════════════════════════
# epoch 0-19：只用主对齐（防梯度冲突）
# ═══════════════════════════════════════
L_total = InfoNCE(eeg_sem, clip_h14) + 0.5*MSE(eeg_sem, clip_h14)

# ═══════════════════════════════════════
# epoch 20-39：加多视觉目标 + 结构对齐
# ═══════════════════════════════════════
L_total += 0.3*InfoNCE(eeg_sem, clip_b32)
L_total += 0.3*InfoNCE(eeg_sem, clip_rn50)
L_total += 0.3*SmoothL1(eeg_str, vae_latent.flatten(1))

# ═══════════════════════════════════════
# epoch 40+：加难负样本 + 类别对比（精修）
# ═══════════════════════════════════════
L_total += 0.1*hard_negative_infonce(eeg_sem, clip_h14)
L_total += 0.2*supervised_contrastive(eeg_sem, text_labels)

# epoch 60+：L_text（λ=0.05，默认关闭，最后可选）
```

**训练崩溃降级策略：** 去掉 L_sem_aux → 去掉 L_struct → 只保留 L_sem_main

### 2.3 生成管道

```
EEG → Encoder → Z_sem [1024], Z_str [4·64·64]
                    │                  │
                    ▼                  ▼
          Prior Diffusion U-Net    VAE Decoder
          (~20M, CFG, 10% null)   (模糊底图)
                    │                  │
                    ▼                  ▼
            Z_I_refined            blurry_image
                    │                  │
                    ▼                  ▼
            IP-Adapter (1路)      img2img (strength=0.5)
                    └────── SDXL-Turbo (4步) ──────┘
                                   ↓
                              重建图像
```

---

## 三、执行计划

### Milestone 0：基线固化（Day 1 上午）
- 确认 80-trial averaging：`eeg.mean(axis=1)`
- 固定随机种子，统一评估脚本
- 记录 Arch A Joint 所有基准指标

**验收：** 基准表完整，脚本可复现

---

### Milestone 1：底座升级（Day 1 下午 – Day 2）
- SD v1.5 → SDXL-Turbo（`num_inference_steps=4`）
- 预计算缓存：H/14、B32、RN50、VAE Latent

**验收：** Recon CLIP ≥ 0.750 · SSIM ≥ 0.290

---

### Milestone 2：Encoder 升级（Day 3–5）
- DualPathConv + 位置编码 + Region Gating + Subject Token + 双Head
- 热身训练：前 20 epoch 只用 L_sem_main，第 21 epoch 起加 L_sem_aux + L_struct
- 只做检索评估，不生成图像

**验收：** Top-1 ≥ 18% · Top-5 ≥ 45% · VAE Latent 可视化结构合理

---

### Milestone 3：Prior Diffusion（Day 5–6）
- 实现 PriorUNet（6层，~20M 参数）
- CFG 训练（10% null vector）
- 串联：Z_sem → Prior → Z_I_refined

**验收：** `cosine_sim(Z_pred, Z_true)` > 0.85 · Recon CLIP 比 M1 再 +0.03

---

### Milestone 4：完整管道（Day 7）
- 串联：Encoder → Prior → IP-Adapter + img2img → SDXL-Turbo
- 10个 seed 批量评估

**验收：** Top-1 ≥ 20% · Top-5 ≥ 48% · SSIM ≥ 0.310 · CLIP ≥ 0.760

---

### Milestone 5：可选增强（Day 8–10）
- 加 L_hard + L_class（epoch 40+）
- 若结构预测稳定：升级为 3 路 IP-Adapter
- 可选：L_text（λ=0.05）

**验收：** Top-1 ≥ 23% · SSIM ≥ 0.330

---

### 甘特图

```
Day:   1am 1pm  2    3    4    5    6    7    8    9   10
M0    [==]
M1         [========]
M2                  [==============]
M3                           [=========]
M4                                     [====]
M5                                          [============]
```

---

## 四、消融实验

| 实验 | 改动 | 对比 |
|---|---|---|
| Exp-0 | Arch A Joint 基准 | — |
| Exp-1 | +SDXL-Turbo | Exp-0 |
| Exp-2 | +DualPathConv + 位置编码 + Region Gating | Exp-1 |
| Exp-3 | +多视觉目标（H/14+B32+RN50+VAE） | Exp-2 |
| Exp-4 | +Prior Diffusion | Exp-3 |
| Exp-5 | +img2img 结构底图 | Exp-4 |
| Exp-6 | +L_hard + L_class | Exp-5 |
| Exp-7（可选） | +3路 IP-Adapter | Exp-6 |

**推荐提交：** 时间紧选 Exp-5，时间充裕选 Exp-6

---

## 五、明确排除项

| 方案 | 原因 |
|---|---|
| SD v1.5 底座 | 2022年模型，落后两代 |
| T2I-Adapter 作为主线核心 | 推理时无像素空间输入，噪声传播 |
| Triplet Loss 替换 InfoNCE | InfoNCE 是多负样本升级版 |
| VLM Caption 进主推理链 | ATM 已验证不稳定 |
| 所有 loss 同时大权重训练 | 梯度冲突，必须热身 |
| EVA（2026）作为引用来源 | **ICLR 2026 撤稿论文，不可引用** |

---

## 六、参考论文与 GitHub 仓库

### A. EEG 视觉解码主线论文

**1. ATM ✅** — 本方案最直接来源（Prior Diffusion + Low-level 管道）
- 📄 arXiv: https://arxiv.org/abs/2403.07721
- 📄 NeurIPS PDF: https://proceedings.neurips.cc/paper_files/paper/2024/file/ba5f1233efa77787ff9ec015877dbd1f-Paper-Conference.pdf
- 📄 OpenReview: https://openreview.net/forum?id=RxkcroC8qP
- 💻 GitHub: https://github.com/dongyangli-del/EEG_Image_decode

**2. NICE ✅** — 对比基线，TSConv + 对比学习参考
- 📄 arXiv: https://arxiv.org/abs/2308.13234
- 📄 OpenReview: https://openreview.net/forum?id=dhLIno8FmH
- 💻 GitHub: https://github.com/eeyhsong/NICE-EEG

**3. 分层视觉嵌入 ✅** — 多视觉目标对齐（H/14+B32+RN50+VAE）的直接依据
- 📄 arXiv: https://arxiv.org/abs/2602.07495
- 📄 HTML: https://arxiv.org/html/2602.07495v1

**4. 结构表征对齐 ✅** — Structural Head 独立解耦的理论支撑
- 📄 arXiv: https://arxiv.org/abs/2603.07077
- 📄 HTML: https://arxiv.org/html/2603.07077v1

**5. NECOMIMI / NERV ✅** — DualPathConv 双路卷积设计来源
- 📄 arXiv: https://arxiv.org/abs/2410.00712
- 📄 JMIR: https://medinform.jmir.org/2025/1/e72027
- 📄 OpenReview: https://openreview.net/forum?id=ZLZs2QG7vz
- 💻 GitHub: https://github.com/ChiShengChen/EEG_gen_img_NECOMIMI

**6. Perceptogram ✅** — 线性 CLIP 对齐可行性分析
- 📄 arXiv: https://arxiv.org/abs/2404.01250
- 💻 GitHub: https://github.com/desa-lab/Perceptogram
- 📄 OpenReview: https://openreview.net/forum?id=IZOeRDS6zU

**7. CognitionCapturer ✅** — 多模态先验 + 扩散管道参考
- 📄 arXiv: https://arxiv.org/abs/2412.10489
- 💻 GitHub: https://github.com/XiaoZhangYES/CognitionCapturer

**8. CognitionCapturerPro ⚠️** — 3路 IP-Adapter + 不确定性加权参考
- 📄 arXiv: https://arxiv.org/abs/2603.12722
- 💻 GitHub: https://github.com/XiaoZhangYES/CognitionCapturerPro （需手动确认是否已公开）

**9. EEG-CLIP ⚠️** — 类别对比损失 L_class 来源
- 📄 ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/S0893608025010470
- 💻 GitHub: 三方均未找到官方仓库

**10. DreamDiffusion ✅** — EEG 预训练编码器思路参考
- 📄 arXiv: https://arxiv.org/abs/2306.16934
- 💻 GitHub: https://github.com/bbaaii/DreamDiffusion

**11. Brain Decoding (Benchetrit et al.) ✅** — MEG 对比基线
- 📄 arXiv: https://arxiv.org/abs/2310.19812
- 📄 OpenReview: https://openreview.net/forum?id=3y1K6buO8c

**12. JMVR ⚠️** — 高保真重建参考
- 📄 arXiv: https://arxiv.org/abs/2603.19667

**13. UMind ⚠️** — 统一多任务框架参考
- 📄 arXiv: https://arxiv.org/abs/2509.14772

---

### B. 数据集

**14. THINGS-EEG2 ✅**（本项目数据集）
- 📄 DOI: https://doi.org/10.1016/j.neuroimage.2022.119754
- 🗂️ OSF: https://osf.io/3jk45/
- 💻 GitHub: https://github.com/gifale95/eeg_encoding
- 🌐 官网: https://things-initiative.org/

**15. THINGS-EEG1 ✅**（RSVP 范式说明）
- 📄 DOI: https://doi.org/10.1038/s41597-021-01102-7
- 🗂️ OSF: https://osf.io/hd6zk/

**16. THINGS-data ✅**（数据生态总览）
- 📄 DOI: https://doi.org/10.7554/eLife.82580

---

### C. 生成模型与核心组件

**17. IP-Adapter ✅**（主力语义控制器）
- 📄 arXiv: https://arxiv.org/abs/2308.06721
- 💻 GitHub: https://github.com/tencent-ailab/IP-Adapter
- 🌐 项目主页: https://ip-adapter.github.io

**18. SDXL-Turbo ✅**（生成底座）
- 📄 arXiv: https://arxiv.org/abs/2311.17042
- 💻 GitHub: https://github.com/Stability-AI/generative-models
- 🤗 HuggingFace: https://huggingface.co/stabilityai/sdxl-turbo
- 📚 Diffusers 文档: https://huggingface.co/docs/diffusers/using-diffusers/sdxl_turbo

**19. T2I-Adapter ✅**（排除出主线，消融实验备用）
- 📄 arXiv: https://arxiv.org/abs/2302.08453
- 💻 GitHub: https://github.com/TencentARC/T2I-Adapter

**20. DALL-E 2 / unCLIP ✅**（Prior Diffusion 设计思想来源）
- 📄 arXiv: https://arxiv.org/abs/2204.06125
- 💻 非官方实现: https://github.com/lucidrains/DALLE2-pytorch

**21. SDEdit / img2img ✅**（低层结构底图引导）
- 📄 arXiv: https://arxiv.org/abs/2108.01073

---

### D. 预训练视觉模型与工具

**22. CLIP ✅**（H/14 + B/32 + RN50）
- 📄 arXiv: https://arxiv.org/abs/2103.00020
- 💻 OpenAI CLIP（ViT-B/32、RN50）: https://github.com/openai/CLIP
- 💻 OpenCLIP（ViT-H/14）: https://github.com/mlfoundations/open_clip

**23. Diffusers ✅**
- 💻 GitHub: https://github.com/huggingface/diffusers
- 📚 文档: https://huggingface.co/docs/diffusers

**24. MindEye2 ✅**（重建指标脚本来源）
- 📄 arXiv: https://arxiv.org/abs/2305.18274
- 💻 GitHub: https://github.com/MedARC-AI/MindEyeV2
- 🌐 项目主页: https://medarc-ai.github.io/mindeye2/

**25. MNE-Python ✅**（EEG 数据预处理）
- 💻 GitHub: https://github.com/mne-tools/mne-python
- 📚 文档: https://mne.tools/stable/index.html

---

### E. 可选阶段工具

**26. Depth Anything V2 ✅**（深度图提取，3路 IP-Adapter 备用）
- 💻 GitHub: https://github.com/DepthAnything/Depth-Anything-V2

**27. Qwen2.5-VL ✅**（VLM Caption，L_text 可选实验用）
- 🌐 博客: https://qwenlm.github.io/blog/qwen2.5-vl/
- 💻 GitHub: https://github.com/QwenLM/Qwen2.5-VL

---

## 七、三方核对结论

| 论文/资源 | 最终状态 | 说明 |
|---|---|---|
| ATM | ✅ 定稿 | 三方一致 |
| NICE | ✅ 定稿 | 三方一致 |
| arXiv:2602.07495 | ✅ 真实存在 | ChatGPT+Gemini 确认 |
| arXiv:2603.07077 | ✅ 真实存在 | ChatGPT 第二轮确认 |
| NECOMIMI GitHub | ✅ 已补全 | ChatGPT 找到 |
| CognitionCapturer GitHub | ✅ 已补全 | ChatGPT 找到 |
| CognitionCapturerPro GitHub | ⚠️ 填入待确认 | 论文声明存在 |
| EEG-CLIP GitHub | ⚠️ 无官方仓库 | 三方均未找到 |
| EVA（OpenReview） | ⛔ 永久排除 | ICLR 2026 撤稿 |
| Depth Anything V2 | ✅ 已加入 | Gemini 补充 |
| Qwen2.5-VL | ✅ 已加入 | Gemini 补充 |
| MindEye2 项目主页 | ✅ 已完善 | ChatGPT 补充 |

---

*本文件为 Claude + ChatGPT + Gemini 三方三轮核对后的最终唯一权威版本。*
*共收录 27 条资源，22 条已完全确认，3 条暂无代码，2 条需手动验证。*
*下一步：直接按本文件修改 model.py、train.py、reconstruct.py。*
