# EEG-to-Image：统一架构实施计划（v2）

## 背景

DSAA2012 项目 A：基于 THINGS-EEG 的脑信号到图像检索与重建。

**为什么有 v2**：之前的方案把工作拆成两个独立架构（A：共享编码器；B：独立编码器），需要维护两套代码库。我们意识到 **架构 B 在数学上是架构 A 的特殊情形** —— 当其中一个损失权重（alpha 或 beta）取零时，联合训练就退化为单任务训练，等价于一个独立编码器。这一观察使我们可以：写**一套**代码，把架构 B 当作架构 A 的某种配置；并进一步把损失权重做成**可学习参数**，让模型自动平衡两个任务。

**目标**：构建一个能跑通的 EEG-to-Image 系统，在 Top-1/Top-5 检索（25 分）和 SSIM/CLIP 重建（25 分）两个评分项上拿到不错的分数，同时保证方法学清晰（20 分）和代码可复现（10 分）。一个人专注主干评分架构，两个队友后期做微观消融。

**时间线**：今天 2026-04-09。展示 2026-04-28（约 20 天），报告 2026-05-10。

---

## 1. 已确认可用的模型

所有模型均位于 `/hpc2hdd/home/ckwong627/workdir/models/`，所需模型已全部下载完毕：

| 模型 | 路径 | 输出维度 | 用途 |
|------|------|---------|------|
| Stable Diffusion v1.5 | `stable-diffusion-v1-5/` | — | 图像生成主干 |
| IP-Adapter（完整） | `IP-Adapter/models/ip-adapter_sd15.bin` | — | 让 SD 接受图像嵌入作为条件 |
| **IP-Adapter image encoder** | `IP-Adapter/models/image_encoder/` | **1024-d** | **CLIP-ViT-H-14，IP-Adapter 原生 CLIP** |
| LAION CLIP ViT-L/14 | `CLIP-ViT-L-14-laion2B-s32B-b82K/` | 768-d | 备用方案（更小、更快） |
| LAION CLIP ViT-H/14 | `CLIP-ViT-H-14-laion2B-s32B-b79K/` | 1024-d | 与 IP-Adapter 自带的编码器同款 |
| LAION CLIP ViT-B/32 | `CLIP-ViT-B-32-laion2B-s34B-b79K/` | 512-d | 极小的 baseline |

**LAION CLIP 还是 OpenAI CLIP**：使用 LAION CLIP **完全没问题，反而更好**：
1. 助教提供的 `eval_images()` 函数内部确实用了 OpenAI ViT-L/14 来计算 `eval_clip` 指标，但这是把**生成图**和**真实图**做黑盒比较，**不会**接触我们模型的嵌入空间。
2. 检索（Top-1/Top-5）部分由我们自己提供 [N, N] 相似度矩阵，**任何**嵌入空间都可以，只要 EEG 端的对齐目标和图像端的测试嵌入用同一个编码器。
3. 对 IP-Adapter 而言，LAION ViT-H/14 是它**原生匹配**的编码器。换成 OpenAI CLIP 反而要额外加一个学习投影层，丢失信息。

**决定**：整条流水线都使用 IP-Adapter 自带的 image encoder（LAION ViT-H/14，1024-d）。EEG 嵌入直接是 1024-d，不需要任何投影层就能喂给 IP-Adapter。

如果 1024-d 头检索效果不佳，可以回退到 LAION ViT-L/14（768-d），并加一个 768→1024 的投影层供 IP-Adapter 使用。

---

## 2. 统一架构

### 核心数学

```
EEG [B, 63, 250]
      |
  EEG Encoder（CNN + Transformer，约 3M 参数）
      |
  EEG Embedding [B, 1024]（对齐到 LAION ViT-H/14 空间）
      |
      +---> L_retrieval = 对称 InfoNCE       （余弦排名）
      |
      +---> L_reconstruction = CLIP 空间内的 MSE  （绝对位置接近）

总损失 = alpha * L_retrieval + beta * L_reconstruction
```

### 架构 B 如何作为特例出现

| 模式 | alpha | beta | 等价于 |
|------|-------|------|--------|
| 纯检索（架构 B 任务 1） | 1.0 | 0.0 | 独立检索编码器 |
| 纯重建（架构 B 任务 2） | 0.0 | 1.0 | 独立重建编码器 |
| 联合训练（架构 A） | >0 | >0 | 共享编码器 + 双损失 |
| **可学习** | `exp(log_alpha)` | `exp(log_beta)` | 模型自动平衡 |

### 为什么用两个损失？
- **InfoNCE（检索）**：判别性最强的排名信号，但只约束相对顺序，并不约束在 CLIP 空间中的绝对位置。
- **MSE（重建）**：把 EEG 嵌入推到与真实 CLIP 嵌入相同的区域 —— 这是 IP-Adapter 的关键需求，因为它期待"看起来像真实 CLIP 输出"的向量。

这两个损失互补。只用 InfoNCE 时，嵌入可能漂离 CLIP 流形（排名仍然正确，但生成不友好）。只用 MSE 时，模型会塌缩到均值（没有判别信号）。两者结合：既有判别能力，又落在正确区域。

### 为什么用可学习权重？
对 `log_alpha`、`log_beta` 用 `nn.Parameter`（再 `exp()` 保证正定），让模型自动平衡两个任务。训练初期检索梯度更强，后期重建可以接管。这是 Kendall 等人 2018 年同方差不确定性方法的简化版本。

### EEG 编码器架构

```
输入 [B, 63, 250]
  → 空间维：Conv1d(63→128, k=1) + BN + GELU      [B, 128, 250]
            Conv1d(128→128, k=1) + BN + GELU      [B, 128, 250]
  → 时间维：Conv1d(128→192, k=15, s=2) + BN + GELU + Drop  [B, 192, 125]
            Conv1d(192→256, k=15, s=2) + BN + GELU + Drop  [B, 256, 63]
            Conv1d(256→320, k=15, s=2) + BN + GELU + Drop  [B, 320, 32]
  → 转置为 [B, 32, 320]，加可学习位置编码（32 个位置）
  → 3× TransformerEncoderLayer(d=320, heads=8, FFN=640, drop=0.1)  [B, 32, 320]
  → 全局平均池化                                      [B, 320]
  → MLP：Linear(320→640) + GELU + Drop + Linear(640→1024)
  → L2 归一化                                         [B, 1024]
```

预计参数量约 3M。在单 GPU 上能够在 30 分钟的 SLURM 任务内训练完毕。

### 重建管线

```
EEG → EEG Encoder → 1024-d 嵌入（在 LAION ViT-H/14 空间）
              ↓
       IP-Adapter (h94/ip-adapter_sd15.bin)
              ↓
       Stable Diffusion v1.5 UNet（冻结）
              ↓
       512×512 图像 → 缩放为 256×256 用于评估
```

不需要投影层。diffusers 库支持通过 `ip_adapter_image_embeds=` 参数直接传入预先计算好的图像嵌入。

---

## 3. 文件结构

### Python 代码：`DL_Project/codes/`

| 文件 | 用途 |
|------|------|
| `config.py` | 一个 dataclass 容纳所有超参（路径、模型维度、损失权重、训练、增广） |
| `utils.py` | 从 sample code 原样搬来：`set_seed`、`compute_retrieval_metrics`、`summarize_metrics_over_seeds`、`eval_images()` 及其所有子函数，以及 `build_image_id_to_path()` |
| `data.py` | 原样搬来 `load_eeg_dataset()`，`EEGImageDataset`（按 image_id 把 EEG 与缓存的 CLIP 嵌入配对），`EEGAugmentation`（5 种增广） |
| `model.py` | `EEGEncoder`、`UnifiedModel`：双损失 + 固定/可学习 alpha/beta + 可学习温度 |
| `cache_clip_features.py` | 一次性脚本：用 IP-Adapter 的 `image_encoder/` 提取并缓存所有训练+测试图像的 1024-d 特征。保存到 `../clip_cache/` |
| `train.py` | 主训练脚本：两阶段训练、checkpoint 保存、验证。命令行参数：phase/alpha/beta/learnable_weights/resume |
| `reconstruct.py` | IP-Adapter + SD 推理：编码 200 个测试 EEG，每种 seed 生成 200 张图，共 10 个 seed |
| `evaluate.py` | 检索（10 seed）+ 重建（10 seed）评估，打印汇总并保存 JSON |
| `run_all.ipynb` | 给助教的最终 notebook：导入各模块，跑完整流程，展示指标 + 定性结果网格 |

### SLURM 脚本：`DL_Project/slurm_scripts/`

所有脚本统一使用：`partition=debug`、`1 GPU`、`conda env=test`、`module load cuda/12.1`、`--time=00:30:00`。

| 脚本 | 预计耗时 | 运行内容 |
|------|---------|----------|
| `run_cache_clip.sh` | ~15 min | `cache_clip_features.py` |
| `run_train_phase1.sh` | ~15 min | `train.py --phase 1 --alpha 1.0 --beta 0.5 --epochs 50` |
| `run_train_phase2.sh` | ~25 min | `train.py --phase 2 --resume phase1.pt --alpha 0.5 --beta 1.0 --epochs 100` |
| `run_train_learnable.sh` | ~25 min | `train.py --phase 2 --resume phase1.pt --learnable_weights --epochs 100` |
| `run_train_retrieval_only.sh` | ~15 min | `train.py --phase 1 --alpha 1.0 --beta 0.0`（架构 B 检索） |
| `run_train_recon_only.sh` | ~15 min | `train.py --phase 1 --alpha 0.0 --beta 1.0`（架构 B 重建） |
| `run_reconstruct.sh` | ~20 min | `reconstruct.py --seeds 0..9` |
| `run_evaluate.sh` | ~20 min | `evaluate.py` |

### 输出目录（首次运行时自动创建）
```
DL_Project/
  clip_cache/      # 缓存的 LAION ViT-H/14 特征（约 70MB）
  checkpoints/     # 模型权重，每种实验配置一份
  outputs/         # 生成图像（.pt）+ 指标（.json）
  temp/            # SLURM 日志
  plan/            # 计划文档（中英文）
```

---

## 4. 训练策略

### 阶段 1：粗训练
- **数据**：`avg_trials=True`。代码运行时打印实际的 N。
- **增广（完整）**：时间抖动 ±5、通道随机置零 10%、高斯噪声 std=0.02、时间窗口遮盖 20 步、振幅缩放 [0.8, 1.2]。每种以 p=0.5 独立施加。
- **超参**：batch=128, lr=3e-4, AdamW(wd=0.05), 余弦退火, 50 个 epoch
- **损失**：alpha=1.0, beta=0.5, 可学习温度初始值 0.07

### 阶段 2：微调
- 从阶段 1 的最佳 checkpoint 恢复
- batch=64, lr=5e-5, 100 个 epoch
- 损失：alpha=0.5, beta=1.0（向重建侧倾斜）
- 较弱的增广（仅保留时间抖动 + 噪声）

### 阶段 2 变体：可学习权重
- 与阶段 2 相同，加上 `--learnable_weights` 标志
- 用阶段 1 的 alpha/beta 值作为初始化
- 每个 epoch 记录 alpha/beta，便于追踪自动平衡过程

### 架构 B 基线（消融用）
- `--alpha 1.0 --beta 0.0`：纯检索编码器
- `--alpha 0.0 --beta 1.0`：纯重建编码器
- 这就是"架构 B"的两个基线，**用同一份代码**跑出来

---

## 5. 关键实现细节

### 复用已有函数（不要重写）

从 `sample_codes/eeg_project_sample_code.ipynb` 原样搬到 `utils.py`：
- `set_seed(seed)` —— 复现性
- `compute_retrieval_metrics(logits)` —— 从 [N,N] 矩阵算 Top-1/Top-5
- `summarize_metrics_over_seeds(metric_list)` —— 多 seed mean ± std 汇总
- `eval_images(real_images, fake_images, device)` —— 官方评估
- 所有子函数：`pixcorr`、`ssim`、`alexnet`、`inception`、`clip_`、`effnet`、`swav`、`two_way_identification`

从 `sample_codes/eeg_project_sample_code.ipynb` 原样搬到 `data.py`：
- `_selected_channel_indices_from_jsonl()` 和 `load_eeg_dataset()`

### CLIP 特征缓存（`cache_clip_features.py`）

```python
# 加载 IP-Adapter 自带的 image encoder（CLIP-ViT-H-14，1024-d 投影）
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

encoder = CLIPVisionModelWithProjection.from_pretrained(
    "/hpc2hdd/home/ckwong627/workdir/models/IP-Adapter/models/image_encoder"
).to("cuda").eval()
processor = CLIPImageProcessor.from_pretrained(
    "/hpc2hdd/home/ckwong627/workdir/models/IP-Adapter/models/image_encoder"
)

# 对每张图：
inputs = processor(images=PIL_image, return_tensors="pt").to("cuda")
with torch.no_grad():
    image_embeds = encoder(**inputs).image_embeds  # [1, 1024]
image_embeds = F.normalize(image_embeds, dim=-1)
```

保存字典：`{image_id (str): tensor[1024]}` 到 `clip_cache/clip_train_features.pt` 和 `clip_cache/clip_test_features.pt`。

### 损失计算（`model.py`）

```python
def compute_loss(self, eeg_emb, clip_emb):
    alpha, beta = self.get_weights()  # 固定标量或 nn.Parameter 的 exp

    # 检索：对称 InfoNCE
    temp = torch.exp(self.log_temperature)
    logits = (eeg_emb @ clip_emb.T) * temp           # [B, B]
    labels = torch.arange(len(eeg_emb), device=eeg_emb.device)
    L_ret = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    # 重建：CLIP 空间 MSE
    L_rec = F.mse_loss(eeg_emb, clip_emb)

    return alpha * L_ret + beta * L_rec, L_ret.detach(), L_rec.detach()
```

### 重建推理（`reconstruct.py`）

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "/hpc2hdd/home/ckwong627/workdir/models/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")
pipe.load_ip_adapter(
    "/hpc2hdd/home/ckwong627/workdir/models/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter_sd15.bin",
)
pipe.set_ip_adapter_scale(0.7)

# 对每个测试 EEG：
with torch.no_grad():
    eeg_emb = model.eeg_encoder(eeg_tensor)  # [1, 1024]

image = pipe(
    prompt="",
    ip_adapter_image_embeds=[eeg_emb.unsqueeze(0)],
    guidance_scale=7.5,
    num_inference_steps=50,
    generator=torch.Generator("cuda").manual_seed(seed),
).images[0]
```

### 评估（`evaluate.py`）
- **检索**：编码 200 个测试 EEG → [200, 1024]；加载缓存的测试图像嵌入 → [200, 1024]；logits = `eeg @ img.T`；调用 `compute_retrieval_metrics(logits)`。这是确定性的。
- **重建**：每个 seed 加载 200 张生成图，共 10 seed；加载 200 张真实测试图为 [200, 3, 256, 256]；对每个 seed 调用 `eval_images(real, fake, device)`；汇报 mean ± std。

---

## 6. 执行顺序（按优先级）

### P0：必须完成（第 1-12 天）

**第 1-2 天：环境 + CLIP 缓存**
- 验证 `test` conda 环境：`python -c "import torch, transformers, diffusers, datasets, clip"`
- 实现 `config.py`、`utils.py`、`data.py`
- 实现 `cache_clip_features.py`，通过 `sbatch run_cache_clip.sh` 运行

**第 3-5 天：EEG 编码器 + 检索**
- 实现 `model.py`（EEGEncoder + UnifiedModel）
- 实现 `train.py` 阶段 1
- 第一次跑：`--alpha 1.0 --beta 0.0`（纯检索，最简单的调试场景）
- 验证 Top-1 > 1%（10× 随机基线）
- 然后跑：`--alpha 1.0 --beta 0.5`（联合训练）

**第 6-8 天：阶段 2 微调 + 分数优化**
- 用各种 alpha/beta 组合跑阶段 2：{(1,0), (0.8,0.2), (0.5,0.5), (0.2,0.8), (0,1)}
- 跑可学习权重变体
- 目标：Top-1 约 15-25%, Top-5 约 40-55%

**第 9-12 天：重建管线**
- 实现 `reconstruct.py`
- 先用 1 个 seed 生成 200 张图 → 肉眼看质量
- 如果质量差：调 IP-Adapter scale（0.5-1.0）和 guidance scale（5-10）
- 跑通后：生成 10 seed × 200 张图
- 跑 `evaluate.py` 算完整重建指标
- **若 IP-Adapter 完全失败的兜底方案**：在 CLIP 空间最近邻 —— 找到 EEG 嵌入最近的训练图像作为"重建结果"

### P1：分数优化（第 13-16 天）
- 检索的测试时增广（test-time augmentation）
- 微调 IP-Adapter 推理参数
- 试一下其他重建损失：余弦相似度损失、Smooth L1
- 用多 seed 重训练胜出的配置

### P2：微观消融（第 13-16 天，队友接手）
- 队友实现 EEG 编码器变体（纯 CNN、纯 Transformer、EEGNet），在 `codes/encoders/` 加新文件
- 同一份 `train.py` 直接可用 —— 通过 config 切换编码器
- 图像编码器消融：试 LAION ViT-L/14（768-d）变体

### P3：报告与展示（第 17-20 天）
- 生成 8-12 个定性样本（成功 + 失败案例）
- 撰写技术报告
- 完善 `run_all.ipynb` 供助教复现性检查
- 从头验证整条管线可复现

---

## 7. 分数最大化策略

### 检索（25 分）
- 对齐到 LAION ViT-H/14（1024-d）比 ViT-L/14（768-d）有更强的表达能力
- 两阶段训练（大 lr → 小 lr）+ 阶段 1 重增广
- 测试时增广：对同一个 EEG 用小扰动编码 5 次后取平均
- InfoNCE + MSE 联合损失应当优于纯 InfoNCE

### 重建（25 分）
- **SSIM（12.5 分）**：像素级。提高 IP-Adapter scale（0.8-0.9）→ 生成更确定；降低 guidance scale（5-6）→ 减少创造性偏移
- **CLIP Score（12.5 分）**：2-way 识别。生成图必须语义清晰可辨
- 在 512×512 生成后再缩到 256×256

### 方法学（20 分）
- "架构 B 是架构 A 的特例"是一个清晰的方法学贡献 —— 把它做成报告的核心
- 消融表：alpha/beta 扫描（包括纯检索、纯重建、联合、可学习）
- 展示可学习权重确实收敛到有意义的比例（或者没有收敛，并解释原因）

### 代码质量（10 分）
- 关注点分离清晰，每个文件一个职责
- 单一 config dataclass —— 不要让 argparse 失控膨胀
- 复现性：所有 seed 入 config
- README 说明如何复现报告里的所有数字

---

## 8. 风险缓释

| 风险 | 缓释 |
|------|------|
| IP-Adapter 的 `ip_adapter_image_embeds` API 不符合预期 | 读 diffusers 源码；备选：直接 hook 到 UNet 的 cross-attention |
| 生成图像是噪声 | 检查 L2 归一化处理；调 IP-Adapter scale |
| 训练不收敛 | 先跑 `--alpha 1 --beta 0`（纯检索，更简单） |
| 30 分钟 SLURM 上限不够 | 算账：16540 样本 × 50 epoch / batch 128 ≈ 6450 batch × 100ms ≈ 11 分钟，余量充足 |
| `test` conda 环境缺包 | 用 `pip install <pkg>` 安装（**不要** `conda install`，避免环境冲突） |
| 嵌入幅度不匹配（L2 归一化 vs 原始 CLIP） | 两种都试；IP-Adapter image encoder 的输出本身是未归一化的 |
| 小数据集过拟合 | 5 种增广 + dropout 0.1 + weight decay 0.05 + 早停 |

---

## 9. 端到端验证流程

实现完成后，按顺序运行：

1. **CLIP 缓存**：`sbatch slurm_scripts/run_cache_clip.sh`
2. **阶段 1 sanity check**：`train.py --alpha 1 --beta 0 --epochs 5`
3. **阶段 1 完整跑**：`sbatch slurm_scripts/run_train_phase1.sh`
4. **阶段 2 完整跑**：`sbatch slurm_scripts/run_train_phase2.sh`
5. **检索评估**：Top-1 > 5%（远高于随机的 0.5%）
6. **重建**：`sbatch slurm_scripts/run_reconstruct.sh`
7. **重建评估**：报告 SSIM 和 CLIP 分数
8. **最终 notebook**：打开 `codes/run_all.ipynb`
9. **复现性检查**：删除 `checkpoints/` 和 `outputs/`，从 CLIP 缓存重新跑一遍

---

## 10. 实现时需要现场确认的事项

1. **训练集实际大小**：在 `load_eeg_dataset(avg_trials=True)` 后打印 `len(train_ds)`。可能是约 1654 或约 16540 —— 影响 batch 大小和 epoch 数。
2. **IP-Adapter 嵌入格式**：确认 `ip_adapter_image_embeds=` 接收的是 L2 归一化嵌入还是原始投影输出。
3. **conda `test` 环境包检查**：确认 `diffusers` 和 `clip` 是否已安装。
4. **`open_clip_torch`**：如果用 open_clip API 加载 LAION CLIP 可能需要。
5. **`EEG_CHANNELS.jsonl` 中的通道名**：列出了 62 个通道但数据有 63 通道 —— 需要检查。
