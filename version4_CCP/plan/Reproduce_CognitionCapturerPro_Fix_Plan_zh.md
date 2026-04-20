# CognitionCapturerPro 修复计划：缩小与论文得分的差距

## 背景

此前 Codex 已将 CognitionCapturerPro 仓库在课程 EEG 数据集上跑通，但所得指标与论文报告值相差悬殊：

| 指标 | 当前结果 | 论文值（10受试者均值）|
|------|---------|----------------------|
| 检索 Top-1（任意模态） | 23.5% | 61.2% |
| 检索 Top-5（任意模态） | 55.5% | 90.8% |
| CLIP 重建得分 | 0.5（随机水平） | 0.830 |
| PixCorr | 0.0 | 0.163 |
| SSIM | 0.006 | 0.398 |

重建指标（CLIP、AlexNet2/5、Inception）全部停在 0.5，恰好等于二选一识别任务的随机基线水平；PixCorr 为 0.0。这说明生成图像几乎是无意义的常量输出——问题不在于"训练不足"，而是流水线存在根本性的缺陷。本计划按优先级逐一排查并修复。

---

## 根本原因分析

### 1. 【严重】扩散嵌入字典的文件名碰撞

**文件**：`src/cogcappro/generate_image/generator.py` ~L272–290

扩散嵌入字典以 `os.path.basename(img_path)`（仅文件名）作为键。由于课程图像树按类别分文件夹存放，不同类别目录中的图像经常拥有相同文件名（例如 `class_A/n01234.jpg` 和 `class_B/n01234.jpg` 都被存到键 `n01234.jpg` 下），后者会悄悄覆盖前者。代码在生成嵌入时甚至会主动打印 "Duplicate filename … previous embedding will be overwritten" 警告，但并未阻止覆盖。

**后果**：大量测试图像在送入 SDXL-Turbo 时获得了错误的条件嵌入，模型实际上处于无条件（或错误条件）生成状态，输出图像几乎相同且与真实图像无关。这一个 bug 可以完整解释以下现象：
- CLIP / AlexNet / Inception 全部停在 0.5（二选一随机水平）
- PixCorr = 0.0（生成图像与真实图像在像素层面无相关性）
- SSIM ≈ 0.006（结构相似度接近于零）

**修复方案**：将嵌入键从 `os.path.basename(img_path)` 改为 `f"{class_name}/{img_filename}"`，其中 `class_name = os.path.basename(os.path.dirname(img_path))`。这样可以产生形如 `n01234_aircraft_carrier/n01234_aircraft_carrier_06s.jpg` 的唯一键，彻底避免跨类别碰撞。

同时更新 `src/cogcappro/align/data.py` 中的嵌入查找逻辑，使用相同的键方案。

---

### 2. 【严重】训练轮次严重不足

**文件**：`slurm_scripts/07_train_retrieval_full.sh`、`slurm_scripts/08_align_full.sh`

基础配置（`configs/cogcappro.yaml`）规定检索阶段分三段训练：20 + 40 + 20 = **80 epoch**。而实际运行仅训练了 **10 epoch**（仅为配置值的 1/8）；对齐阶段仅训练了 **1 epoch**（约为合理值的 1/15~1/20）。

**后果**：EEG 投影头严重欠训，无法将 EEG 信号可靠地映射到正确的 CLIP 嵌入空间区域。即便修复嵌入碰撞问题，欠训的骨干网络也会限制生成质量和检索准确率。

**修复方案**：检索训练增加至 80 epoch（使用配置默认值），对齐训练增加至 15 epoch。将分区从 `debug`（限 30 分钟）改为 `long_gpu` 或 `emergency_gpua40`。创建新脚本 `07b_train_retrieval_full_v2.sh` 和 `08b_align_full_v2.sh`，保留原始脚本不变。

---

### 3. 【较高】对齐阶段绕过了不确定性感知遮掩（UM）

**文件**：`src/cogcappro/align/data.py` ~L89–94

论文的核心贡献之一是不确定性加权遮掩（Uncertainty-weighted Masking，UM）模块。当前代码在对齐阶段强制覆盖为 `DirectT`（直接透传，即无遮掩）：

```python
config.data.uncertainty_aware = False
config.data.blur_type = OmegaConf.create(
    {"target": "cogcappro.models.inpainting_data.DirectT", "params": {}}
)
```

**后果**：对齐模型无法利用 UM 提供的空间选择性模糊信号来聚焦高确定性区域，降低了扩散空间对齐的质量。

**修复方案**：删除上述 6 行代码，使配置文件的默认值（`uncertainty_aware=True`）生效。若课程数据集的 FoveaBlur 特征缓存（`Image_feature_new/FoveaBlur/`）尚不存在，下次对齐运行时会自动计算（约需额外 10~30 分钟 GPU 时间）。

---

### 4. 【中等】数据集规模差异

论文在 10 名受试者 × 完整 Things-EEG 发布版上评估，而本次仅使用 1 名受试者的课程数据集（且预处理协议可能有所不同）。部分得分差距是不可避免的。但修复上述三个问题后，指标应能明显超过随机基线，进入单受试者的合理范围。

---

## 实施步骤

### 第一步：修复嵌入键碰撞

**文件**：`src/cogcappro/generate_image/generator.py`

在 `prepare_embedding()` 函数中，修改 ~L272–290：

- 新增：`class_name = os.path.basename(os.path.dirname(img_path))`
- 新增：`embed_key = f"{class_name}/{img_filename}"`
- 替换：`target_dict[img_filename] = valid_embedding` → `target_dict[embed_key] = valid_embedding`
- 删除"Duplicate filename"警告块（键唯一后不会再触发）

**文件**：`src/cogcappro/align/data.py`

在 `load_diffusion_embeddings()` 中修改查找逻辑（~L152–169）：

- 新增：`embed_key = f"{class_name}/{img_filename}"`（class_name 已计算）
- 将主查找键从 `img_filename` 改为 `embed_key`
- 保留简单的 basename 回退，以兼容旧版 `.pt` 文件
- 删除复杂的前缀/类后缀回退链（它是为错误键方案设计的补丁，不再需要）

修复后重新运行扩散嵌入准备脚本，验证作业日志中没有"Duplicate filename"警告。

---

### 第二步：恢复不确定性感知对齐

**文件**：`src/cogcappro/align/data.py`

删除第 89–94 行（强制 `uncertainty_aware=False` 和 `blur_type=DirectT` 的代码），让配置文件的默认值生效。

---

### 第三步：增加训练轮次

**新脚本**：`slurm_scripts/07b_train_retrieval_full_v2.sh`
- 去掉 `--max_epochs 10` 覆盖参数，使训练器使用配置默认值（80 epoch）
- 分区：`--partition long_gpu` 或 `emergency_gpua40`
- 时间限制：`--time 24:00:00`

**新脚本**：`slurm_scripts/08b_align_full_v2.sh`
- 设置 `--epoch 15`
- 分区：`--partition long_gpu`
- 时间限制：`--time 12:00:00`

---

### 第四步：重新运行完整流水线

所有代码修复完成后，按顺序运行：

| 步骤 | 脚本 | 等待条件 |
|------|------|---------|
| 1 | `02b_reprepare_diffusion_embeddings.sh` | 日志中零条"Duplicate filename"警告 |
| 2 | `07b_train_retrieval_full_v2.sh` | metrics.csv 中有 80 行 epoch 记录且 loss 下降 |
| 3 | `08b_align_full_v2.sh` | 作业正常退出 |
| 4 | `09_generate_full.sh` | `generated_image/all/` 下有生成图像 |
| 5 | `10_eval_reconstruction_full.sh` | `reconstruction_metrics.json` 已更新 |
| 6 | `11_multi_seed_summary.sh` | `summary_metrics.json` 已更新 |

---

## 关键文件清单

| 文件 | 作用 |
|------|------|
| `src/cogcappro/generate_image/generator.py` | 修复 basename→类名/basename 键（最高优先级）|
| `src/cogcappro/align/data.py` | 修复查找逻辑 + 删除 DirectT 强制覆盖 |
| `slurm_scripts/07b_train_retrieval_full_v2.sh` | 80 epoch 检索训练 |
| `slurm_scripts/08b_align_full_v2.sh` | 15 epoch 对齐训练 |
| `slurm_scripts/02b_reprepare_diffusion_embeddings.sh` | 重新生成正确的扩散嵌入 |
| `configs/cogcappro.yaml` | 查阅正确的训练阶段 epoch 数 |
| `configs/local.yaml` | 数据集与模型路径配置 |

---

## 验证清单

1. **嵌入修复**：`02b` 作业日志中**零条**"Duplicate filename"警告。
2. **训练**：`runs/full/.../lightning_logs/metrics.csv` 有 80 行 epoch 记录且 loss 持续下降。
3. **对齐**：对齐作业日志显示 FoveaBlur 缓存已加载或已新建。
4. **重建指标**：`reconstruction_metrics.json` 中 CLIP > 0.6，PixCorr > 0.05。
5. **检索指标**：`test_results.json` 中任意模态 Top-1 > 30%。

---

## 修复后的预期改善

| 指标 | 修复前 | 修复后预期 |
|------|--------|-----------|
| 检索 Top-1（任意模态）| 23.5% | 40–55% |
| 检索 Top-5（任意模态）| 55.5% | 75–88% |
| CLIP 重建得分 | 0.5 | 0.65–0.80 |
| PixCorr | 0.0 | 0.05–0.15 |
| SSIM | 0.006 | 0.10–0.30 |

注：单受试者性能无法达到论文10受试者均值（61.2% Top-1），但应明显高于随机基线。

---

## 已知的残留局限性

即使完成所有修复，仍可能存在与论文的部分差距，原因如下：
- 课程数据集仅涵盖 1 名受试者，论文取 10 名均值。
- 课程数据的预处理协议可能与原始 Things-EEG 发布版有所不同。
- 论文可能使用了公开配置中未体现的额外训练技巧或超参数调优。

---

## 执行过程中发现的额外 Bug

运行修复后的流水线时，又发现两个关键问题：

### Bug 5. 【严重】VAE Float16 溢出 → 生成图像全为纯黑

**文件**：`src/cogcappro/generate_image/generator.py` — `_init_pipeline()`

原始代码调用 `self.pipe.upcast_vae()` 后，立即又将 VAE 强制恢复为 float16：
```python
if hasattr(self.pipe, "vae") and getattr(self.pipe.vae.config, "force_upcast", False):
    self.pipe.vae.config.force_upcast = False
    self.pipe.vae.to(dtype=torch.float16)
```
此操作抵消了 upcast，导致 VAE 解码器以 float16 运行。SDXL-Turbo 的 VAE 在 float16 下已知会发生数值溢出，产生 NaN 值，NaN 转为 uint8 后变为 0（纯黑像素）。200 张生成图像因此字节完全相同（全为纯黑），MD5 一致。

**修复方案**：将上述 3 行代码替换为 `self.pipe.vae.config.force_upcast = True`。流水线自身的解码逻辑会临时将 VAE 提升至 float32 后再解码，避免 NaN 溢出。

### Bug 6. 【较高】`guidance_scale=0.0` 时 IP-Adapter 嵌入维度错误

**文件**：`src/cogcappro/generate_image/generator.py` — `_prepare_embeddings()`

当 `guidance_scale=0.0` 时，`do_classifier_free_guidance=False`，流水线的 `prepare_ip_adapter_image_embeds` 直接透传输入嵌入（不执行 `chunk(2)` 拆分）。然而 `_prepare_embeddings` 始终将 `[uncond, cond]` 堆叠为 `[2, 1, 1024]`。这个 3D 张量被原样传入 IP-Adapter 的交叉注意力处理器，处理器将其 reshape 为 `[1, 2, heads, head_dim]`，把零填充的 uncond 行和真实 EEG 行都当作条件 token——信号被稀释，叠加上 Bug 5 的效果，导致输出完全相同。

**修复方案**：当 `do_classifier_free_guidance=False` 时，返回 `embed.unsqueeze(0)`，形状为 `[1, 1, 1024]`（仅条件嵌入，满足 3D 要求）；当 `do_cfg=True` 时，保持原来的 `[2, 1, 1024]` 堆叠。

---

## 实际达成的结果（2026-04-19 至 2026-04-20）

全部修复已应用，流水线在受试者 sub-01、seed 0 上重新运行。

### 检索任务（EEG → CLIP 匹配）

| 指标 | 修复前 | 修复后 | 论文值 |
|------|--------|--------|--------|
| Top-1（任意模态） | 23.5% | **61.0%** | 61.2% |
| Top-5（任意模态） | 55.5% | **88.0%** | 90.8% |

检索性能与论文几乎完全一致。

### 重建任务（IP-Adapter 图像生成）

经过对齐阶段的多次尝试（详见下节），最终结果如下：

| 指标 | `all_before`（EEG 直接输出）| `all`（SimpleAlignPipe 对齐后）| 论文值 |
|------|----------------------------|-------------------------------|--------|
| CLIP (↑) | **0.707** | 0.659 | 0.830 |
| PixCorr (↑) | 0.130 | **0.133** | 0.163 |
| SSIM (↑) | **0.316** | 0.236 | 0.398 |
| AlexNet-2 (↑) | **0.663** | 0.618 | 0.831 |
| AlexNet-5 (↑) | **0.698** | 0.682 | 0.937 |
| Inception (↑) | 0.597 | **0.607** | 0.720 |

`all_before` 模式（直接使用 EEG 检索模型的 CLIP 嵌入驱动 IP-Adapter）达到了**论文重建指标约 80% 的水平**，是目前最优结果。SimpleAlignPipe 对齐后的 PixCorr 和 Inception 略有提升，但 CLIP、SSIM、AlexNet 指标下降，综合来看未获得明显改善。

### 总结

流水线现已正常工作。与论文的剩余差距主要来自：
1. **对齐阶段无法带来提升**：EEG 检索模型已经输出 CLIP 兼容嵌入，IP-Adapter 直接使用效果最好；将其投影到"扩散嵌入空间"反而引入噪声。
2. **单受试者 vs. 10 受试者均值**：不可避免的固有差距。

---

## 对齐阶段深入调查（2026-04-20）

为修复最初的模式崩溃问题（DiffusionPriorUNet，15 epoch 后余弦相似度 0.005，原因为训练步数过少导致 warmup 过长），先后尝试了两种方案：

### 方案一：DiffusionPriorUNet（30 epoch，修复 warmup）

**原崩溃根因**：`num_warmup_steps=100` 被硬编码，而实际训练总步数仅约 30 步（batch_size=10240，训练集 16,540 样本）。LR 在整个训练中始终处于线性预热阶段，从未达到目标值。

**代码修复**（`diffusion_pipe.py`、`main.py`）：
- warmup 步数 = `max(1, total_steps // 10)`（比例计算，不再硬编码）
- 训练 batch_size 10240 → 512（每 epoch 33 步，warmup 比例正常）

**结果**：仍崩溃。30 epoch 后最佳余弦相似度仅 0.009。训练损失有所下降（1.19 → 0.33），但扩散推理阶段无法产生与目标对齐的嵌入。该模型可能需要 100+ epoch 才能收敛。

### 方案二：SimpleAlignPipe（直接 MLP，100 epoch，修复损失函数）

**代码修复**（`diffusion_pipe.py`）：
- `SimpleAlignMLP.forward()`：去除与未归一化 MSE 目标冲突的 L2 归一化输出
- `SDEmbeddingLoss`：对预测值和目标值均归一化后再计算 MSE；去除 `loss_reg = u.pow(2).mean()`（对已归一化输出无意义）
- `OneCycleLR`：将 `max_lr` 从 `lr×8` 降低至 `lr×3`；重新启用模态 masking

**结果**：验证余弦相似度达到 **0.770**（第 31 epoch 最优），在第 51 epoch 触发 early stopping。相比 DiffusionPriorUNet（0.009），提升极为显著。

**重建指标**：相比 `all_before` 结果喜忧参半（详见上表）。对齐嵌入（与扩散目标余弦相似度 0.770）在 PixCorr 和 Inception 上略优，但 CLIP、SSIM 等关键指标反而下降。说明 EEG CLIP 嵌入已天然适合 IP-Adapter 的语义空间，将其投影至"扩散嵌入空间"引入了失真。
