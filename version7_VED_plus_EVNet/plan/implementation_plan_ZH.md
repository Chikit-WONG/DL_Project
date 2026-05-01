# Version 7：VED + EVNet Fixed —— 实现计划（中文）

## 概述

本版本复现并扩展了参考实现（`temp/yliu674/v5_evnet`）中 EVNet 增强的脑电图-图像检索流程。共运行四组实验，涵盖两种模糊级别配置与两种训练数据策略，完成课程 Task 1（EEG-to-image retrieval）。

---

## 模型架构

### Brain_Visual_Encoder_EEG

模型将原始 EEG 信号编码，通过对比学习（CLIP 式损失）检索匹配图像。

**EEG 分支：**
- 空间卷积（`Conv2dWithAbs`，25 个滤波器，覆盖 63 个通道）
- 时间 MLP：250 → 200 → 200（ELU + Dropout）
- 线性适配器：25×200 → 1024

**图像分支——多尺度模糊特征：**
- 使用 OpenCLIP RN50 在 8 或 12 种高斯模糊级别下预计算
- 通过可学习注意力权重（`learned_scale`）聚合 → 1024 维向量

**EVNet Fixed 融合：**
- EVNet 预计算特征：EVNetFrontEnd（SubcorticalBlock + VOneBlock，所有权重固定）→ RN50 → 1024 维
- 融合方式：`fused = softmax(fusion_logits)[0] × blur_agg + softmax(fusion_logits)[1] × evnet_feat`
- 融合 MLP（`fusion_adapter`）：1024 → 768 → 1024
- `fusion_logits` 初始值为 `[0.7, 0.3]`，训练过程中可学习

**"Fixed"（固定）** 指 EVNet 神经生物学前端的权重在训练中从不更新——只有融合权重和适配器被训练。

---

## 模糊级别配置

| 配置 | 模糊级别 |
|---|---|
| 8 级 | `l_1, l_3, l_15, l_21, l_33, l_45, l_57, l_63` |
| 12 级 | `l_1, l_3, l_9, l_15, l_21, l_27, l_33, l_39, l_45, l_51, l_57, l_63` |

---

## 实验设计

### Phase A —— 95/5 分割（基于验证集选取检查点）

- `train.pt` 的 95% 作为训练集，5% 作为验证集
- 按验证集 Top-1 准确率选取最优检查点
- 在完整测试集上评估

### Phase B —— 完整训练集（无验证集分割）

- `train.pt` 的 100% 作为训练集，无验证集
- `select` 检查点 = 最后一个 epoch；`best` 检查点 = 所有 epoch 中测试集最优
- 在完整测试集上评估

### 四组实验

| 提交脚本 | 配置 | 阶段 |
|---|---|---|
| `02_train_8blur_evnet_split.sh` | 8 blur + EVNet fixed | A（分割） |
| `03_train_12blur_evnet_split.sh` | 12 blur + EVNet fixed | A（分割） |
| `04_full_train_8blur_evnet.sh` | 8 blur + EVNet fixed | B（完整） |
| `05_full_train_12blur_evnet.sh` | 12 blur + EVNet fixed | B（完整） |

所有实验：10 个随机种子（21–30），200 个 epoch，batch size 1024，lr 0.001，AdamW 优化器。

---

## 关键文件路径

| 资源 | 路径 |
|---|---|
| 主训练脚本 | `version7_VED_plus_EVNet/main_eeg_course.py` |
| 特征生成脚本 | `version7_VED_plus_EVNet/preprocess/process_image_course.py` |
| MultiBlur 特征 | `output/Image_feature/MultiBlur_RN50_{train,test}.pt`（符号链接 → version5_VED） |
| EVNet 特征 | `output/Image_feature/EVNet_RN50_{train,test}.pt`（由脚本 01 生成） |
| EEG 训练/测试数据 | `.../image-eeg-data/converted_for_cogcappro/ThingsEEG/Preprocessed_data_250Hz_whiten/sub-01/{train,test}.pt` |
| CLIP RN50 权重 | `/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/open_clip_pytorch_model.bin` |
| 训练输出日志 | `output/logs/{8,12}blur_evnet_{split,full}/` |

---

## 运行顺序

### 第一步：生成 EVNet 特征

```bash
sbatch slurm_scripts/01_gen_evnet_features.sh
```

等待任务完成后验证：
```bash
ls output/Image_feature/
# 应能看到 EVNet_RN50_train.pt 和 EVNet_RN50_test.pt
```

### 第二步（可选）：冒烟测试

在 debug 节点上交互式运行（1 epoch，1 个种子）：
```bash
python main_eeg_course.py \
  --epoch 1 --n_seeds 1 --first_seed 999 \
  --blur_config 8 --use_evnet \
  --output_dir output/logs/smoke_test
```

### 第三步：Phase A 训练（可并行提交）

```bash
sbatch slurm_scripts/02_train_8blur_evnet_split.sh
sbatch slurm_scripts/03_train_12blur_evnet_split.sh
```

### 第四步：Phase B 训练（可并行提交）

```bash
sbatch slurm_scripts/04_full_train_8blur_evnet.sh
sbatch slurm_scripts/05_full_train_12blur_evnet.sh
```

---

## 预期结果

来自参考 README（`temp/yliu674/README.md`）：

| 配置 | 验证集选取 Top-1 | 验证集选取 Top-5 | 最优测试 Top-1 | 最优测试 Top-5 |
|---|---|---|---|---|
| 8 blur + EVNet fixed | 0.8530 ± 0.0081 | 0.9845 ± 0.0035 | 0.8890 ± 0.0107 | 0.9855 ± 0.0035 |
| 12 blur + EVNet（非 fixed） | 0.8325 ± 0.0237 | 0.9825 ± 0.0040 | 0.8825 ± 0.0051 | 0.9820 ± 0.0060 |

12 blur + EVNet **fixed** 的结果在 README 中未记录——本实验将提供新数据。

---

## 差异分析

若结果与参考值偏差超过约 2%，按以下顺序排查：

1. **图像键名不匹配**：version5_VED 生成的模糊特征 `.pt` 与 version7 生成的 EVNet 特征 `.pt` 必须使用相同的相对路径格式（`train_images/category/file.jpg`）。两者均由相同的 `collect_image_paths` 函数生成，应一致。验证方法：
   ```python
   import torch
   blur = torch.load('output/Image_feature/MultiBlur_RN50_train.pt', weights_only=False)
   evnet = torch.load('output/Image_feature/EVNet_RN50_train.pt', weights_only=False)
   print(len(set(blur['1'].keys()) & set(evnet.keys())))
   ```

2. **EVNet 是否真正固定**：确认 `process_image_course.py` 中 `EVNetFrontEnd` 所有参数的 `requires_grad=False`（代码中已执行此设置）。

3. **随机种子对齐**：`set_seed()` 同时设置了 `torch.manual_seed` 和 `np.random.seed` ✓

4. **模糊级别集合正确性**：8 级 = `['l_1','l_3','l_15','l_21','l_33','l_45','l_57','l_63']`（与参考一致）✓

5. **EEG 数据版本**：确认 `Preprocessed_data_250Hz_whiten` 是参考实现所用的正确预处理变体。
