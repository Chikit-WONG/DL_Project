# VisualEEGDecoding 复现计划（中文）

## 背景

复现 **VisualEEGDecoding** 仓库（AAAI 2026 论文：《Leveraging Visual Blur Perception Characteristics for EEG Decoding》），在课程数据集（DSAA2012 Project A）上完成 **脑-图像检索（Brain-to-Image Retrieval）** 任务。论文在原始 Things-EEG 数据集（10 名受试者）上报告了 Top-1 准确率 80%、Top-5 准确率 96.9%。我们只有课程指定的单一受试者（sub-01）数据，需要相应调整 pipeline。

**仓库路径：** `/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/repository/VisualEEGDecoding`  
（下文简称 `$REPO`）

---

## 关键发现

### 数据格式不兼容
- **课程数据**（`image-eeg-data/train.pt`、`test.pt`）：由教学团队预处理，包含 EEG 张量 + 图像 ID（如 `"aardvark_01b"`）。形状：`[N_images, N_trials, 63, 250]`
- **VisualEEGDecoding 期望格式**：`{'eeg': np.float16 [N, T, 63, 250], 'img': np.array [N, T] 图像相对路径}` 其中图像路径相对于 `Image_set/`（如 `"train_images/00001_aircraft_carrier/aircraft_carrier_01b.jpg"`）

### 模型依赖
- 需要 **OpenCLIP RN50** 预训练权重，用于生成 12 个高斯模糊级别的图像特征（每个 1024 维）
- 图像特征文件格式：`{blur_key: {image_path: torch.Tensor(1024)}}`，键为 `'1','3','9','15','21','27','33','39','45','51','57','63'`

### 运行环境
- 复用 **`test` conda 环境**（Python 3.10、torch 2.10+cu126、open-clip-torch 2.32.0、mne 1.8.0、scipy、numpy——所有必要包均已安装）
- 仓库的 `environment.yml` 是 Windows 版本，忽略即可

### 任务范围
- 只做**受试者内（intra-subject）训练**（1 个受试者 = sub-01），不做跨受试者训练（需要原始 Things-EEG 10 名受试者数据，我们没有）
- 该仓库**只有检索任务，没有重建（Reconstruction）任务**

---

## 实现步骤

### 阶段 0：检查课程数据格式（登录节点，无需 GPU）

**脚本：** `$REPO/scripts/inspect_course_data.py`

在登录节点用 `conda run -n test` 运行，打印 `train.pt` 和 `test.pt` 的精确键名和形状，为后续转换提供依据。

```bash
conda run -n test python $REPO/scripts/inspect_course_data.py
```

需要确认的关键信息：
- `train.pt` 字典的键名
- `eeg` 张量的形状
- 图像 ID 字段的键名和值格式

---

### 阶段 1：下载 OpenCLIP RN50 模型（登录节点）

**模型存放路径：** `/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/`

`process_image.py` 调用 `open_clip.create_model_and_transforms('RN50', pretrained=<本地路径>)`。我们改用 `pretrained='openai'` 自动从 OpenAI CDN 下载，然后保存缓存的权重。

**脚本：** `$REPO/scripts/download_rn50.py`

```python
import open_clip, torch, os

model, _, _ = open_clip.create_model_and_transforms('RN50', pretrained='openai')
save_dir = '/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai'
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, 'open_clip_pytorch_model.bin'))
print("已保存到", save_dir)
```

运行：`conda run -n test python $REPO/scripts/download_rn50.py`

**模型大小：** ~100MB（OpenAI CLIP RN50 权重）

---

### 阶段 2：将课程数据转换为 VisualEEGDecoding 格式（登录节点）

**脚本：** `$REPO/scripts/convert_course_data.py`

该脚本执行以下操作：
1. 加载课程 `train.pt` 和 `test.pt`
2. 扫描 `training_images/` 和 `test_images/` 目录，建立从图像 stem → 相对路径的映射（如 `"aircraft_carrier_01b"` → `"train_images/00001_aircraft_carrier/aircraft_carrier_01b.jpg"`）
3. 将 `eeg['img']` 中的图像 ID 转换为相对路径
4. 确保 EEG 形状为 `[N, T, 63, 250]`，numpy float16 格式
5. 创建目录结构并保存：
   - `$REPO/data/things-eeg/Preprocessed_data/sub-01/train.pt`
   - `$REPO/data/things-eeg/Preprocessed_data/sub-01/test.pt`
6. 创建符号链接：
   - `$REPO/data/things-eeg/Image_set/train_images` → `image-eeg-data/training_images`
   - `$REPO/data/things-eeg/Image_set/test_images` → `image-eeg-data/test_images`

**输出字典格式：**
```python
{
  'eeg': np.array([N_images, N_trials, 63, 250], dtype=np.float16),
  'img': np.array([N_images, N_trials], dtype=object),  # 相对路径
  'label': ...,  # 从原始数据保留
}
```

运行：`conda run -n test python $REPO/scripts/convert_course_data.py`

---

### 阶段 3：生成多模糊图像特征（SLURM debug 作业）

**修改后的脚本：** `$REPO/preprocess/process_image_course.py`（新文件，基于 `process_image.py` 改写）

相对于原始脚本的改动：
- 修复 Windows 路径分隔符（`\\` → `/`，统一使用 `os.path.join`）
- 更新 `base_path` 为 `$REPO/data/things-eeg/Image_set`
- 更新预训练权重路径为 `/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/open_clip_pytorch_model.bin`
- 更新保存路径为 `$REPO/data/things-eeg/Image_feature/`

**SLURM 脚本：** `$REPO/slurm_scripts/02_gen_blur_features.sh`
```bash
#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --time=00:30:00
#SBATCH -o $REPO/slurm_scripts/logs/02_gen_blur_features_%j.out
#SBATCH -e $REPO/slurm_scripts/logs/02_gen_blur_features_%j.err
source ~/miniconda3/etc/profile.d/conda.sh && conda activate test
cd $REPO
python preprocess/process_image_course.py
```

**输出：**
- `$REPO/data/things-eeg/Image_feature/MultiBlur_RN50_train.pt`（~600MB）
- `$REPO/data/things-eeg/Image_feature/MultiBlur_RN50_test.pt`（~15MB）

**时间预估：** 1 块 GPU 约 20-25 分钟（12 个模糊级别 × 约 8000 张训练图像）

---

### 阶段 4：训练 EEG 编码器（SLURM 非 debug 作业）

**修改后的脚本：** `$REPO/main_eeg_course.py`（新文件，基于 `main_eeg.py` 改写）

相对于原始脚本的改动：
- `data_path` 设为 `$REPO/data/things-eeg`
- 只循环 `sub=1`（单受试者），而非 sub 1-10
- `cross_subject=False`（只做受试者内评估）
- 使用 10 个随机种子（21-30），以满足课程要求的 mean ± std 格式

**SLURM 脚本：** `$REPO/slurm_scripts/03_train_eeg.sh`
```bash
#!/bin/bash
#SBATCH -p gpu_8h   # 或合适的分区
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --time=02:00:00
#SBATCH -o $REPO/slurm_scripts/logs/03_train_eeg_%j.out
#SBATCH -e $REPO/slurm_scripts/logs/03_train_eeg_%j.err
source ~/miniconda3/etc/profile.d/conda.sh && conda activate test
cd $REPO
python main_eeg_course.py
```

**时间预估：** 每个种子约 20 分钟 × 10 个种子 ≈ 3-4 小时（需使用 `gpu` 或 `gpu_8h` 分区）

**超参数（与论文一致）：**
- epochs=200, lr=0.001, batch_size=1024, 不用 mixup, 不用滤波器
- 全部 63 个通道，全部 250 个时间点

---

### 阶段 5：用课程指标评估

**脚本：** `$REPO/scripts/evaluate_course_metrics.py`

加载训练好的模型，按课程评估协议运行：
- 200-way 零样本检索（200 张测试图像）
- 报告 Top-1 和 Top-5 准确率
- 用 10 个随机种子重复，报告 mean ± std

---

## 关键文件清单

| 文件 | 操作 | 用途 |
|------|------|------|
| `$REPO/scripts/inspect_course_data.py` | **新建** | 检查课程数据格式 |
| `$REPO/scripts/download_rn50.py` | **新建** | 下载 OpenCLIP RN50 |
| `$REPO/scripts/convert_course_data.py` | **新建** | 转换 EEG 数据格式 |
| `$REPO/preprocess/process_image_course.py` | **新建** | 生成多模糊特征（Linux 修复版） |
| `$REPO/main_eeg_course.py` | **新建** | 单受试者 × 10 seeds 训练 |
| `$REPO/slurm_scripts/02_gen_blur_features.sh` | **新建** | SLURM 特征生成作业 |
| `$REPO/slurm_scripts/03_train_eeg.sh` | **新建** | SLURM 训练作业 |
| `$REPO/scripts/evaluate_course_metrics.py` | **新建** | 最终评估脚本 |

---

## 模型/数据下载

| 项目 | 大小 | 存储路径 | 命令 |
|------|------|----------|------|
| OpenCLIP RN50 (openai) | ~100MB | `/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/` | `python scripts/download_rn50.py` |

---

## SLURM 分区策略

| 步骤 | 分区 | 时间 | GPU | 说明 |
|------|------|------|-----|------|
| 特征生成 | `debug` | 30 分钟 | 1 块 A40 | 在 debug 配额内能完成 |
| 训练（10 seeds） | `gpu` 或 `gpu_8h` | 3-4 小时 | 1 块 A40 | 超过 debug 的 30 分钟限制 |

---

## 验证方法

1. **阶段 2 转换后**：检查 `$REPO/data/things-eeg/Preprocessed_data/sub-01/train.pt` 的键名和形状是否正确
2. **阶段 3 后**：验证 `MultiBlur_RN50_train.pt` 的字典结构——12 个键，每个键映射约 8000 条图像路径到 1024 维张量
3. **阶段 4 后**：训练日志应显示 Top-1 准确率逐渐上升，向 60-80% 趋近
4. **最终评估**：报告 Top-1 和 Top-5 准确率（mean ± std，10 seeds，200-way）

---

## 预期结果与论文比较

| 指标 | 论文（10 受试者平均，200 张图像） | 预期（1 受试者） |
|------|---------------------------------|----------------|
| Top-1 准确率 | ~80% | 60-75%（较低，因为只有 1 个受试者） |
| Top-5 准确率 | ~96.9% | 85-95% |

预期低于论文的原因：
- 论文使用 10 名受试者的数据（数据多样性更高）
- 论文使用原始 Things-EEG 数据（经过其自有 pipeline 预处理）
- 课程数据的预处理方式可能与 Things-EEG 不同

---

## 注意事项

- 该仓库**没有脑图像重建模块**，只有检索任务
- `main_meg.py` 需要 MEG 数据（不可用），跳过
- `visualization/brain_area.ipynb` 可在训练后运行，用于分析空间滤波器
- 如果 debug 分区排队时间过长，换用 `short` 或 `gpu_4h` 分区
- 若 debug 分区 30 分钟内无法完成特征生成，将该作业也移至 `gpu` 分区
