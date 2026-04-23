# VisualEEGDecoding 任务交接文档

**时间：** 2026-04-22  
**接手方：** Codex  
**任务：** 在 DSAA2012 课程数据上复现 VisualEEGDecoding（Brain-to-Image Retrieval）

---

## 关键路径

```
REPO = /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/repository/VisualEEGDecoding

DATA = /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data

CONDA ENV = test  (Python 3.10, torch 2.10+cu126, open-clip-torch 2.32.0)
```

---

## 已完成的工作

### ✅ 数据符号链接（已设置）

```
$REPO/data/things-eeg/Preprocessed_data/sub-01/train.pt
  → $DATA/train.pt   (原始课程数据，格式正确，eeg shape [16540, 80, 63, 250] float16)

$REPO/data/things-eeg/Preprocessed_data/sub-01/test.pt
  → $DATA/test.pt    (格式正确，eeg shape [200, 80, 63, 250] float16)

$REPO/data/things-eeg/Image_set/train_images
  → $DATA/training_images/

$REPO/data/things-eeg/Image_set/test_images
  → $DATA/test_images/
```

**img 字段格式确认（关键）：**  
`test.pt` 的 `img` 字段是 ndarray shape `(200, 80)` dtype `<U59`，  
值为相对路径，如：`'test_images/00001_aircraft_carrier/aircraft_carrier_06s.jpg'`  
→ 与 VisualEEGDecoding 的特征字典 key 格式一致，**无需额外转换**。

### ✅ OpenCLIP RN50 模型（已下载）

```
/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/open_clip_pytorch_model.bin
大小：408.4 MB
```

注：`process_image_course.py` 使用 `pretrained='openai'`（open_clip 自动用缓存），不直接读这个 .bin 文件。

### ✅ 已创建的脚本文件

| 文件 | 用途 |
|------|------|
| `$REPO/preprocess/process_image_course.py` | 生成 12 级模糊 RN50 图像特征 |
| `$REPO/main_eeg_course.py` | 训练 EEG 编码器（单受试者，10个种子） |
| `$REPO/slurm_scripts/02_gen_blur_features.sh` | SLURM job：特征生成（debug 分区） |
| `$REPO/slurm_scripts/03_train_eeg.sh` | SLURM job：模型训练（gpu 分区） |
| `$REPO/scripts/evaluate_course_metrics.py` | 读取训练结果，输出课程格式指标 |
| `$REPO/scripts/download_rn50.py` | 下载 RN50（已完成，备用） |
| `$REPO/scripts/inspect_course_data.py` | 检查数据格式（已完成，备用） |

---

## 当前状态：等待 SLURM Job 完成

### Job 9703515（特征生成）

```bash
# 提交时间：2026-04-22
# 分区：debug（最多 30 分钟）
# 节点：gpu3-9
# 状态：已提交/运行中（可能已完成）
```

**检查方法：**
```bash
squeue -u ckwong627          # 查看是否还在运行
# 如果已结束，查看日志：
cat $REPO/slurm_scripts/logs/02_gen_blur_features_9703515.out
cat $REPO/slurm_scripts/logs/02_gen_blur_features_9703515.err
```

**预期输出：**
```
$REPO/data/things-eeg/Image_feature/MultiBlur_RN50_train.pt  (~600MB)
$REPO/data/things-eeg/Image_feature/MultiBlur_RN50_test.pt   (~15MB)
```

---

## 下一步操作

### Step A：检查 Job 9703515 结果

```bash
ls -lh /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/repository/VisualEEGDecoding/data/things-eeg/Image_feature/
```

**情况1：特征文件已生成** → 直接进行 Step B

**情况2：Job 还在运行** → 等待完成再继续

**情况3：Job 失败** → 查看错误日志，修复后重新提交：
```bash
cat $REPO/slurm_scripts/logs/02_gen_blur_features_9703515.err
# 修复问题后：
sbatch $REPO/slurm_scripts/02_gen_blur_features.sh
```

常见问题：
- 如果 debug 分区 30 分钟不够，改 `03` 脚本的分区为 `gpu` 并重新提交
- 如果 open_clip 下载失败（网络问题），改用本地 bin 文件加载（见下方备用方案）

**备用：改用本地 bin 文件加载 RN50（如果 open_clip 网络下载失败）**

在 `process_image_course.py` 的 `Make_dataset.__init__` 中，将：
```python
self.vlmodel, _, _ = open_clip.create_model_and_transforms(
    'RN50', pretrained='openai'
)
```
改为：
```python
self.vlmodel, _, _ = open_clip.create_model_and_transforms(
    'RN50',
    pretrained='/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/open_clip_pytorch_model.bin'
)
```
注：这样加载可能有 QuickGELU 不匹配问题，特征会有微小差异，但可以正常运行。

---

### Step B：提交训练 Job

特征生成成功后：

```bash
cd /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/repository/VisualEEGDecoding

sbatch slurm_scripts/03_train_eeg.sh
```

**训练参数：**
- 受试者：sub-01（课程单一受试者）
- Epoch：200
- Batch size：1024
- LR：0.001
- Seeds：21-30（10个种子，满足课程 mean ± std 要求）
- 分区：gpu（4小时，时间应该足够）

**监控训练：**
```bash
squeue -u ckwong627
tail -f $REPO/slurm_scripts/logs/03_train_eeg_<JOBID>.out
```

---

### Step C：查看结果

训练完成后：

```bash
conda run -n test python $REPO/scripts/evaluate_course_metrics.py
```

输出格式：
```
=== Course Evaluation Metrics (val-selected model) ===
Top-1 Accuracy: X.XXXX ± X.XXXX
Top-5 Accuracy: X.XXXX ± X.XXXX
```

详细结果在：
```
$REPO/logs/main_eeg_course/Brain_Visual_Encoder_EEG/<timestamp>/all_metrics.csv
```

---

## 课程评估要求

来自 `/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/Final_Project_Instructions/`：

- **Retrieval Task**：200-way zero-shot，Top-1 和 Top-5 准确率
- **报告格式**：mean ± std（10 个随机种子）
- **评分**：Top-1（12.5分）+ Top-5（12.5分）= 共 25 分

---

## 论文对比（参考）

论文路径：`$REPO/../../../references/paper/Liu 等 - 2026 - Leveraging Visual Blur Perception Characteristics for EEG Decoding.pdf`

| 指标 | 论文（10受试者平均） | 预期（本次1受试者） |
|------|--------------------|--------------------|
| Top-1 | ~80% | 60-75% |
| Top-5 | ~96.9% | 85-95% |

---

## 注意事项

1. **debug 分区限制**：最多 2 块 A40，最多 30 分钟
2. **如果特征生成超时**：改用 `gpu` 分区（修改 `02_gen_blur_features.sh` 中的 `#SBATCH -p debug` 为 `#SBATCH -p gpu`）
3. **训练日志目录**：`$REPO/logs/main_eeg_course/`（自动按时间戳创建）
4. **模型权重保存位置**：同训练日志目录，文件名格式 `Brain_Visual_Encoder_EEG_sub1_seed<N>_best.pth`
