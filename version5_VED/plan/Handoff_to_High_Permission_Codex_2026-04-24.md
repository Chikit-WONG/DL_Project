# version5_VED 交接文档（给更高权限 Codex）

更新时间：2026-04-24

## 1. 当前目标

本仓库 `version5_VED` 已完成 task 2 的主要代码实现和文档整理。接下来需要由一个**更高权限、能正常提交和等待 HPC 作业**的 Codex 会话完成：

1. 提交并跑通 task 2 smoke test
2. smoke 通过后提交 full run
3. 等待作业完成
4. 检查输出、汇总指标、必要时修补小问题
5. 最终整理为可提交到 GitHub 的状态

## 2. 已完成内容

### 2.1 task 2 主体实现已落地

已新增脚本：

- `scripts/task2_common.py`
- `scripts/train_task2_semantic.py`
- `scripts/generate_task2_reconstructions.py`
- `scripts/evaluate_task2_reconstruction.py`
- `scripts/run_task2_pipeline.py`
- `scripts/make_task2_qualitative_grid.py`

### 2.2 方案实现内容

当前 task 2 的实现与之前讨论和计划一致：

- 复用 `version5_VED` 的 task 1 EEG retrieval backbone
- 保留 image alignment 路线
- 新增 class-text prototype supervision
- 使用 OpenCLIP RN50 text prototype
- 在同一 1024 维空间中做 image + text semantic joint fine-tune
- 推理时先检索训练图像，再按 top-k 结果做 class aggregation
- 用聚合出的训练类填固定模板 prompt：
  - `a realistic photo of a {class_name}`
- 用 top-1 检索图作为 IP-Adapter reference image
- 用 Stable Diffusion v1.5 + IP-Adapter 生成重建图像
- 用课程口径评估 `SSIM` 和 `CLIP`
- 所有输出统一写入 `output/`

### 2.3 已新增/修改文档

已更新：

- `version5_VED/README.md`
- `version5_VED/README-CN.md`
- `version5_VED/同学运行说明.md`
- 根目录 `README.md`
- 根目录 `README-CN.md`

已新增计划文件：

- `plan/Task2_Retrieval_Augmented_Reconstruction_Plan_EN.md`
- `plan/Task2_Retrieval_Augmented_Reconstruction_Plan_ZH.md`

### 2.4 已新增运行脚本

SLURM 脚本：

- `slurm_scripts/04_run_task2_smoke.sh`
- `slurm_scripts/05_run_task2_full.sh`

直接运行脚本（给不用 `sbatch` 的 A800 机器）：

- `run_task1_direct.sh`
- `run_task2_direct.sh`

### 2.5 已完成的代理修复

用户自己验证过：如果在提交作业前先执行 `unclash`，再 `sbatch`，作业可以正常运行。

因此我已经把下面这段逻辑固化进以下脚本开头：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
type unclash >/dev/null 2>&1 && unclash || true
```

已加到：

- `slurm_scripts/02_gen_blur_features.sh`
- `slurm_scripts/03_train_eeg.sh`
- `slurm_scripts/04_run_task2_smoke.sh`
- `slurm_scripts/05_run_task2_full.sh`
- `run_task1_direct.sh`
- `run_task2_direct.sh`

文档中也已经同步说明。

## 3. 已验证内容

### 3.1 Python 语法检查通过

已通过：

```bash
python -m py_compile \
  scripts/task2_common.py \
  scripts/train_task2_semantic.py \
  scripts/generate_task2_reconstructions.py \
  scripts/evaluate_task2_reconstruction.py \
  scripts/run_task2_pipeline.py \
  scripts/make_task2_qualitative_grid.py
```

### 3.2 Shell 脚本语法检查通过

已通过：

```bash
bash -n \
  slurm_scripts/02_gen_blur_features.sh \
  slurm_scripts/03_train_eeg.sh \
  slurm_scripts/04_run_task2_smoke.sh \
  slurm_scripts/05_run_task2_full.sh \
  run_task1_direct.sh \
  run_task2_direct.sh
```

### 3.3 重要实现修正

在 `train_task2_semantic.py` 中，validation prompt-class 指标已经修正为**和实际生成逻辑一致**：

- 现在使用 `aggregate_retrieval(...)`
- 不再只是简单拿 top-k 里的第一个 unique class

这点很重要，否则训练时的语义指标和生成时的 class selection 不一致。

## 4. 当前已有数据与资源

### 4.1 课程数据路径

```text
/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data
```

### 4.2 模型路径

```text
/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/open_clip_pytorch_model.bin
/hpc2hdd/home/ckwong627/workdir/models/stable-diffusion-v1-5
/hpc2hdd/home/ckwong627/workdir/models/IP-Adapter
```

这些路径已经存在。本地也已经确认：

- `CLIP-RN50-openai/open_clip_pytorch_model.bin` 存在
- `stable-diffusion-v1-5` 目录存在
- `IP-Adapter/models/ip-adapter_sd15.bin` 和 `models/image_encoder/*` 存在

### 4.3 task 1 已有输出

已确认存在：

```text
output/Image_feature/MultiBlur_RN50_train.pt
output/Image_feature/MultiBlur_RN50_test.pt
```

也已确认已有 task 1 checkpoint，例如：

```text
output/logs/main_eeg_course/Brain_Visual_Encoder_EEG/2026-04-23-16-33/Brain_Visual_Encoder_EEG_sub1_seed21_select.pth
```

`04_run_task2_smoke.sh` 当前默认就是指向这个 checkpoint。

## 5. 我这次会话没法继续自动完成的原因

注意：**不是代码问题**。

当前这个 Codex 会话的限制有两层：

1. 原生 Slurm 客户端在本会话里报：

```text
Invalid user for SlurmUser hpcadmin, ignored
fatal: Unable to process configuration file
```

2. 学校另一套 `jsub/jqueues` 客户端也无法在本会话里建立到调度服务的连接

但用户后来自己验证到一条很关键的信息：

- 在他自己的正常终端里
- 先执行 `unclash`
- 再 `sbatch`
- 作业是可以正常提交并运行的

所以后续高权限 Codex 应该直接在**用户自己的正常终端环境**里继续，不要再纠结当前这个受限会话的提交失败。

## 6. 当前 git 状态

最后一次 `git status --short` 结果如下：

```text
 M ../README-CN.md
 M ../README.md
 M README-CN.md
 M README.md
 M requirements.txt
 M slurm_scripts/02_gen_blur_features.sh
 M slurm_scripts/03_train_eeg.sh
 M "同学运行说明.md"
?? plan/Notes_for_Attention.md
?? plan/Task2_Retrieval_Augmented_Reconstruction_Plan_EN.md
?? plan/Task2_Retrieval_Augmented_Reconstruction_Plan_ZH.md
?? plan/finish_task2_in_version5.md
?? run_task1_direct.sh
?? run_task2_direct.sh
?? scripts/evaluate_task2_reconstruction.py
?? scripts/generate_task2_reconstructions.py
?? scripts/make_task2_qualitative_grid.py
?? scripts/run_task2_pipeline.py
?? scripts/task2_common.py
?? scripts/train_task2_semantic.py
?? slurm_scripts/04_run_task2_smoke.sh
?? slurm_scripts/05_run_task2_full.sh
```

说明：

- `plan/Notes_for_Attention.md` 和 `plan/finish_task2_in_version5.md` 是用户已有笔记，不要删。
- 当前还没有实际提交 smoke/full job 的运行结果文件。
- 代码和文档改动都还在工作区里，尚未统一 commit。

## 7. 高权限 Codex 建议的下一步顺序

### 第一步：在正常终端里确认提交环境

在用户自己的 shell 里执行：

```bash
unclash
sinfo
squeue -u ckwong627
```

如果这些能正常返回，再继续。

### 第二步：先投 smoke

建议先跑：

```bash
cd /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version5_VED
unclash
sbatch slurm_scripts/04_run_task2_smoke.sh
```

然后监控：

```bash
squeue -u ckwong627
tail -f slurm_scripts/logs/04_run_task2_smoke_<jobid>.out
tail -f slurm_scripts/logs/04_run_task2_smoke_<jobid>.err
```

### 第三步：如果 `debug` 等太久，换分区

用户明确要求：

- 如果一个分区等太久，可以换其他分区
- 优先考虑 `emergency_gpua40` / `emergency_gpu`
- 不要死等

所以高权限 Codex 应该：

1. 先查当前各 GPU 分区队列情况
2. 如果 `debug` 长时间不动，就改投更快的可用分区
3. 只要资源足够完成当前作业即可

### 第四步：smoke 成功后投 full

```bash
unclash
sbatch slurm_scripts/05_run_task2_full.sh
```

必要时根据实际排队情况修改 `05_run_task2_full.sh` 的分区。

用户明确要求：

- 如果一个分区等太久，可以换其他分区
- 优先考虑 `emergency_gpua40` / `emergency_gpu`
- 不要死等

### 第五步：等待 full 完成并检查输出

重点检查：

- `output/task2/pipeline_runs/...`
- `output/task2/.../task2_reconstruction_metrics.csv`
- `output/task2/.../task2_reconstruction_summary.json`
- 生成图像和 ground truth 是否齐全
- `retrieval_metadata.json` 是否完整

### 第六步：如果跑完了，再补最终 README 分数

当前 README 里 task 2 分数还没有填最终值，因为我没有在本会话里跑完 full job。

高权限 Codex 如果拿到最终结果，应继续：

- 更新 `version5_VED/README.md`
- 更新 `version5_VED/README-CN.md`
- 必要时同步更新根目录 README

## 8. 如果不走 sbatch 的替代路径

如果要在你同学那台 A800 机器上直接运行，不用 `sbatch`，已经可以直接用：

```bash
bash run_task1_direct.sh
bash run_task2_direct.sh
```

其中：

- `run_task2_direct.sh` 会自动找 `output/logs/main_eeg_course/` 下最新的 `*_select.pth`
- 也支持手动指定：

```bash
TASK1_CKPT=/abs/path/to/checkpoint.pth bash run_task2_direct.sh
```

## 9. 对下一个 Codex 的明确提醒

1. 不要重做 task 2 代码主体，已经落地完了。
2. 先利用现有脚本和现有 checkpoint 跑 smoke，再跑 full。
3. 提交前先 `unclash`，或者至少确认代理被清掉。
4. 如果一个分区等太久，按用户要求主动切换更快分区。
5. 当前最缺的是**真实运行结果**，不是新代码。
6. 若运行中出现 import 或模型路径问题，优先小修，不要大改方案。

