# 继续复现 VisualEEGDecoding 的执行计划

## 总结

本计划接续 `Handoff_to_Codex.md`。Claude 已经创建课程适配脚本，但图像特征生成失败了，原因是 `process_image_course.py` 仍使用 `pretrained='openai'`，导致 OpenCLIP 在 GPU 节点尝试访问 `hf-mirror.com`。所需 RN50 权重已经在本地存在。

当前已确认状态：

- 课程 EEG 数据和图片目录已经符号链接到 `data/things-eeg/`。
- 本地 RN50 权重位于 `/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/open_clip_pytorch_model.bin`。
- Job `9703515` 没有生成特征文件；`data/things-eeg/Image_feature/` 为空。
- `open_clip.create_model_and_transforms('RN50', pretrained=<local_bin>)` 已经验证可以成功加载。

## 关键修改

- 修改 `preprocess/process_image_course.py`，直接加载本地 RN50 checkpoint，避免 SLURM 运行时依赖外部网络。
- 给 `slurm_scripts/02_gen_blur_features.sh` 和 `slurm_scripts/03_train_eeg.sh` 添加 `set -eo pipefail`，确保 Python 出错时 SLURM job 明确失败，同时避免 conda 激活脚本引用可选环境变量时被 `set -u` 中断。
- 保持当前基于符号链接的数据组织方式；不需要重新转换 EEG 数据，因为 `img` 路径已经匹配 VisualEEGDecoding 的特征字典 key。
- 训练时过滤到已有图像特征的 EEG 样本，因为 `train.pt` 中的图片引用多于课程实际提供的训练图片目录。
- 不再在登录节点直接运行完整 `scripts/inspect_course_data.py`，因为它会加载大 tensor 并已被 kill；需要重检查时使用轻量检查或放到 SLURM job 中运行。

## 执行步骤

1. 修补图像特征脚本和 SLURM 脚本。
2. 运行本地权重加载 smoke test：
   `conda run -n test python -c "import open_clip; open_clip.create_model_and_transforms('RN50', pretrained='<local_bin>')"`
3. 提交图像特征生成任务：
   `sbatch slurm_scripts/02_gen_blur_features.sh`
4. 任务完成后验证：
   - `data/things-eeg/Image_feature/MultiBlur_RN50_train.pt`
   - `data/things-eeg/Image_feature/MultiBlur_RN50_test.pt`
   - 12 个 blur key：`1, 3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63`
   - 抽样特征 tensor 形状为 `[1024]`
5. 特征验证通过后再提交 EEG 训练：
   `sbatch slurm_scripts/03_train_eeg.sh`
6. 如果完整训练还在排队，先运行 `--epoch 1 --n_seeds 1` 的 debug smoke job，确认训练集会过滤缺失 feature 的样本，测试集不丢样本。
7. 监控训练日志，确认 seeds `21-30` 全部完成。
8. 汇总课程指标：
   `conda run -n test python scripts/evaluate_course_metrics.py`

## 测试计划

- 确认特征生成日志中没有 OpenCLIP 下载尝试或 proxy 错误。
- 确认特征文件可以通过 `torch.load(..., weights_only=False)` 加载。
- 确认 train/test 特征字典都有 12 个 blur level，并且 image-path key 可用。
- 在正式依赖排队中的完整 10-seed 训练前，先运行 `--epoch 1 --n_seeds 1` 的 smoke training。
- 确认 `all_metrics.csv` 有 10 行，并包含 `test_top1_acc` 和 `test_top5_acc`。

## 假设

- 使用现有 `test` conda 环境。
- 只使用 `sub-01`，进行 intra-subject 训练。
- 除非运行时间或显存失败，否则保持论文超参数。
- 如果 `debug` 分区在图像特征生成时超时，则将 `02_gen_blur_features.sh` 改到 `i64m1tga40u` 分区后重跑。
- 完整训练使用 `long_gpu` 分区，因为原来的 `i64m1tga40u` 队列预计启动更晚，而 `long_gpu` 有足够的 A800 资源且预计等待更短。
