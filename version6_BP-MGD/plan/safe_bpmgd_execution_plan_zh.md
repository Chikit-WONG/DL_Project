# Safe BP-MGD 执行计划

## 概要

在 `version6_BP-MGD` 中实现 Task 2 的防泄漏 EEG 到图像重建流程。开发阶段只从训练集内部划分 validation 来选择模型和生成参数。最终上报测试集分数前，必须用完整训练集重新训练最终模型，并固定所有超参数。

## 实现内容

- 创建 package、configs、scripts、Slurm 脚本、outputs、checkpoints 和 feature_cache。
- 优先实现 `LeakageGuard`，并在特征缓存、memory bank、生成、重排和评估 wrapper 中调用。
- 只读取 `image-eeg-data/train.pt` 和 `image-eeg-data/test.pt`；强制 EEG shape 为 `[63, 250]`；测试集强制 `avg_trials=True`。
- 只从训练图片缓存 CLIP 和 multiblur 特征，并只用训练记录构建 `prototype_bank.pt`。
- 先训练 ATM-style EEG encoder 的 semantic 和 multiblur 分支，再加入结构 fallback loss，最后训练 MLP prior mapper。
- 测试集生成阶段只能使用 test EEG、训练好的 checkpoint、冻结预训练模型和 train-only prototype。
- 生成完成后才进入评估；test GT 只允许用于官方评估和 qualitative grid。

## 最终完整训练集规则

dev validation 确认最终配置后，使用 `--mode full_train` 重新执行 cache、encoder training、prior training 和 prototype bank 构建。之后才生成 test 重建图片并运行 Task 2 评估。

## 验收标准

- `outputs/<run>/leakage_report.txt` 显示生成和重排阶段没有访问 test image。
- `outputs/<run>/metrics.json` 输出 SSIM，CLIP 依赖可用时输出 CLIP score。
- `outputs/<run>/qualitative_grid.png` 包含 8-12 个 GT/reconstruction 对比样例。
