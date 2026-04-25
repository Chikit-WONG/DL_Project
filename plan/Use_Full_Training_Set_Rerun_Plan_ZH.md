# 使用完整训练集重跑 `version1`-`version4` 计划

## 概要

目标是基于现在已经完整上传的训练集，重新运行 `version1`、`version2`、`version3_ATM`、`version4_CCP` 的正式训练与评估流程，并按课程要求重新生成检索和重建分数。`version5_VED` 不在本次任务范围内。

本次“重跑”定义为：

- 重建可能曾基于不完整训练集生成的缓存
- 重新训练各版本既定主流程
- 重新执行重建或生成流程
- 重新执行官方或课程口径评估
- 将新分数与此前本地记录的结果做对比

## 实施改动

- 在 `plan/` 下新增本任务的中英文计划文档。
- 统一正式重跑入口，确保脚本指向正确工作目录，并使用适合全量运行的 SLURM 资源。
- 保留 smoke test 和 debug 入口，但不再把它们作为最终结果复现的默认入口。
- 为 `version3_ATM` 增加稳定的 `LATEST_RETRIEVAL` 与 `LATEST_RECONSTRUCTION` 软链接，避免后续评估还要手改时间戳路径。
- 为 `version4_CCP` 增加一个非 debug 命名的完整评估入口，一次评估 `all_before` 与 `all` 两种模式。

## 各版本执行方式

### `version1`

- 基于完整共享数据集重建 CLIP 缓存。
- 重新训练 `phase1_main` 与 `phase2_main`。
- 使用 10 个 seed 重新做图像重建。
- 重新评估并生成新的 `metrics_phase2_main_best.json`。
- SLURM 工作目录固定为 `version1`，避免 `codes/config.py` 解析到错误路径。

### `version2`

- 继续使用现有正式脚本完成 cache、warmup、multitarget、finetune、prior、reconstruct、evaluate。
- 确保正式脚本不传 `--limit`、`--limit_train`、`--limit_test`。
- 保留 `compare_v1` 汇总逻辑，使更新后的 `version1` 结果能自动纳入比较。

### `version3_ATM`

- 用完整课程数据重新训练 retrieval 与 reconstruction。
- 训练结束后分别刷新 `LATEST_RETRIEVAL` 和 `LATEST_RECONSTRUCTION` 软链接。
- 基于这两个稳定路径重新执行 retrieval 评估、图像生成和 reconstruction 指标评估。

### `version4_CCP`

- 重新执行 diffusion embedding 准备、retrieval 训练、alignment、图像生成、重建评估和 summary。
- 继续把 `all_before` 作为推荐的主结果，但仍同时生成和评估 `all_before` 与 `all`。
- 使用修复后的正式生成脚本和新的完整评估脚本，不再默认引用旧的 debug 命名入口。

## 测试与验收

- 确认 `train.pt` 与 `test.pt` 可以正常读取，且对应课程数据集。
- 优先复用 `test` conda 环境；只有在出现明确依赖问题时才切换方案。
- 若某版本已有 smoke 或环境检查脚本，先做轻量检查再提交长任务。
- 确认每个版本都能产出新的 checkpoint、评估结果和最终分数字段，且未退回到 debug/sample limit 模式。
- 最终汇总表至少包含：
  - Retrieval 的 Top-1 / Top-5
  - Reconstruction 的 SSIM / CLIP Score
  - 新结果文件路径

## 假设

- 本次任务按“重新训练并重新评估”理解，不是直接复用旧 checkpoint 重算。
- 只修改与完整训练集重跑直接相关的部分，不做模型结构升级。
- 实际提交作业时，可根据排队情况切换到更快且可用的分区。
