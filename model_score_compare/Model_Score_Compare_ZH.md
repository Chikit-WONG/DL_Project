# 用于提交的模型得分对比

这份文档汇总了目前 `version1` 到 `version5_VED` 可用的模型分数。

- `version1` 到 `version4_CCP`：本轮“完整训练集重跑”后的结果。
- `version5_VED`：本轮没有重跑；Task 1 使用 README 中你指定用于提交的 `Best test checkpoint` 结果，Task 2 使用本地已经存在的多 seed 评估输出。

## 分数总表

| Version | Task 1 Top-1 | Task 1 Top-5 | Task 2 SSIM | Task 2 CLIP | 状态 / 来源 |
|---|---:|---:|---:|---:|---|
| `version1` | 0.2450 | 0.5300 | 0.2633 | 0.7836 | 已完成重跑 |
| `version2` | 0.2000 | 0.5050 | 0.3753 | 0.2755 | 已完成重跑 |
| `version3_ATM` | 0.3350 | 0.6350 | 0.2709 ± 0.0052 | 0.6089 ± 0.0123 | 已完成重跑 |
| `version4_CCP` | 0.6150 | 0.8900 | 0.3732 | 0.8981 | 已完成重跑 |
| `version5_VED` | 0.8685 ± 0.0063 | 0.9810 ± 0.0052 | 0.2977 ± 0.0066 | 0.7610 ± 0.0148 | 本地已有结果；本轮未重跑 |

## 来源说明

- `version1`
  - Task 1 / Task 2 汇总文件：`version1/outputs/metrics_phase2_main_best.json`
- `version2`
  - Task 1 / Task 2 汇总文件：`version2/results/metrics_v2_final.json`
- `version3_ATM`
  - retrieval：`version3_ATM/outputs/retrieval_eval_run01.csv`
  - reconstruction：`version3_ATM/outputs/reconstruction_eval_run02_multiseed.csv`
  - 说明：retrieval 仍按标准 10 个随机 200-way seed 输出，但因为候选集本来就是全部 200 个测试类别，所以每一行结果都相同；reconstruction 现在已经是真正的 10-seed 生成与评估结果。
- `version4_CCP`
  - 完整重跑汇总来自 `version4_CCP/runs/summary_metrics_v2.json`
  - 这个对比表中的 reconstruction 使用的是本次重跑后更强的 `all` 模式
- `version5_VED`
  - Task 1 使用 `version5_VED/README.md` 中 “Best test checkpoint” 这一行作为提交结果
  - Task 2 使用 `version5_VED/output/task2/pipeline_runs/2026-04-24-20-55/evaluation/task2_reconstruction_summary.json`

## 提交建议

如果课程允许 Task 1 和 Task 2 分开选不同版本，目前建议是：

- Task 1：提交 `version5_VED`
  - 这是当前所有版本里最强的 retrieval 结果。
  - 你当前选定的提交结果为：`Top-1 = 86.85% ± 0.63%`，`Top-5 = 98.10% ± 0.52%`。
- Task 2：提交 `version4_CCP`
  - 这次完整重跑后，它在 `SSIM` 和 `CLIP` 上都明显领先。
  - 当前汇总为：`SSIM = 0.3732`，`CLIP = 0.8981`。

## 实际上的补充判断

如果你更保守，想优先要稍高一点的 `SSIM`，那 Task 2 的主要备选是 `version4_CCP`：

- `version4_CCP`：`SSIM = 0.316`，`CLIP = 0.707`

所以可以简单理解成：

- 想要最强 retrieval：选 `version5_VED`
- 想要当前最强的重建指标：选 `version4_CCP`
- 如果你特别想保留 retrieval-augmented IP-Adapter 这条路线，Task 2 的次选才是 `version5_VED`
