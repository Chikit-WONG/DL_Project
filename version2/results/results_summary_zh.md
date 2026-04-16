# Version 2 结果汇总

标签：`v2_final`

## 核心对比

| 模型 | Top-1 | Top-5 | SSIM | CLIP |
|---|---:|---:|---:|---:|
| Version2 `v2_final` | 15.00% | 35.00% | 0.3709 | 0.2779 |
| Version1 Joint | 13.50% | 36.50% | 0.2762 | 0.7081 |

## 参考行

| 参考方法 | 来源 | Top-1 | Top-5 | SSIM | CLIP |
|---|---|---:|---:|---:|---:|
| Version1 Joint Baseline | local | 13.5% | 36.5% | 0.276 | 0.708 |
| Version1 Retrieval-only | local | 14.5% | 34.5% | 0.198 | 0.658 |
| Version1 Reconstruction-only | local | 9.0% | 24.0% | 0.275 | 0.753 |
| Consensus Target | version2 plan target | >=20% | >=48% | >=0.310 | >=0.760 |

## 说明

- 当前表格中的 Version2 数值来自已经完成的 `evaluate.py` 评估。
- 参考对比表目前只放本地基线与统一计划目标，避免凭空编造论文数值。
