# Version 2 Result Summary

Tag: `v2_final`

## Core Comparison

| Model | Top-1 | Top-5 | SSIM | CLIP |
|---|---:|---:|---:|---:|
| Version2 `v2_final` | 15.00% | 35.00% | 0.3709 | 0.2779 |
| Version1 Joint | 13.50% | 36.50% | 0.2762 | 0.7081 |

## Reference Rows

| Reference | Source | Top-1 | Top-5 | SSIM | CLIP |
|---|---|---:|---:|---:|---:|
| Version1 Joint Baseline | local | 13.5% | 36.5% | 0.276 | 0.708 |
| Version1 Retrieval-only | local | 14.5% | 34.5% | 0.198 | 0.658 |
| Version1 Reconstruction-only | local | 9.0% | 24.0% | 0.275 | 0.753 |
| Consensus Target | version2 plan target | >=20% | >=48% | >=0.310 | >=0.760 |

## Notes

- Version2 metrics in this table come from the completed `evaluate.py` run.
- Literature rows are intentionally conservative: local baselines plus consensus targets only.
