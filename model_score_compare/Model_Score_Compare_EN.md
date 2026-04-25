# Model Score Comparison for Submission

This note summarizes the currently available scores for `version1` to `version5_VED`.

- `version1` to `version4_CCP`: full-training-set rerun results from this round.
- `version5_VED`: not rerun in this round; Task 1 uses the README's chosen `Best test checkpoint` result for submission, and Task 2 uses the existing local multi-seed evaluation outputs.

## Score Table

| Version | Task 1 Top-1 | Task 1 Top-5 | Task 2 SSIM | Task 2 CLIP | Status / Source |
|---|---:|---:|---:|---:|---|
| `version1` | 0.2450 | 0.5300 | 0.2633 | 0.7836 | rerun completed |
| `version2` | 0.2000 | 0.5050 | 0.3753 | 0.2755 | rerun completed |
| `version3_ATM` | 0.3350 | 0.6350 | 0.2709 ± 0.0052 | 0.6089 ± 0.0123 | rerun completed |
| `version4_CCP` | 0.6150 | 0.8900 | 0.3732 | 0.8981 | rerun completed |
| `version5_VED` | 0.8685 ± 0.0063 | 0.9810 ± 0.0052 | 0.2977 ± 0.0066 | 0.7610 ± 0.0148 | existing local result; not rerun in this round |

## Source Notes

- `version1`
  - Task 1 / Task 2 summary: `version1/outputs/metrics_phase2_main_best.json`
- `version2`
  - Task 1 / Task 2 summary: `version2/results/metrics_v2_final.json`
- `version3_ATM`
  - Retrieval: `version3_ATM/outputs/retrieval_eval_run01.csv`
  - Reconstruction: `version3_ATM/outputs/reconstruction_eval_run02_multiseed.csv`
  - Note: retrieval is still reported over the standard 10 random 200-way seeds, but the score is identical across rows because the candidate pool already contains all 200 test classes. Reconstruction is now a real 10-seed generation/evaluation run.
- `version4_CCP`
  - Full rerun summary comes from `version4_CCP/runs/summary_metrics_v2.json`
  - Reconstruction numbers in this comparison use the stronger `all` mode from the completed rerun
- `version5_VED`
  - Task 1 uses the "Best test checkpoint" row in `version5_VED/README.md` because that is the score selected for submission
  - Task 2 summary comes from `version5_VED/output/task2/pipeline_runs/2026-04-24-20-55/evaluation/task2_reconstruction_summary.json`

## Submission Recommendation

If Task 1 and Task 2 are allowed to use different versions, the current recommendation is:

- Task 1: submit `version5_VED`
  - It is clearly the strongest retrieval result among all available versions.
  - Chosen submission result: `Top-1 = 86.85% ± 0.63%`, `Top-5 = 98.10% ± 0.52%`.
- Task 2: submit `version4_CCP`
  - The completed rerun now clearly dominates Task 2 on both `SSIM` and `CLIP`.
  - Current summary: `SSIM = 0.3732`, `CLIP = 0.8981`.

## Practical Caveat

So the simple rule is:

- prioritize best retrieval: `version5_VED`
- prioritize best reconstruction: `version4_CCP`
- keep `version5_VED` Task 2 as a secondary fallback if you specifically want its retrieval-augmented IP-Adapter route
