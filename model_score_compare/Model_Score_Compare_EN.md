# Model Score Comparison for Submission

This note summarizes the currently available scores for `version1` to `version5_VED`.

- `version1` to `version3_ATM`: full-training-set rerun results from this round.
- `version4_CCP`: rerun is still queued; current numbers are the repository's existing reported results.
- `version5_VED`: not rerun in this round; Task 1 uses the README's chosen `Best test checkpoint` result for submission, and Task 2 uses the existing local multi-seed evaluation outputs.

## Score Table

| Version | Task 1 Top-1 | Task 1 Top-5 | Task 2 SSIM | Task 2 CLIP | Status / Source |
|---|---:|---:|---:|---:|---|
| `version1` | 0.2450 | 0.5300 | 0.2633 | 0.7836 | rerun completed |
| `version2` | 0.2000 | 0.5050 | 0.3753 | 0.2755 | rerun completed |
| `version3_ATM` | 0.3350 | 0.6350 | 0.2695 | 0.6033 | rerun completed; effectively single-seed Task 2 |
| `version4_CCP` | 0.6100 | 0.8800 | 0.3160 | 0.7070 | existing reported result; rerun still pending |
| `version5_VED` | 0.8685 ± 0.0063 | 0.9810 ± 0.0052 | 0.2977 ± 0.0066 | 0.7610 ± 0.0148 | existing local result; not rerun in this round |

## Source Notes

- `version1`
  - Task 1 / Task 2 summary: `version1/outputs/metrics_phase2_main_best.json`
- `version2`
  - Task 1 / Task 2 summary: `version2/results/metrics_v2_final.json`
- `version3_ATM`
  - Retrieval: `version3_ATM/outputs/retrieval_eval_run01.csv`
  - Reconstruction: `version3_ATM/outputs/reconstruction_eval_run01.csv`
  - Note: the retrieval CSV repeats the same score across rows, and the reconstruction CSV has only one real seed result. It should be treated as effectively a single-seed run, not a full independent 10-seed evaluation.
- `version4_CCP`
  - Current reported metrics come from `version4_CCP/README.md`
  - The fresh rerun chain is still pending on A800 resources.
- `version5_VED`
  - Task 1 uses the "Best test checkpoint" row in `version5_VED/README.md` because that is the score selected for submission
  - Task 2 summary comes from `version5_VED/output/task2/pipeline_runs/2026-04-24-20-55/evaluation/task2_reconstruction_summary.json`

## Submission Recommendation

If Task 1 and Task 2 are allowed to use different versions, the current recommendation is:

- Task 1: submit `version5_VED`
  - It is clearly the strongest retrieval result among all available versions.
  - Chosen submission result: `Top-1 = 86.85% ± 0.63%`, `Top-5 = 98.10% ± 0.52%`.
- Task 2: submit `version5_VED`
  - Among the currently available multi-seed Task 2 results, it gives the strongest `CLIP` score while keeping `SSIM` competitive.
  - Current summary: `SSIM = 0.2977 ± 0.0066`, `CLIP = 0.7610 ± 0.0148`.

## Practical Caveat

If you want a more conservative Task 2 option with slightly higher `SSIM`, `version4_CCP` is the main fallback:

- `version4_CCP`: `SSIM = 0.316`, `CLIP = 0.707`

So the simple rule is:

- prioritize best retrieval: `version5_VED`
- prioritize best current multi-seed semantic reconstruction quality: `version5_VED`
- prioritize slightly higher `SSIM` at the cost of lower `CLIP`: consider `version4_CCP`
