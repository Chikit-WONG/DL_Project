"""
Task 1 Retrieval Evaluation Script
===================================
Loads similarity matrices saved by main_eeg_course.py (one per seed) and
evaluates retrieval using TA's official compute_retrieval_metrics() protocol.

Usage:
    python TA_Evaluation/task1_eval.py --run_dir task1/output/logs/main_eeg_course/Brain_Visual_Encoder_EEG/<run_name>

The script finds all *_sim_matrix.pt files under --run_dir recursively,
computes Top-1 / Top-5 accuracy per seed, and reports mean ± std over seeds.

This script does NOT modify any TA-provided code.  The two evaluation
functions below are reproduced exactly from the TA sample notebook.
"""
import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


# ============================================================================
#  TA evaluation functions (identical to the sample notebook)
# ============================================================================

def compute_retrieval_metrics(logits: torch.Tensor) -> Dict[str, float]:
    """Compute Top-1 and Top-5 retrieval accuracy.

    Parameters
    ----------
    logits:
        Similarity matrix of shape [N, N], where row i corresponds to the
        i-th EEG sample and column j corresponds to the j-th candidate image.

    Returns
    -------
    Dict[str, float]
        A dictionary with Top-1 and Top-5 accuracy.
    """
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("Expected a square similarity matrix of shape [N, N].")

    n = logits.shape[0]
    targets = torch.arange(n)

    top1_pred = logits.argmax(dim=1)
    top1_acc = (top1_pred == targets).float().mean().item()

    top5_idx = logits.topk(k=5, dim=1).indices
    top5_acc = (top5_idx == targets[:, None]).any(dim=1).float().mean().item()

    return {
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
    }


def summarize_metrics_over_seeds(metric_list: List[Dict]) -> Dict:
    """Summarize metrics across seeds as mean ± std (ddof=1)."""
    keys = metric_list[0].keys()
    summary = {}
    for key in keys:
        values = np.array([m[key] for m in metric_list], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return summary


# ============================================================================
#  File discovery and main evaluation loop
# ============================================================================

def find_sim_matrices(run_dir: str, checkpoint: str) -> List[str]:
    """Find all *_<checkpoint>_sim_matrix.pt files recursively under run_dir, sorted."""
    pattern = os.path.join(run_dir, "**", f"*_{checkpoint}_sim_matrix.pt")
    files = sorted(glob.glob(pattern, recursive=True))
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Task 1 Retrieval Evaluation (TA protocol)"
    )
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Training run directory containing *_sim_matrix.pt files "
             "(e.g. task1/output/logs/main_eeg_course/Brain_Visual_Encoder_EEG/<run_name>)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="select", choices=["select", "best"],
        help="Which checkpoint to evaluate: 'select' (val-selected or last-epoch) "
             "or 'best' (best test accuracy). Default: select."
    )
    args = parser.parse_args()

    sim_files = find_sim_matrices(args.run_dir, args.checkpoint)

    if not sim_files:
        print(f"[ERROR] No *_{args.checkpoint}_sim_matrix.pt files found under {args.run_dir}")
        print("Make sure you have run main_eeg_course.py first (it saves one matrix per seed).")
        return

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Found {len(sim_files)} similarity matrix file(s):")
    for f in sim_files:
        print(f"  {f}")

    retrieval_results = []
    for sim_file in sim_files:
        data = torch.load(sim_file, weights_only=True)
        sim = data["sim_matrix"]
        seed = data.get("seed", "unknown")

        metrics = compute_retrieval_metrics(sim)
        metrics["seed"] = seed
        retrieval_results.append(metrics)

    # Summarize
    metric_dicts = [
        {k: v for k, v in m.items() if k != "seed"}
        for m in retrieval_results
    ]
    retrieval_summary = summarize_metrics_over_seeds(metric_dicts)

    print("\n" + "=" * 60)
    print("Task 1  Retrieval Evaluation Results")
    print("=" * 60)
    print("\nPer-seed metrics:")
    for metrics in retrieval_results:
        print(
            f"  seed={str(metrics['seed']):>4s}  |  "
            f"Top-1={metrics['top1_acc']:.4f}  |  "
            f"Top-5={metrics['top5_acc']:.4f}"
        )

    print("\nSummary (mean ± std over seeds):")
    for key, stats in retrieval_summary.items():
        print(f"  {key}:  {stats['mean']:.4f} ± {stats['std']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
