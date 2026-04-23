"""
Summarize training results from main_eeg_course.py runs.
Reads all_metrics.csv from the most recent (or specified) log dir and reports
Top-1 / Top-5 accuracy in course format (mean ± std over seeds).
"""
import pandas as pd
import numpy as np
import os
import argparse

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(REPO_DIR, "output", "logs", "main_eeg_course")

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default=None,
                    help='Specific log dir; defaults to most recent run')
args = parser.parse_args()

if args.log_dir:
    run_dir = args.log_dir
else:
    # Find most recent run across all net_name subdirs
    candidates = []
    for net in os.listdir(LOGS_DIR):
        net_path = os.path.join(LOGS_DIR, net)
        if os.path.isdir(net_path):
            for run in os.listdir(net_path):
                candidates.append(os.path.join(net_path, run))
    candidates.sort()
    run_dir = candidates[-1]

csv_path = os.path.join(run_dir, "all_metrics.csv")
print(f"Reading: {csv_path}")
df = pd.read_csv(csv_path)
print(df[['seed', 'test_top1_acc', 'test_top3_acc', 'test_top5_acc',
          'best_test_top1_acc', 'best_test_top5_acc']].to_string(index=False))

print("\n=== Course Evaluation Metrics (val-selected model) ===")
print(f"Top-1 Accuracy: {df['test_top1_acc'].mean():.4f} ± {df['test_top1_acc'].std():.4f}")
print(f"Top-5 Accuracy: {df['test_top5_acc'].mean():.4f} ± {df['test_top5_acc'].std():.4f}")
print(f"\n=== Best Test Model ===")
print(f"Top-1 Accuracy: {df['best_test_top1_acc'].mean():.4f} ± {df['best_test_top1_acc'].std():.4f}")
print(f"Top-5 Accuracy: {df['best_test_top5_acc'].mean():.4f} ± {df['best_test_top5_acc'].std():.4f}")
