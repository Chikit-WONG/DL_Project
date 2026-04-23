"""
Summarize training results from main_eeg_course.py runs.
Reads all_metrics.csv from the most recent (or specified) log dir and reports
Top-1 / Top-5 accuracy in course format (mean ± std over seeds).
"""
import pandas as pd
import numpy as np
import os
import argparse
import json
import re

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(REPO_DIR, "output", "logs", "main_eeg_course")
OUTPUT_DIR = os.path.join(REPO_DIR, "output")

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default=None,
                    help='Specific log dir; defaults to most recent run')
parser.add_argument('--json_path', type=str, default=os.path.join(OUTPUT_DIR, "course_metrics_summary.json"),
                    help='Where to write a compact JSON metric summary')
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

if "val_way" not in df.columns or "test_way" not in df.columns:
    dataset_sizes = None
    for name in sorted(os.listdir(run_dir)):
        if not name.endswith(".log"):
            continue
        with open(os.path.join(run_dir, name), "r", encoding="utf-8") as f:
            for line in f:
                match = re.search(r"Dataset sizes: train=(\d+), val=(\d+), test=(\d+)", line)
                if match:
                    dataset_sizes = tuple(int(x) for x in match.groups())
                    break
        if dataset_sizes:
            break
    if dataset_sizes:
        train_size, val_way, test_way = dataset_sizes
        if "train_size" not in df.columns:
            df["train_size"] = train_size
        if "val_way" not in df.columns:
            df["val_way"] = val_way
        if "test_way" not in df.columns:
            df["test_way"] = test_way
display_columns = [
    'seed',
    'val_way',
    'test_way',
    'test_top1_acc',
    'test_top3_acc',
    'test_top5_acc',
    'best_test_top1_acc',
    'best_test_top5_acc',
]
display_columns = [col for col in display_columns if col in df.columns]
print(df[display_columns].to_string(index=False))

def summarize(top1_col, top5_col):
    return {
        "top1_mean": float(df[top1_col].mean()),
        "top1_std": float(df[top1_col].std()),
        "top5_mean": float(df[top5_col].mean()),
        "top5_std": float(df[top5_col].std()),
    }

per_seed_columns = [
    "seed",
    "val_way",
    "test_way",
    "test_top1_acc",
    "test_top3_acc",
    "test_top5_acc",
    "best_test_top1_acc",
    "best_test_top3_acc",
    "best_test_top5_acc",
]
per_seed_columns = [col for col in per_seed_columns if col in df.columns]

summary = {
    "run_dir": run_dir,
    "all_metrics_csv": csv_path,
    "num_seeds": int(len(df)),
    "seeds": [int(x) for x in df["seed"].tolist()],
    "course_metrics_val_selected": summarize("test_top1_acc", "test_top5_acc"),
    "best_test_metrics_reference_only": summarize("best_test_top1_acc", "best_test_top5_acc"),
    "per_seed": df[per_seed_columns].to_dict(orient="records"),
}

if "val_way" in df.columns:
    summary["validation_way"] = sorted({int(x) for x in df["val_way"].dropna().tolist()})
if "test_way" in df.columns:
    summary["test_way"] = sorted({int(x) for x in df["test_way"].dropna().tolist()})

if "validation_way" in summary:
    print(f"\nValidation retrieval candidate pool: {summary['validation_way']}-way")
if "test_way" in summary:
    print(f"Test retrieval candidate pool: {summary['test_way']}-way")

print("\n=== Course Evaluation Metrics (val-selected model) ===")
print(
    f"Top-1 Accuracy: {summary['course_metrics_val_selected']['top1_mean']:.4f} "
    f"± {summary['course_metrics_val_selected']['top1_std']:.4f}"
)
print(
    f"Top-5 Accuracy: {summary['course_metrics_val_selected']['top5_mean']:.4f} "
    f"± {summary['course_metrics_val_selected']['top5_std']:.4f}"
)
print(f"\n=== Best Test Model ===")
print(
    f"Top-1 Accuracy: {summary['best_test_metrics_reference_only']['top1_mean']:.4f} "
    f"± {summary['best_test_metrics_reference_only']['top1_std']:.4f}"
)
print(
    f"Top-5 Accuracy: {summary['best_test_metrics_reference_only']['top5_mean']:.4f} "
    f"± {summary['best_test_metrics_reference_only']['top5_std']:.4f}"
)

os.makedirs(os.path.dirname(args.json_path), exist_ok=True)
with open(args.json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\nMetric summary JSON saved to: {args.json_path}")
