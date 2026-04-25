import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_DIR = Path(__file__).resolve().parents[1]


def run(cmd):
    print("\n$ " + " ".join(str(item) for item in cmd), flush=True)
    subprocess.run([str(item) for item in cmd], cwd=REPO_DIR, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="One-command task2 pipeline for version5_VED.")
    parser.add_argument("--data_root", required=True, help="Path to image-eeg-data.")
    parser.add_argument("--clip_checkpoint", required=True, help="Path to OpenCLIP RN50 checkpoint.")
    parser.add_argument("--task1_ckpt", required=True, help="Path to the task1 retrieval checkpoint.")
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--first_seed", type=int, default=21)
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--prompt_topk", type=int, default=20)
    parser.add_argument("--prompt_template", type=str, default="a realistic photo of a {class_name}")
    parser.add_argument("--sd_model_path", type=str, default="/hpc2hdd/home/ckwong627/workdir/models/stable-diffusion-v1-5")
    parser.add_argument("--ip_adapter_root", type=str, default="/hpc2hdd/home/ckwong627/workdir/models/IP-Adapter")
    parser.add_argument("--skip_finetune", action="store_true")
    parser.add_argument("--skip_generate", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_name = time.strftime("%Y-%m-%d-%H-%M")
    run_root = REPO_DIR / "output" / "task2" / "pipeline_runs" / run_name
    finetune_root = run_root / "semantic_finetune"
    recon_root = run_root / "reconstructions"
    eval_root = run_root / "evaluation"
    run_root.mkdir(parents=True, exist_ok=True)

    run([sys.executable, "scripts/prepare_course_data.py", "--data_root", args.data_root])

    semantic_run_name = "semantic_finetune"
    semantic_dir = finetune_root / semantic_run_name
    if not args.skip_finetune:
        run(
            [
                sys.executable,
                "scripts/train_task2_semantic.py",
                "--init_ckpt",
                args.task1_ckpt,
                "--clip_checkpoint",
                args.clip_checkpoint,
                "--epoch",
                str(args.epoch),
                "--n_seeds",
                str(args.n_seeds),
                "--first_seed",
                str(args.first_seed),
                "--train_batch_size",
                str(args.train_batch_size),
                "--lr",
                str(args.lr),
                "--prompt_topk",
                str(args.prompt_topk),
                "--prompt_template",
                args.prompt_template,
                "--output_dir",
                finetune_root,
                "--run_name",
                semantic_run_name,
            ]
        )
    if not semantic_dir.exists():
        raise FileNotFoundError(f"Task2 semantic checkpoint directory not found: {semantic_dir}")

    best_metrics_path = semantic_dir / "task2_best_metrics.csv"
    if not best_metrics_path.exists():
        raise FileNotFoundError(f"Missing {best_metrics_path}")
    best_metrics = pd.read_csv(best_metrics_path)

    per_seed_eval = []
    for _, row in best_metrics.iterrows():
        seed = int(row["seed"])
        ckpt_path = row["best_ckpt"]
        train_bank = row["best_train_bank"]
        recon_run_name = f"seed{seed:02d}"
        recon_dir = recon_root / recon_run_name

        if not args.skip_generate:
            run(
                [
                    sys.executable,
                    "scripts/generate_task2_reconstructions.py",
                    "--checkpoint",
                    ckpt_path,
                    "--train_bank",
                    train_bank,
                    "--data_path",
                    str(REPO_DIR / "data" / "things-eeg"),
                    "--feature_path",
                    str(REPO_DIR / "output" / "Image_feature"),
                    "--output_dir",
                    recon_root,
                    "--run_name",
                    recon_run_name,
                    "--prompt_template",
                    args.prompt_template,
                    "--top_k",
                    str(args.prompt_topk),
                    "--generation_seed",
                    str(seed),
                    "--sd_model_path",
                    args.sd_model_path,
                    "--ip_adapter_root",
                    args.ip_adapter_root,
                ]
            )

        if not args.skip_eval:
            run(
                [
                    sys.executable,
                    "scripts/evaluate_task2_reconstruction.py",
                    "--real-root",
                    recon_dir / "ground_truth",
                    "--fake-root",
                    recon_dir / "generated",
                    "--output",
                    eval_root / f"seed{seed:02d}.json",
                ]
            )
            with (eval_root / f"seed{seed:02d}.json").open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            metrics["seed"] = seed
            per_seed_eval.append(metrics)

    if per_seed_eval:
        df = pd.DataFrame(per_seed_eval)
        df.to_csv(eval_root / "task2_reconstruction_metrics.csv", index=False)
        summary = {
            "eval_ssim_mean": float(np.mean(df["eval_ssim"])),
            "eval_ssim_std": float(np.std(df["eval_ssim"])),
            "eval_clip_mean": float(np.mean(df["eval_clip"])),
            "eval_clip_std": float(np.std(df["eval_clip"])),
            "n_seeds": int(len(df)),
            "run_root": str(run_root),
        }
        with (eval_root / "task2_reconstruction_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
