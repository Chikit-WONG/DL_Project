#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from safe_bpmgd.data.dataset import load_test_eeg
from safe_bpmgd.eval.eval_task2 import evaluate_reconstruction
from safe_bpmgd.eval.qualitative_grid import make_grid
from safe_bpmgd.utils.config import load_config
from safe_bpmgd.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/safe_bpmgd.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(cfg.paths.outputs) / args.run_name
    gt_dir = ensure_dir(run_dir / "ground_truth")
    test_ds = load_test_eeg(cfg, avg_trials=True, include_image_paths=True)
    final_dir = run_dir / "final_recon"
    limit_samples = len(test_ds) if args.limit_samples is None else min(args.limit_samples, len(test_ds))
    for idx in range(limit_samples):
        sample = test_ds[idx]
        src = Path(sample["image_path"])
        if src.exists():
            out_name = f"sample_{idx:05d}.png"
            if not (final_dir / out_name).exists():
                continue
            with Image.open(src) as image:
                image.convert("RGB").resize((int(cfg.generation.eval_size), int(cfg.generation.eval_size)), Image.BILINEAR).save(gt_dir / out_name)
    payload = evaluate_reconstruction(gt_dir, final_dir, run_dir / "metrics.json", image_size=int(cfg.generation.eval_size))
    make_grid(gt_dir, final_dir, run_dir / "qualitative_grid.png")
    print(payload)


if __name__ == "__main__":
    main()
