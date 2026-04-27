#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from safe_bpmgd.data.dataset import load_test_eeg, load_train_dataset
from safe_bpmgd.data.splits import make_train_val_indices, save_split_indices
from safe_bpmgd.utils.config import load_config
from safe_bpmgd.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/safe_bpmgd.yaml")
    parser.add_argument("--run-name", default="data_check")
    parser.add_argument("--mode", choices=["dev", "full_train"], default="dev")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train = load_train_dataset(cfg)
    test = load_test_eeg(cfg, avg_trials=True)
    run_dir = ensure_dir(Path(cfg.paths.outputs) / args.run_name)

    if args.mode == "dev":
        train_idx, val_idx = make_train_val_indices(len(train), float(cfg.data.val_fraction), int(cfg.seed))
    else:
        train_idx, val_idx = list(range(len(train))), []
    save_split_indices(train_idx, val_idx, run_dir / "val_indices.json")
    payload = {
        "mode": args.mode,
        "train_samples_loaded": len(train),
        "test_samples_loaded_avg_trials": len(test),
        "train_indices": len(train_idx),
        "val_indices": len(val_idx),
        "first_train_shape": list(train[0]["eeg"].shape),
        "first_test_shape": list(test[0]["eeg"].shape),
        "data_root": cfg.paths.data_root,
    }
    write_json(payload, run_dir / "data_check.json")
    print(payload)


if __name__ == "__main__":
    main()
