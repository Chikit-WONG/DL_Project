#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from safe_bpmgd.utils.io import write_json


ABLATIONS = [
    "A0_baseline_clip",
    "A1_prior_mapper",
    "A2_multiblur",
    "A3_vae_low_level",
    "A4_evnet_struct",
    "A5_edge_depth",
    "A6_train_only_prototype",
    "A7_self_rerank",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/ablation_plan.json")
    args = parser.parse_args()
    rows = [
        {
            "name": name,
            "status": "planned",
            "selection_rule": "compare on train-derived validation only; never tune on test metrics",
        }
        for name in ABLATIONS
    ]
    write_json(rows, Path(args.output))
    print(f"Wrote ablation manifest to {args.output}")


if __name__ == "__main__":
    main()
