#!/usr/bin/env python3
"""Print the resolved data_root path for use in SLURM shell scripts.

Resolution order:
  1. paths.data_root in task2/configs/local.yaml
  2. paths.eeg_data_dir in task2/configs/local.yaml  (appends converted_for_cogcappro/)
  3. COGCAPPRO_DATA_ROOT environment variable
  4. Auto-detect: DL_Project/image-eeg-data/converted_for_cogcappro/

Exit 1 if none of the above resolves to an existing directory.
"""
import os
import sys
from pathlib import Path

TASK2_DIR = Path(__file__).resolve().parents[1]
DL_PROJECT = TASK2_DIR.parent

data_root = None

local_yaml = TASK2_DIR / "configs" / "local.yaml"
if local_yaml.exists():
    try:
        import yaml
        c = yaml.safe_load(local_yaml.read_text()) or {}
        paths = c.get("paths") or {}
        data_root = paths.get("data_root") or None
        if not data_root and paths.get("eeg_data_dir"):
            candidate = Path(paths["eeg_data_dir"]) / "converted_for_cogcappro"
            if candidate.exists():
                data_root = str(candidate)
    except Exception:
        pass

if not data_root:
    data_root = os.environ.get("COGCAPPRO_DATA_ROOT") or None

if not data_root:
    auto = DL_PROJECT / "image-eeg-data" / "converted_for_cogcappro"
    if auto.exists():
        data_root = str(auto)

if not data_root:
    print(
        "ERROR: Cannot resolve data_root.\n"
        "  Place image-eeg-data/ in the DL_Project root, or\n"
        "  set paths.eeg_data_dir in task2/configs/local.yaml.",
        file=sys.stderr,
    )
    sys.exit(1)

print(data_root)
