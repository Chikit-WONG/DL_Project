#!/usr/bin/env python3
"""
Download all required model weights to DL_Project/models/ and update local.yaml.

Models downloaded:
  1. CLIP ViT-H-14  laion/CLIP-ViT-H-14-laion2B-s32B-b79K  (~2.5 GB)
  2. CLIP RN50      laion/CLIP-RN50-openai                  (~350 MB)
  3. SDXL-Turbo     stabilityai/sdxl-turbo                  (~6 GB, fp16 safetensors only)
  4. IP-Adapter     h94/IP-Adapter  (sdxl_vit-h subset)     (~300 MB)

Usage:
  python scripts/download_models.py
  python scripts/download_models.py --dest /custom/path/to/models
  python scripts/download_models.py --hf-token hf_xxxx
  python scripts/download_models.py --skip-yaml   # do not update local.yaml

After a successful download the script rewrites weights_root in
task2/configs/local.yaml so all other scripts pick up the new path.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_YAML = PROJECT_ROOT / "task2" / "configs" / "local.yaml"

# Single-file downloads: (hf_repo, filename, local_subdir)
HF_SINGLE_FILES = [
    (
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "open_clip_pytorch_model.bin",
        "CLIP-ViT-H-14-laion2B-s32B-b79K",
    ),
    (
        "laion/CLIP-RN50-openai",
        "open_clip_pytorch_model.bin",
        "CLIP-RN50-openai",
    ),
]

# Snapshot downloads: (hf_repo, local_subdir, allow_patterns, ignore_patterns)
HF_SNAPSHOTS = [
    (
        "stabilityai/sdxl-turbo",
        "sdxl-turbo",
        None,
        # Skip non-PyTorch formats; keep all safetensors (including *.fp16.safetensors)
        ["*.msgpack", "*.ot", "*.h5", "flax_model*", "rust_model*", "tf_model*", "tf_*"],
    ),
    (
        "h94/IP-Adapter",
        "IP-Adapter",
        # Only the SDXL ViT-H adapter and its image encoder — skip everything else
        [
            "models/image_encoder/*",
            "sdxl_models/ip-adapter_sdxl_vit-h.safetensors",
        ],
        None,
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dest",
        default=None,
        help="Download root directory (default: <project_root>/models/)",
    )
    p.add_argument(
        "--hf-token",
        default=None,
        metavar="TOKEN",
        help="HuggingFace API token (optional, needed only for gated models)",
    )
    p.add_argument(
        "--skip-yaml",
        action="store_true",
        help="Do not update task2/configs/local.yaml after download",
    )
    return p.parse_args()


def _hf_download_file(
    repo_id: str,
    filename: str,
    local_dir: Path,
    token: str | None,
) -> None:
    from huggingface_hub import hf_hub_download

    dest_file = local_dir / filename
    if dest_file.exists():
        print(f"    [skip] {filename} already present")
        return
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"    Downloading {repo_id} / {filename} ...")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
        token=token,
    )
    print(f"    Saved → {dest_file}")


def _hf_snapshot(
    repo_id: str,
    local_dir: Path,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
    token: str | None,
) -> None:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)

    marker = local_dir / ".download_complete"
    if marker.exists():
        print(f"    [skip] {repo_id} already downloaded (marker present)")
        return

    print(f"    Downloading {repo_id} ...")
    kwargs: dict = dict(repo_id=repo_id, local_dir=str(local_dir), token=token)
    if allow_patterns is not None:
        kwargs["allow_patterns"] = allow_patterns
    if ignore_patterns is not None:
        kwargs["ignore_patterns"] = ignore_patterns

    snapshot_download(**kwargs)
    marker.touch()
    print(f"    Saved → {local_dir}")


def update_local_yaml(weights_root: Path) -> None:
    if not LOCAL_YAML.exists():
        print(f"  [warn] {LOCAL_YAML} not found — skipping")
        return
    content = LOCAL_YAML.read_text()
    new_content = re.sub(
        r"^(\s*weights_root:\s*).*$",
        lambda m: f"{m.group(1)}{weights_root}",
        content,
        flags=re.MULTILINE,
    )
    if new_content == content:
        print(f"  [info] local.yaml weights_root is already up to date")
        return
    LOCAL_YAML.write_text(new_content)
    print(f"  Updated local.yaml → weights_root: {weights_root}")


def main() -> None:
    args = parse_args()
    dest = Path(args.dest).resolve() if args.dest else (PROJECT_ROOT / "models")
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Download destination: {dest}\n")

    for repo_id, filename, subdir in HF_SINGLE_FILES:
        local_dir = dest / subdir
        print(f"[{subdir}]")
        try:
            _hf_download_file(repo_id, filename, local_dir, args.hf_token)
        except Exception as exc:
            print(f"  ERROR downloading {repo_id}: {exc}", file=sys.stderr)
            sys.exit(1)

    for repo_id, subdir, allow_patterns, ignore_patterns in HF_SNAPSHOTS:
        local_dir = dest / subdir
        print(f"\n[{subdir}]")
        try:
            _hf_snapshot(repo_id, local_dir, allow_patterns, ignore_patterns, args.hf_token)
        except Exception as exc:
            print(f"  ERROR downloading {repo_id}: {exc}", file=sys.stderr)
            sys.exit(1)

    if not args.skip_yaml:
        print(f"\n[Updating local.yaml]")
        update_local_yaml(dest)

    print(f"\nDone. weights_root = {dest}")
    print("Next steps:")
    print("  1. sbatch task1/slurm_scripts/00_setup_env.sh   # set up conda env (if not done)")
    print("  2. sbatch task1/slurm_scripts/01_gen_evnet_features.sh")
    print("  3. sbatch task1/slurm_scripts/04_full_train_8blur_evnet.sh")
    print("  4. Chain task2 jobs: 06 → 07 → 08 → 09 → 10  (see README.md)")


if __name__ == "__main__":
    main()
