#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from safe_bpmgd.data.dataset import load_train_dataset
from safe_bpmgd.data.leakage_guard import LeakageGuard
from safe_bpmgd.features.cache_clip import encode_image_paths, encode_multiblur_paths, save_feature_payload
from safe_bpmgd.features.cache_edge_depth import cache_edge_depth_maps
from safe_bpmgd.features.cache_evnet import cache_evnet_fallback
from safe_bpmgd.features.cache_vae import cache_compact_vae_like_latents
from safe_bpmgd.features.train_memory_bank import build_train_memory_bank
from safe_bpmgd.utils.config import load_config
from safe_bpmgd.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/safe_bpmgd.yaml")
    parser.add_argument("--run-name", default="cache_train")
    parser.add_argument("--mode", choices=["dev", "full_train"], default="dev")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--skip-structural", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    guard = LeakageGuard(cfg, args.run_name)
    guard.register_split_paths(Path(cfg.paths.data_root) / "train.pt", "train-internal-validation", Path(cfg.paths.data_root) / "test.pt")
    records = load_train_dataset(cfg).records
    if args.limit is not None:
        records = records[: args.limit]
    for rec in records:
        guard.assert_no_test_image_path(rec.image_path, "feature cache source")

    cache_dir = ensure_dir(Path(cfg.paths.feature_cache) / ("final_train" if args.mode == "full_train" else "train"))
    clip_cfg = cfg.features.clip
    if not args.skip_clip:
        image_paths = [rec.image_path for rec in records]
        clip = encode_image_paths(
            image_paths,
            clip_cfg.model_name,
            clip_cfg.pretrained,
            int(cfg.runtime.cache_batch_size),
            args.device,
        )
        save_feature_payload(cache_dir / "clip_rn50.pt", records, clip, model_name=clip_cfg.model_name)
        multiblur = encode_multiblur_paths(
            image_paths,
            clip_cfg.model_name,
            clip_cfg.pretrained,
            int(cfg.runtime.cache_batch_size),
            args.device,
        )
        save_feature_payload(cache_dir / "multiblur_clip.pt", records, multiblur, model_name=clip_cfg.model_name)
        guard.register_feature_cache(cache_dir / "clip_rn50.pt")
        guard.register_feature_cache(cache_dir / "multiblur_clip.pt")

    if not args.skip_structural:
        cache_edge_depth_maps(records, cache_dir)
        cache_compact_vae_like_latents(records, cache_dir / "vae_latents.pt", latent_dim=int(cfg.model.vae_dim))
        cache_evnet_fallback(records, cache_dir / "evnet_struct.pt", struct_dim=int(cfg.model.struct_dim))

    build_train_memory_bank(cache_dir, cache_dir / "prototype_bank.pt", guard)
    guard.write_report(Path(cfg.paths.outputs) / args.run_name)
    print(f"Cached train-only features to {cache_dir}")


if __name__ == "__main__":
    main()
