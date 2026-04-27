#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from safe_bpmgd.data.dataset import load_test_eeg
from safe_bpmgd.data.leakage_guard import LeakageGuard
from safe_bpmgd.encoders.eeg_cogcap import SafeBPMGDEEGModel
from safe_bpmgd.generation.generate_candidates import NEUTRAL_PROMPT, generate_prototype_candidates, generate_sdxl_ipadapter_candidates
from safe_bpmgd.generation.rerank import prototype_candidate_features, retrieve_topk, score_candidates
from safe_bpmgd.generation.sdxl_ipadapter import build_sdxl_ipadapter_pipeline
from safe_bpmgd.prior.prior_diffusion import MLPPriorMapper
from safe_bpmgd.utils.config import load_config
from safe_bpmgd.utils.io import ensure_dir, write_json
from safe_bpmgd.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/safe_bpmgd.yaml")
    parser.add_argument("--run-name", default="final_generate")
    parser.add_argument("--encoder-ckpt", required=True)
    parser.add_argument("--prior-ckpt", default=None)
    parser.add_argument("--mode", choices=["dev", "full_train"], default="full_train")
    parser.add_argument("--backend", choices=["prototype", "sdxl_ipadapter"], default="prototype")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))
    guard = LeakageGuard(cfg, args.run_name)
    guard.register_split_paths(Path(cfg.paths.data_root) / "train.pt", "fixed-config-no-validation-at-test-time", Path(cfg.paths.data_root) / "test.pt")
    guard.assert_prompt_is_neutral(cfg.generation.prompt)
    device = torch.device(args.device)
    run_dir = ensure_dir(Path(cfg.paths.outputs) / args.run_name)
    candidates_root = ensure_dir(run_dir / "candidates")
    final_root = ensure_dir(run_dir / "final_recon")
    cache_dir = Path(cfg.paths.feature_cache) / ("final_train" if args.mode == "full_train" else "train")
    bank = torch.load(cache_dir / "prototype_bank.pt", map_location="cpu", weights_only=False)
    guard.assert_train_memory_bank_paths(bank["image_paths"])

    model = SafeBPMGDEEGModel(cfg).to(device)
    ckpt = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval().requires_grad_(False)
    prior = None
    if args.prior_ckpt:
        prior = MLPPriorMapper(dim=int(cfg.model.semantic_dim)).to(device)
        prior.load_state_dict(torch.load(args.prior_ckpt, map_location=device, weights_only=False)["prior"])
        prior.eval().requires_grad_(False)

    pipe = None
    if args.backend == "sdxl_ipadapter":
        pipe = build_sdxl_ipadapter_pipeline(cfg, str(device))

    test_ds = load_test_eeg(cfg, avg_trials=True, include_image_paths=False)
    limit_samples = len(test_ds) if args.limit_samples is None else min(args.limit_samples, len(test_ds))
    top_k = args.top_k or int(cfg.generation.top_k)
    num_candidates = args.num_candidates or int(cfg.generation.num_candidates_per_eeg)
    bank_device = {key: value.to(device) if torch.is_tensor(value) else value for key, value in bank.items()}
    metadata = []
    with torch.no_grad():
        for i in range(limit_samples):
            sample = test_ds[i]
            eeg = sample["eeg"].unsqueeze(0).to(device)
            out = model(eeg)
            z_sem = prior(out["z_sem"]) if prior is not None else out["z_sem"]
            values, indices = retrieve_topk(z_sem[0], bank_device, top_k)
            idx_list = [int(x) for x in indices.detach().cpu().reshape(-1).tolist()]
            proto_paths = [bank["image_paths"][idx] for idx in idx_list]
            sample_name = f"sample_{i:05d}"
            cand_dir = ensure_dir(candidates_root / sample_name)
            if args.backend == "sdxl_ipadapter" and pipe is not None:
                proto_images = [Image.open(path).convert("RGB").resize((512, 512)) for path in proto_paths[:4]]
                cand_paths = generate_sdxl_ipadapter_candidates(
                    pipe,
                    proto_images,
                    cand_dir,
                    sample_name,
                    cfg,
                    str(device),
                    num_candidates=num_candidates,
                )
            else:
                cand_paths = generate_prototype_candidates(
                    proto_paths,
                    cand_dir,
                    sample_name,
                    num_candidates,
                    int(cfg.generation.eval_size),
                )
            predicted = {"clip": F.normalize(z_sem[0].detach().cpu(), dim=-1)}
            cand_feats = prototype_candidate_features(bank, idx_list[: len(cand_paths)])
            ranked = score_candidates(cand_feats, predicted)
            best = ranked[0]["candidate_index"]
            final_name = f"{sample_name}.png"
            Image.open(cand_paths[best]).convert("RGB").save(final_root / final_name)
            metadata.append(
                {
                    "sample_index": i,
                    "output_name": final_name,
                    "prompt": NEUTRAL_PROMPT,
                    "top_prototype_paths": proto_paths[: top_k],
                    "top_scores": values.detach().cpu().reshape(-1).tolist(),
                    "candidate_paths": cand_paths,
                    "rerank": ranked,
                }
            )
            print(f"Generated {sample_name} with {len(cand_paths)} candidates")
    write_json(metadata, run_dir / "rerank_scores.json")
    guard.write_report(run_dir)
    print(f"Saved {len(metadata)} final reconstructions to {final_root}")


if __name__ == "__main__":
    main()
