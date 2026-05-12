"""
Hungarian-algorithm based retrieval evaluation for ThingsEEG2.

This script loads pre-trained models (10 seeds) from a model directory,
computes EEG and image embeddings for the test set, builds a similarity
(score) matrix, and applies the Hungarian algorithm to ensure one-to-one
matching between EEG queries and image candidates.

Metrics reported:
  - Standard Top-1 / Top-5 (per-query ranking, with/without Hungarian constraint)
  - Hungarian Top-1: one-to-one matching accuracy
  - Hungarian Top-5: multi-round Hungarian, checking if correct image is among
    up to 5 assigned candidates

Usage:
    python scripts/hungarian_retrieval.py \
        --model_dir output/logs/main_eeg_course/Brain_Visual_Encoder_EEG/EVNet_2026-05-12-22-09 \
        [--data_path data/things-eeg] \
        [--feature_path output/Image_feature] \
        [--sub 1] \
        [--device cuda]
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import sys
import argparse
import json
import re

# Ensure we can import the project models
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import models

from scipy.optimize import linear_sum_assignment

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Dataset – mirrors Pair_dataset from main_eeg_course.py
# ---------------------------------------------------------------------------
class PairDatasetEval(Dataset):
    """Dataset for evaluation: EEG + precomputed image features."""

    def __init__(self, eeg_data, blur_features, evnet_features, index_dict,
                 select_period=(0, 250), use_evnet=True):
        super().__init__()
        self.blur_features = blur_features
        self.evnet_features = evnet_features if use_evnet else None
        self.eeg_data = eeg_data  # tensor [N, C, T]
        self.index_dict = index_dict  # numpy array of image paths
        self.selected_period = select_period
        self.use_evnet = use_evnet

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, index):
        x = self.eeg_data[index][:, self.selected_period[0]:self.selected_period[1]].float()
        x_key = self.index_dict[index].replace('\\', '/')

        sample = {'eeg': x, 'x_key': x_key}
        for level in self.blur_features.keys():
            sample[f"l_{level}"] = self.blur_features[level][x_key].float()

        if self.use_evnet and self.evnet_features is not None:
            sample['evnet'] = self.evnet_features[x_key].float()

        return sample


# ---------------------------------------------------------------------------
# Data loading (mirrors get_dataset from main_eeg_course.py)
# ---------------------------------------------------------------------------
def load_test_data(base_path, feature_path, sub, select_channels,
                   select_period=(0, 250), use_evnet=True):
    """Load test EEG data and image features."""
    ALL_CHANNELS = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                    'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
                    'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                    'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                    'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                    'O1', 'Oz', 'O2']
    selected_idx = [ALL_CHANNELS.index(ch) for ch in select_channels]

    # Load image features
    test_blur = torch.load(os.path.join(feature_path, 'MultiBlur_RN50_test.pt'),
                           weights_only=False)

    test_evnet = None
    if use_evnet:
        evnet_path = os.path.join(feature_path, 'EVNet_RN50_test.pt')
        if os.path.exists(evnet_path):
            test_evnet = torch.load(evnet_path, weights_only=False)
            print(f"Loaded EVNet features: test={len(test_evnet)}")
        else:
            print("Warning: EVNet features not found, disabling EVNet.")
            use_evnet = False

    # Load test EEG
    test_path = os.path.join(base_path, 'Preprocessed_data',
                             f'sub-{sub:02d}', 'test.pt')
    test_loaded = torch.load(test_path, weights_only=False)
    test_eeg = test_loaded['eeg']
    if isinstance(test_eeg, np.ndarray):
        test_eeg = torch.from_numpy(test_eeg.astype(np.float32))
    else:
        test_eeg = test_eeg.float()
    # [N_images, N_trials, N_channels, N_timepoints] -> average over trials
    test_eeg_data = test_eeg[:, :, selected_idx, :].mean(dim=1)
    test_index_dict = test_loaded['img'][:, 0]
    del test_eeg, test_loaded

    # Filter to available features
    available = set(test_blur['1'].keys())
    if use_evnet and test_evnet is not None:
        available &= set(test_evnet.keys())
    normalized = np.array([str(k).replace('\\', '/') for k in test_index_dict],
                          dtype=object)
    keep = np.array([i for i, key in enumerate(normalized) if key in available],
                    dtype=np.int64)
    if len(keep) < len(normalized):
        print(f"Test: dropped {len(normalized) - len(keep)}/{len(normalized)} "
              f"samples without image features.")
    test_eeg_data = test_eeg_data[torch.from_numpy(keep).long()]
    test_index_dict = normalized[keep]

    dataset = PairDatasetEval(test_eeg_data, test_blur, test_evnet,
                              test_index_dict, select_period, use_evnet)
    return dataset


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def find_model_files(model_dir):
    """Find all model checkpoint files (*.pth) in a directory."""
    pth_files = []
    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith('.pth'):
            pth_files.append(os.path.join(model_dir, fname))
    return pth_files


def load_model(model_path, net_name, num_channels, temporal_len, use_evnet):
    """Load a single model checkpoint (state_dict saved by main_eeg_course.py)."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = models.__dict__[net_name](
        num_channels, 1024, temporal_len, use_evnet=use_evnet
    ).to(device)

    if isinstance(ckpt, dict):
        # state_dict or wrapped dict
        model.load_state_dict(ckpt, strict=False)
    else:
        # Fallback: try as full model object
        try:
            model.load_state_dict(ckpt.state_dict(), strict=False)
        except AttributeError:
            raise TypeError(f"Unrecognized checkpoint format in {model_path}")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_embeddings(model, dataset, blur_levels, use_evnet, batch_size=256):
    """Extract EEG and image embeddings for the full test set."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_eeg_emb = []
    all_img_emb = []
    all_keys = []

    for data in loader:
        eeg = data['eeg'].to(device)
        img_list = torch.cat([data[k][:, None].to(device) for k in blur_levels], 1)
        evnet_feat = data['evnet'].to(device) if (use_evnet and 'evnet' in data) else None

        eeg_emb = model(eeg)
        eeg_emb = F.normalize(eeg_emb, dim=-1)

        img_emb = model.get_image_feature(img_list, evnet_feat)
        img_emb = F.normalize(img_emb, dim=-1)

        all_eeg_emb.append(eeg_emb.cpu())
        all_img_emb.append(img_emb.cpu())
        all_keys.extend(data['x_key'])

    eeg_emb = torch.cat(all_eeg_emb, dim=0)  # [N, D]
    img_emb = torch.cat(all_img_emb, dim=0)  # [N, D]
    return eeg_emb, img_emb, all_keys


# ---------------------------------------------------------------------------
# Hungarian retrieval evaluation
# ---------------------------------------------------------------------------
def compute_retrieval_metrics(similarity_matrix):
    """
    Compute retrieval metrics using Hungarian algorithm.

    Args:
        similarity_matrix: torch.Tensor of shape [N, N], where
            similarity_matrix[i, j] = cosine_sim(eeg_i, img_j).
            Higher is better.

    Returns:
        dict with top1_hungarian, top5_hungarian, top1_standard, top5_standard
    """
    N = similarity_matrix.shape[0]
    sim_np = similarity_matrix.cpu().numpy().astype(np.float64)

    # --- Standard top-K (per-query ranking, no global constraint) ---
    sorted_indices = np.argsort(-sim_np, axis=1)  # descending
    labels = np.arange(N)
    top1_std = (sorted_indices[:, 0] == labels).mean()
    top5_std = np.any(sorted_indices[:, :5] == labels[:, None], axis=1).mean()

    # --- Hungarian Top-1 ---
    # Hungarian minimizes cost, so use negative similarity as cost
    cost = -sim_np
    row_ind, col_ind = linear_sum_assignment(cost)
    hung_top1 = (col_ind == row_ind).mean()  # row_ind is 0..N-1, col_ind is assigned column

    # --- Hungarian Top-5 (multi-round) ---
    # Run Hungarian up to 5 times, each time removing previously assigned pairs
    remaining_cost = cost.copy()
    correct_matched = np.zeros(N, dtype=bool)

    for _ in range(min(5, N)):
        r, c = linear_sum_assignment(remaining_cost)
        for i in range(len(r)):
            if r[i] == c[i]:  # correct match
                correct_matched[r[i]] = True
        # Remove assigned rows/cols: set their cost to a large value
        remaining_cost[r, :] = 1e10
        remaining_cost[:, c] = 1e10

    hung_top5 = correct_matched.mean()

    return {
        'top1_standard': float(top1_std),
        'top5_standard': float(top5_std),
        'top1_hungarian': float(hung_top1),
        'top5_hungarian': float(hung_top5),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Hungarian-algorithm based retrieval evaluation'
    )
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing model checkpoints (.pth files) '
                             'and config.json from training.')
    parser.add_argument('--data_path', type=str,
                        default=os.path.join(ROOT_DIR, 'data', 'things-eeg'))
    parser.add_argument('--feature_path', type=str,
                        default=os.path.join(ROOT_DIR, 'output', 'Image_feature'))
    parser.add_argument('--sub', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed_pattern', type=str, default=None,
                        help='Regex to filter seed model files, e.g. "seed(\\d+)_best"')
    args = parser.parse_args()

    global device
    device = args.device

    # Load config from model_dir
    config_path = os.path.join(args.model_dir, 'config.json')
    if not os.path.exists(config_path):
        # Try parent directory
        config_path = os.path.join(os.path.dirname(args.model_dir), 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
        net_name = config.get('net_name', 'Brain_Visual_Encoder_EEG')
        blur_levels = config.get('blur_level', ['l_1', 'l_3', 'l_15', 'l_21',
                                                 'l_33', 'l_45', 'l_57', 'l_63'])
        select_channels = config.get('select_chs', None)
        select_period = tuple(config.get('select_period', [0, 250]))
        use_evnet = config.get('use_evnet', True)
    else:
        print(f"Warning: config.json not found, using defaults.")
        net_name = 'Brain_Visual_Encoder_EEG'
        blur_levels = ['l_1', 'l_3', 'l_15', 'l_21', 'l_33', 'l_45', 'l_57', 'l_63']
        select_channels = None
        select_period = (0, 250)
        use_evnet = True

    if select_channels is None:
        select_channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                           'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
                           'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                           'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                           'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                           'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                           'O1', 'Oz', 'O2']

    print(f"Config: net_name={net_name}, blur_levels={blur_levels}, "
          f"use_evnet={use_evnet}")

    # Find model files
    model_files = find_model_files(args.model_dir)
    if not model_files:
        print(f"No .pth files found in {args.model_dir}")
        # Try to find seed subdirectories
        for item in sorted(os.listdir(args.model_dir)):
            item_path = os.path.join(args.model_dir, item)
            if os.path.isdir(item_path):
                model_files.extend(find_model_files(item_path))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {args.model_dir}")

    # Filter by pattern if specified
    if args.seed_pattern:
        pat = re.compile(args.seed_pattern)
        model_files = [f for f in model_files if pat.search(os.path.basename(f))]
        print(f"Filtered to {len(model_files)} files matching '{args.seed_pattern}'")

    print(f"Found {len(model_files)} model files:")
    for f in model_files:
        print(f"  {os.path.basename(f)}")

    # Load test data (once)
    print("\nLoading test data...")
    dataset = load_test_data(
        args.data_path, args.feature_path, args.sub,
        select_channels, select_period, use_evnet
    )
    print(f"Test dataset size: {len(dataset)}")

    N_channels = len(select_channels)
    T_len = select_period[1] - select_period[0]

    # Extract embeddings for each seed model
    all_eeg_embs = []
    all_img_embs = []

    for i, model_path in enumerate(model_files):
        print(f"\n[{i+1}/{len(model_files)}] Loading model: {os.path.basename(model_path)}")
        model = load_model(model_path, net_name, N_channels, T_len, use_evnet)

        eeg_emb, img_emb, keys = extract_embeddings(
            model, dataset, blur_levels, use_evnet, args.batch_size
        )
        all_eeg_embs.append(eeg_emb)
        all_img_embs.append(img_emb)
        print(f"  EEG embeddings: {eeg_emb.shape}, Image embeddings: {img_emb.shape}")

    # Stack: [num_models, N, D]
    all_eeg_embs = torch.stack(all_eeg_embs, dim=0)
    all_img_embs = torch.stack(all_img_embs, dim=0)

    # --- Per-seed metrics ---
    print("\n" + "=" * 60)
    print("Per-seed Hungarian retrieval results:")
    print("=" * 60)
    per_seed_results = []
    for i in range(len(model_files)):
        sim = all_eeg_embs[i] @ all_img_embs[i].T  # [N, N]
        metrics = compute_retrieval_metrics(sim)
        metrics['seed_idx'] = i
        per_seed_results.append(metrics)
        print(f"  Seed {i+1:2d}: "
              f"Top1(std)={metrics['top1_standard']:.4f}  "
              f"Top5(std)={metrics['top5_standard']:.4f}  "
              f"Top1(Hun)={metrics['top1_hungarian']:.4f}  "
              f"Top5(Hun)={metrics['top5_hungarian']:.4f}")

    # --- Ensemble: average embeddings across seeds ---
    print("\n" + "=" * 60)
    print("Ensemble (mean embeddings across seeds):")
    print("=" * 60)
    mean_eeg = F.normalize(all_eeg_embs.mean(dim=0), dim=-1)
    mean_img = F.normalize(all_img_embs.mean(dim=0), dim=-1)
    ensemble_sim = mean_eeg @ mean_img.T
    ensemble_metrics = compute_retrieval_metrics(ensemble_sim)
    print(f"  Top1(std)={ensemble_metrics['top1_standard']:.4f}  "
          f"Top5(std)={ensemble_metrics['top5_standard']:.4f}  "
          f"Top1(Hun)={ensemble_metrics['top1_hungarian']:.4f}  "
          f"Top5(Hun)={ensemble_metrics['top5_hungarian']:.4f}")

    # --- Summary statistics ---
    print("\n" + "=" * 60)
    print("Summary (mean ± std over seeds):")
    print("=" * 60)
    keys_metric = ['top1_standard', 'top5_standard',
                   'top1_hungarian', 'top5_hungarian']
    for k in keys_metric:
        vals = [r[k] for r in per_seed_results]
        print(f"  {k:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\n  Ensemble {keys_metric[0]:20s}: {ensemble_metrics[keys_metric[0]]:.4f}")
    print(f"  Ensemble {keys_metric[1]:20s}: {ensemble_metrics[keys_metric[1]]:.4f}")
    print(f"  Ensemble {keys_metric[2]:20s}: {ensemble_metrics[keys_metric[2]]:.4f}")
    print(f"  Ensemble {keys_metric[3]:20s}: {ensemble_metrics[keys_metric[3]]:.4f}")

    # Save results
    summary = {
        'per_seed': per_seed_results,
        'ensemble': ensemble_metrics,
        'mean_over_seeds': {k: float(np.mean([r[k] for r in per_seed_results]))
                            for k in keys_metric},
        'std_over_seeds': {k: float(np.std([r[k] for r in per_seed_results]))
                           for k in keys_metric},
    }
    result_path = os.path.join(args.model_dir, 'hungarian_retrieval_results.json')
    with open(result_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {result_path}")


if __name__ == '__main__':
    main()
