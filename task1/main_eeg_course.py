"""
Train Brain_Visual_Encoder_EEG on course data (single subject, intra-subject).
Modified to support EVNet preprocessed features with learnable fusion weights.
Version 7 additions:
  --blur_config: choose '8' or '12' blur level preset
  --use_full_train: skip 95/5 split; train on full original training set
  --eeg_data_dir: direct path to the directory containing train.pt and test.pt
  --run_name: explicit leaf directory name for split-per-seed job scheduling
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os, sys, time, json, random, logging, argparse, copy
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)
import models
import tqdm
import torch.nn.functional as F
import pandas as pd
import scipy.signal as signal
from scipy.optimize import linear_sum_assignment

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

# Blur level presets (matching reference implementation exactly)
BLUR_PRESETS = {
    '12': ['l_1', 'l_3', 'l_9', 'l_15', 'l_21', 'l_27', 'l_33', 'l_39', 'l_45', 'l_51', 'l_57', 'l_63'],
    '8':  ['l_1', 'l_3', 'l_15', 'l_21', 'l_33', 'l_45', 'l_57', 'l_63'],
    '1':  ['l_1'],  # no blur (kernel=1, identity transform)
}

_DL_PROJECT_ROOT = os.path.dirname(root_dir)
_AUTO_EEG_DATA_DIR = os.path.join(_DL_PROJECT_ROOT, "image-eeg-data")
DEFAULT_EEG_DATA_DIR = _AUTO_EEG_DATA_DIR if os.path.isdir(_AUTO_EEG_DATA_DIR) else None


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return signal.lfilter(b, a, data)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_logger(logger_file_path):
    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(logger_file_path, log_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(final_log_file, encoding="utf-8")
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s ")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


class Pair_dataset(Dataset):
    """支持多尺度模糊特征 + EVNet 特征的数据集"""
    def __init__(self, eeg_data, blur_features, evnet_features, index_dict,
                 use_filter=False, low_freq=0.1, high_freq=50.0, select_period=[0, 250],
                 use_evnet=True):
        super().__init__()
        self.blur_features = blur_features
        self.evnet_features = evnet_features if use_evnet else None
        self.eeg_data = eeg_data
        self.index_dict = index_dict
        self.selected_period = select_period
        self.use_evnet = use_evnet
        if use_filter:
            self.eeg_data = butter_bandpass_filter(self.eeg_data, low_freq, high_freq, fs=250, order=4)
            self.eeg_data = torch.from_numpy(self.eeg_data).float()

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, index):
        x = self.eeg_data[index][:, self.selected_period[0]:self.selected_period[1]].float()
        x_key = self.index_dict[index].replace('\\', '/')

        sample = {
            'eeg': x,
            'x_key': x_key,
        }
        for level in self.blur_features.keys():
            sample[f"l_{level}"] = self.blur_features[level][x_key].float()

        if self.use_evnet and self.evnet_features is not None:
            sample['evnet'] = self.evnet_features[x_key].float()

        return sample


def filter_by_available_features(eeg_data, index_dict, blur_features, evnet_features, split_name, require_all=False, use_evnet=True):
    """Keep only EEG samples whose image path has precomputed visual features."""
    available = set(blur_features['1'].keys())

    if use_evnet and evnet_features is not None:
        available_evnet = set(evnet_features.keys())
        available = available.intersection(available_evnet)

    normalized = np.array([str(k).replace('\\', '/') for k in index_dict], dtype=object)
    keep = np.array([i for i, key in enumerate(normalized) if key in available], dtype=np.int64)
    missing = len(normalized) - len(keep)

    if missing:
        examples = [key for key in normalized if key not in available][:5]
        msg = (
            f"{split_name}: dropped {missing}/{len(normalized)} samples without image features. "
            f"Examples: {examples}"
        )
        if require_all:
            raise KeyError(msg)
        print(msg)
    else:
        print(f"{split_name}: all {len(normalized)} samples have image features.")

    return eeg_data[torch.from_numpy(keep).long()], normalized[keep]


def get_dataset(base_path, feature_path, sub, select_channels, use_filter=False,
                low_freq=0.1, high_freq=50.0, select_period=[0, 250], use_evnet=True,
                use_full_train=False, eeg_data_dir=None, evnet_prefix='EVNet_RN50',
                blur_prefix='MultiBlur_RN50'):
    """
    Load EEG + image-feature datasets.

    eeg_data_dir: if provided, load train.pt/test.pt directly from this directory
                  (overrides base_path + Preprocessed_data/sub-XX/ lookup).
    use_full_train: if True, skip the 95/5 train/val split and return val_dataset=None.
    """
    channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
                'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                'O1', 'Oz', 'O2']
    selected_idx = [channels.index(ch) for ch in select_channels]

    # Load standard blur features
    train_blur = torch.load(os.path.join(feature_path, f'{blur_prefix}_train.pt'), weights_only=False)
    test_blur  = torch.load(os.path.join(feature_path, f'{blur_prefix}_test.pt'),  weights_only=False)

    # Load EVNet features if requested
    train_evnet = None
    test_evnet = None
    if use_evnet:
        evnet_train_path = os.path.join(feature_path, f'{evnet_prefix}_train.pt')
        evnet_test_path  = os.path.join(feature_path, f'{evnet_prefix}_test.pt')
        if os.path.exists(evnet_train_path) and os.path.exists(evnet_test_path):
            train_evnet = torch.load(evnet_train_path, weights_only=False)
            test_evnet = torch.load(evnet_test_path, weights_only=False)
            print(f"Loaded EVNet features: train={len(train_evnet)}, test={len(test_evnet)}")
        else:
            print(f"Warning: EVNet features not found at {evnet_train_path} or {evnet_test_path}. "
                  "Please run process_image_course.py with EVNet enabled first.")
            use_evnet = False

    # Resolve EEG data paths
    if eeg_data_dir is not None:
        train_path = os.path.join(eeg_data_dir, 'train.pt')
        test_path  = os.path.join(eeg_data_dir, 'test.pt')
    else:
        train_path = os.path.join(base_path, 'Preprocessed_data', 'sub-{:02}'.format(sub), 'train.pt')
        test_path  = os.path.join(base_path, 'Preprocessed_data', 'sub-{:02}'.format(sub), 'test.pt')

    # Training set
    loaded = torch.load(train_path, weights_only=False)
    eeg = loaded['eeg']
    if isinstance(eeg, np.ndarray):
        eeg = torch.from_numpy(eeg.astype(np.float32))
    else:
        eeg = eeg.float()

    eeg_data   = eeg[:, :, selected_idx, :].mean(dim=1)   # [N, C_sel, T]
    index_dict = loaded['img'][:, 0]                       # [N] first trial's path
    del eeg, loaded

    eeg_data, index_dict = filter_by_available_features(
        eeg_data, index_dict, train_blur, train_evnet, "train", require_all=False, use_evnet=use_evnet
    )

    if use_full_train:
        # Use all training data; no validation set
        train_eeg = eeg_data
        train_idx = index_dict
        val_dataset = None
    else:
        rng = np.random.permutation(eeg_data.shape[0])
        n_train = int(eeg_data.shape[0] * 0.95)
        train_eeg = eeg_data[rng[:n_train]]
        train_idx = index_dict[rng[:n_train]]
        val_eeg   = eeg_data[rng[n_train:]]
        val_idx   = index_dict[rng[n_train:]]
        val_dataset = Pair_dataset(val_eeg, train_blur, train_evnet, val_idx,
                                   use_filter, low_freq, high_freq, select_period, use_evnet)

    train_dataset = Pair_dataset(train_eeg, train_blur, train_evnet, train_idx,
                                 use_filter, low_freq, high_freq, select_period, use_evnet)

    # Test set
    test_loaded = torch.load(test_path, weights_only=False)
    test_eeg = test_loaded['eeg']
    if isinstance(test_eeg, np.ndarray):
        test_eeg = torch.from_numpy(test_eeg.astype(np.float32))
    else:
        test_eeg = test_eeg.float()
    test_eeg_data   = test_eeg[:, :, selected_idx, :].mean(dim=1)
    test_index_dict = test_loaded['img'][:, 0]
    del test_eeg, test_loaded

    test_eeg_data, test_index_dict = filter_by_available_features(
        test_eeg_data, test_index_dict, test_blur, test_evnet, "test", require_all=True, use_evnet=use_evnet
    )

    test_dataset = Pair_dataset(test_eeg_data, test_blur, test_evnet, test_index_dict,
                                use_filter, low_freq, high_freq, select_period, use_evnet)

    return train_dataset, val_dataset, test_dataset


class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_ranking_weights(self, loss_list):
        sorted_indices = torch.argsort(loss_list)
        weights = torch.zeros_like(loss_list)
        for i, idx in enumerate(sorted_indices):
            weights[idx] = 1 / (i + 1)
        return weights

    def forward(self, eeg_features, img_features, logit_scale):
        dev = eeg_features.device
        logits_per_eeg = logit_scale * eeg_features @ img_features.T
        num_logits = logits_per_eeg.shape[0]
        labels = torch.arange(num_logits, device=dev, dtype=torch.long)
        eeg_loss = F.cross_entropy(logits_per_eeg, labels, reduction='none')

        logits_per_img = logit_scale * img_features @ eeg_features.T
        image_loss = F.cross_entropy(logits_per_img, labels, reduction='none')
        return eeg_loss, image_loss


def hungarian_topk(sim_np, k=5):
    """对 N×N 相似度矩阵运行 k 轮匈牙利算法（禁止边），返回每行是否在 k 轮内被匹配正确。"""
    N = sim_np.shape[0]
    cost = -sim_np.astype(np.float64)
    correct_matched = np.zeros(N, dtype=bool)
    forbidden = np.zeros_like(cost, dtype=bool)

    for _ in range(min(k, N)):
        current_cost = np.where(forbidden, 1e10, cost)
        row_ind, col_ind = linear_sum_assignment(current_cost)
        for i in range(len(row_ind)):
            r, c = row_ind[i], col_ind[i]
            if r == c:
                correct_matched[r] = True
            forbidden[r, c] = True

    return correct_matched


def get_test_accu(model, params, test_dataloader, device, hungarian=False):
    model.eval()
    all_tfea = []
    all_embed = []
    total_samples = 0

    with torch.no_grad():
        for data in test_dataloader:
            teeg = data['eeg'].to(device)
            img_list = torch.cat([data[k][:, None].to(device) for k in params['blur_level']], 1)
            evnet_feat = None
            if params['use_evnet'] and 'evnet' in data:
                evnet_feat = data['evnet'].to(device)

            tfea = model(teeg)
            tfea = F.normalize(tfea, dim=-1)
            embed = model.get_image_feature(img_list, evnet_feat)
            embed = F.normalize(embed, dim=-1)

            all_tfea.append(tfea)
            all_embed.append(embed)
            total_samples += teeg.shape[0]

    tfea = torch.cat(all_tfea, dim=0)
    embed = torch.cat(all_embed, dim=0)
    sim = tfea @ embed.T
    sim_np = sim.cpu().numpy().astype(np.float64)

    _, indices = sim.topk(5)
    indices = indices.cpu()
    label = torch.arange(total_samples)
    top1 = (label[:, None] == indices[:, :1]).sum().item()
    top3 = (label[:, None].expand(-1, 3) == indices[:, :3]).any(1).sum().item()
    top5 = (label[:, None].expand(-1, 5) == indices).any(1).sum().item()

    result = {
        'top1': float(top1) / total_samples,
        'top3': float(top3) / total_samples,
        'top5': float(top5) / total_samples,
    }

    if hungarian:
        correct_top1 = hungarian_topk(sim_np, k=1)
        correct_top5 = hungarian_topk(sim_np, k=5)
        result['hung_top1'] = float(correct_top1.mean())
        result['hung_top5'] = float(correct_top5.mean())

    return result


def train(params, logger):
    set_seed(params['seed'])

    train_dataset, val_dataset, test_dataset = get_dataset(
        params['data_path'], params['feature_path'], params['sub'], params['select_chs'],
        params['use_filter'], params['low_freq'], params['high_freq'], params['select_period'],
        params['use_evnet'],
        use_full_train=params['use_full_train'],
        eeg_data_dir=params.get('eeg_data_dir'),
        evnet_prefix=params.get('evnet_prefix', 'EVNet_RN50'),
        blur_prefix=params.get('blur_prefix', 'MultiBlur_RN50'),
    )

    use_full_train = params['use_full_train']
    use_hungarian = params.get('hungarian', False)
    val_way  = len(val_dataset) if val_dataset is not None else 0
    test_way = len(test_dataset)

    train_size_str = len(train_dataset)
    logger.info(f"Dataset sizes: train={train_size_str}, val={val_way}, test={test_way}")
    logger.info(f"Using EVNet: {params['use_evnet']}, Full-train mode: {use_full_train}, Hungarian: {use_hungarian}")
    if val_dataset is not None:
        logger.info(
            f"Retrieval candidate pools: VAL={val_way}-way, TEST={test_way}-way."
        )
    else:
        logger.info(
            f"Full-train mode: no validation set. TEST={test_way}-way. "
            "'select' checkpoint = last epoch; 'best' checkpoint = best test epoch."
        )

    train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=val_way,          shuffle=False) if val_dataset is not None else None
    test_loader  = DataLoader(test_dataset,  batch_size=len(test_dataset), shuffle=False)

    model = models.__dict__[params['net_name']](
        len(params['select_chs']), 1024,
        params['select_period'][1] - params['select_period'][0],
        use_evnet=params['use_evnet']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])
    criterion = ClipLoss()

    best_val_accu  = 0
    best_test_accu = 0
    select_model = best_model = None
    saved_metric = {
        'train_size':    len(train_dataset),
        'val_way':       val_way,
        'test_way':      test_way,
        'use_evnet':     params['use_evnet'],
        'use_full_train': use_full_train,
        'hungarian':     use_hungarian,
    }
    loss_points = []
    accu_top1 = []
    accu_top3 = []
    accu_top5 = []
    accu_hung_top1 = []
    accu_hung_top5 = []
    fusion_weights_history = []

    for e in range(params['epoch']):
        model.train()
        all_loss = 0
        steps = 0
        for data in tqdm.tqdm(train_loader, desc=f"Epoch {e}"):
            optimizer.zero_grad()
            x = data['eeg'].to(device)

            img_list = torch.cat([data[k][:, None].to(device) for k in params['blur_level']], 1)

            evnet_feat = None
            if params['use_evnet'] and 'evnet' in data:
                evnet_feat = data['evnet'].to(device)

            if params['mixup']:
                rand_idx = np.random.permutation(x.shape[0])
                lam = np.random.beta(0.2, 0.2)
                x = lam * x + (1 - lam) * x[rand_idx]
                img_list = lam * img_list + (1 - lam) * img_list[rand_idx]
                if evnet_feat is not None:
                    evnet_feat = lam * evnet_feat + (1 - lam) * evnet_feat[rand_idx]

            eeg_f = model(x)
            img_f = model.get_image_feature(img_list, evnet_feat)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)

            eeg_loss, img_loss = criterion(eeg_f, img_f, 1)
            loss = (eeg_loss.mean() + img_loss.mean()) / 2
            loss.backward()
            optimizer.step()
            all_loss += loss.detach().cpu()
            steps += 1

        train_loss = all_loss / steps
        loss_points.append(float(train_loss))

        # Val evaluation (split mode only)
        if val_loader is not None:
            val_res = get_test_accu(model, params, val_loader, device, hungarian=use_hungarian)
        else:
            val_res = {'top1': float('nan'), 'top3': float('nan'), 'top5': float('nan')}

        test_res = get_test_accu(model, params, test_loader, device, hungarian=use_hungarian)
        accu_top1.append(test_res['top1'])
        accu_top3.append(test_res['top3'])
        accu_top5.append(test_res['top5'])
        if use_hungarian:
            accu_hung_top1.append(test_res['hung_top1'])
            accu_hung_top5.append(test_res['hung_top5'])

        if hasattr(model, 'get_fusion_weights'):
            fusion_weights_history.append({
                'epoch': e,
                'w_blur': model.get_fusion_weights()[0],
                'w_evnet': model.get_fusion_weights()[1]
            })

        # Checkpoint selection
        val_sel_key   = 'hung_top1' if use_hungarian else 'top1'
        test_sel_key  = 'hung_top1' if use_hungarian else 'top1'

        if val_loader is not None:
            # Split mode: select by validation accuracy
            if best_val_accu < val_res[val_sel_key]:
                select_model = copy.deepcopy(model.state_dict())
                best_val_accu = val_res[val_sel_key]
                saved_metric.update({
                    'test_top1_acc': test_res['top1'],
                    'test_top3_acc': test_res['top3'],
                    'test_top5_acc': test_res['top5'],
                })
                if use_hungarian:
                    saved_metric.update({
                        'test_hung_top1_acc': test_res['hung_top1'],
                        'test_hung_top5_acc': test_res['hung_top5'],
                    })
        else:
            # Full-train mode: select = last epoch (overwrite every epoch)
            select_model = copy.deepcopy(model.state_dict())
            saved_metric.update({
                'test_top1_acc': test_res['top1'],
                'test_top3_acc': test_res['top3'],
                'test_top5_acc': test_res['top5'],
            })
            if use_hungarian:
                saved_metric.update({
                    'test_hung_top1_acc': test_res['hung_top1'],
                    'test_hung_top5_acc': test_res['hung_top5'],
                })

        # Best-test checkpoint tracked in both modes
        if best_test_accu < test_res[test_sel_key]:
            best_test_accu = test_res[test_sel_key]
            best_model = copy.deepcopy(model.state_dict())
            saved_metric.update({
                'best_test_top1_acc': test_res['top1'],
                'best_test_top3_acc': test_res['top3'],
                'best_test_top5_acc': test_res['top5'],
            })
            if use_hungarian:
                saved_metric.update({
                    'best_test_hung_top1_acc': test_res['hung_top1'],
                    'best_test_hung_top5_acc': test_res['hung_top5'],
                })

        def _fmt(r, keys):
            parts = []
            for k in keys:
                if k in r:
                    parts.append(f"{k}:{r[k]:.4f}")
            return ' '.join(parts)

        val_keys = ['top1', 'top3', 'top5'] + (['hung_top1', 'hung_top5'] if use_hungarian else [])
        test_keys = ['top1', 'top3', 'top5'] + (['hung_top1', 'hung_top5'] if use_hungarian else [])
        if val_loader is not None:
            logger.info(f'VAL  ({val_way}-way) epoch:{e} loss:{train_loss:.4f} {_fmt(val_res, val_keys)}')
        else:
            logger.info(f'FULL-TRAIN epoch:{e} loss:{train_loss:.4f} (no val)')
        logger.info(f'TEST ({test_way}-way) epoch:{e} loss:{train_loss:.4f} {_fmt(test_res, test_keys)}')

        if hasattr(model, 'get_fusion_weights'):
            w_blur, w_evnet = model.get_fusion_weights()
            logger.info(f'Fusion weights: blur={w_blur:.4f}, evnet={w_evnet:.4f}')

    saved_metric['train_loss_points']    = loss_points
    saved_metric['test_top1_acc_points'] = accu_top1
    saved_metric['test_top3_acc_points'] = accu_top3
    saved_metric['test_top5_acc_points'] = accu_top5
    if use_hungarian:
        saved_metric['test_hung_top1_acc_points'] = accu_hung_top1
        saved_metric['test_hung_top5_acc_points'] = accu_hung_top5
    saved_metric['fusion_weights_history'] = fusion_weights_history

    prefix = f"{params['net_name']}_sub{params['sub']}_seed{params['seed']}"
    torch.save(select_model, os.path.join(params['save_path'], f"{prefix}_select.pth"))
    torch.save(best_model,   os.path.join(params['save_path'], f"{prefix}_best.pth"))

    return saved_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_name',         type=str,   default='Brain_Visual_Encoder_EEG')
    parser.add_argument('--epoch',            type=int,   default=200)
    parser.add_argument('--train_batch_size', type=int,   default=1024)
    parser.add_argument('--test_batch_size',  type=int,   default=200)
    parser.add_argument('--lr',               type=float, default=0.001)
    parser.add_argument('--mixup',            type=bool,  default=False)
    parser.add_argument('--use_filter',       type=bool,  default=False)
    parser.add_argument('--low_freq',         type=float, default=0.1)
    parser.add_argument('--high_freq',        type=float, default=50.0)
    parser.add_argument('--n_seeds',          type=int,   default=10)
    parser.add_argument('--first_seed',       type=int,   default=21)
    parser.add_argument('--use_evnet',        action='store_true', default=False,
                        help='Use EVNet preprocessed features')
    parser.add_argument('--evnet_prefix',     type=str,   default='EVNet_RN50',
                        help="Filename prefix for EVNet feature files (default: EVNet_RN50)")
    parser.add_argument('--blur_prefix',      type=str,   default='MultiBlur_RN50',
                        help="Filename prefix for blur feature files (default: MultiBlur_RN50)")
    parser.add_argument('--blur_config',      type=str,   default='12', choices=['1', '8', '12'],
                        help="Blur level preset: '1' (no blur), '8', or '12' levels")
    parser.add_argument('--use_full_train',   action='store_true', default=False,
                        help='Use full training set without 95/5 val split. '
                             'select checkpoint = last epoch; best checkpoint = best test epoch.')
    parser.add_argument('--data_path',        type=str,   default=os.path.join(root_dir, 'data', 'things-eeg'))
    parser.add_argument('--eeg_data_dir',     type=str,   default=DEFAULT_EEG_DATA_DIR,
                        help='Direct path to directory containing train.pt and test.pt. '
                             'If set, overrides --data_path lookup.')
    parser.add_argument('--feature_path',     type=str,   default=os.path.join(root_dir, 'output', 'Image_feature'))
    parser.add_argument('--output_dir',       type=str,   default=os.path.join(root_dir, 'output', 'logs', 'main_eeg_course'))
    parser.add_argument('--run_name',         type=str,   default=None,
                        help='Explicit run directory name under output_dir/net_name. '
                             'Use this to avoid timestamp collisions when one seed is run per Slurm task.')
    parser.add_argument('--hungarian',        action='store_true', default=False,
                        help='Use Hungarian algorithm for evaluation and model selection')
    args = parser.parse_args()

    ALL_CHANNELS = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                    'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
                    'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                    'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                    'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                    'O1', 'Oz', 'O2']

    BLUR_LEVELS = BLUR_PRESETS[args.blur_config]

    mode_tag = f"{'full_' if args.use_full_train else ''}{args.blur_config}blur_{'EVNet_' if args.use_evnet else ''}{'_hungarian' if args.hungarian else ''}"
    if args.run_name:
        run_leaf = args.run_name
    else:
        run_leaf = f"{mode_tag}{time.strftime('%Y-%m-%d-%H-%M')}"
    save_path = os.path.join(args.output_dir, args.net_name, run_leaf)
    if args.run_name and os.path.isdir(save_path) and os.listdir(save_path):
        raise FileExistsError(
            f"Run directory already exists and is non-empty: {save_path}. "
            "Choose a different --run_name to avoid mixing multiple jobs."
        )
    os.makedirs(save_path, exist_ok=True)

    params = {
        'net_name':         args.net_name,
        'epoch':            args.epoch,
        'train_batch_size': args.train_batch_size,
        'test_batch_size':  args.test_batch_size,
        'lr':               args.lr,
        'mixup':            args.mixup,
        'blur_level':       BLUR_LEVELS,
        'blur_config':      args.blur_config,
        'use_filter':       args.use_filter,
        'low_freq':         args.low_freq,
        'high_freq':        args.high_freq,
        'select_chs':       ALL_CHANNELS,
        'select_period':    [0, 250],
        'data_path':        args.data_path,
        'eeg_data_dir':     args.eeg_data_dir,
        'feature_path':     args.feature_path,
        'save_path':        save_path,
        'sub':              1,
        'use_evnet':        args.use_evnet,
        'use_full_train':   args.use_full_train,
        'evnet_prefix':     args.evnet_prefix,
        'blur_prefix':      args.blur_prefix,
        'hungarian':        args.hungarian,
    }

    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(params, f, indent=4)

    logger = create_logger(save_path)
    logger.info(f"Saving to: {save_path}")
    logger.info(f"EVNet enabled: {args.use_evnet}, prefix: {args.evnet_prefix}, blur_config: {args.blur_config} ({len(BLUR_LEVELS)} levels)")
    logger.info(f"Full-train mode: {args.use_full_train}")
    logger.info(f"EEG data dir: {args.eeg_data_dir}")

    all_metrics = []
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))

    for seed in seeds:
        logger.info(f'Training sub-01 seed={seed}')
        params['seed'] = seed
        metric = train(params, logger)
        all_metrics.append({'seed': seed, **metric})

    # Summary
    top1_sel  = [m['test_top1_acc']      for m in all_metrics]
    top5_sel  = [m['test_top5_acc']      for m in all_metrics]
    top1_best = [m['best_test_top1_acc'] for m in all_metrics]
    top5_best = [m['best_test_top5_acc'] for m in all_metrics]

    def _stat(vals):
        return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"

    mode_label = 'last-epoch' if args.use_full_train else 'val-selected'
    logger.info('='*60)
    logger.info(f'Config: blur={args.blur_config}, EVNet={args.use_evnet}, full_train={args.use_full_train}, Hungarian={args.hungarian}')
    logger.info(f'  {"":>20s}  {"val-sel":>22s}  {"best-test":>22s}')
    logger.info(f'  {"Top-1":>20s}  {_stat(top1_sel):>22s}  {_stat(top1_best):>22s}')
    logger.info(f'  {"Top-5":>20s}  {_stat(top5_sel):>22s}  {_stat(top5_best):>22s}')

    if args.hungarian:
        hung_top1_sel  = [m.get('test_hung_top1_acc', np.nan) for m in all_metrics]
        hung_top5_sel  = [m.get('test_hung_top5_acc', np.nan) for m in all_metrics]
        hung_top1_best = [m.get('best_test_hung_top1_acc', np.nan) for m in all_metrics]
        hung_top5_best = [m.get('best_test_hung_top5_acc', np.nan) for m in all_metrics]
        logger.info(f'  {"Hun-Top-1":>20s}  {_stat(hung_top1_sel):>22s}  {_stat(hung_top1_best):>22s}')
        logger.info(f'  {"Hun-Top-5":>20s}  {_stat(hung_top5_sel):>22s}  {_stat(hung_top5_best):>22s}')

    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(save_path, "all_metrics.csv"), index=False)
    logger.info(f"Metrics saved to {save_path}/all_metrics.csv")
