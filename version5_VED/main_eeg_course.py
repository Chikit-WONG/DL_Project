"""
Train Brain_Visual_Encoder_EEG on course data (single subject, intra-subject).
Adapted from main_eeg.py: data_path set to course data, single sub-01, 10 seeds.
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)


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
    def __init__(self, eeg_data, blur_features, index_dict,
                 use_filter=False, low_freq=0.1, high_freq=50.0, select_period=[0, 250]):
        super().__init__()
        self.blur_features = blur_features
        self.eeg_data = eeg_data
        self.index_dict = index_dict
        self.selected_period = select_period
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
            'l_1':  self.blur_features['1'][x_key].float(),
            'l_3':  self.blur_features['3'][x_key].float(),
            'l_9':  self.blur_features['9'][x_key].float(),
            'l_15': self.blur_features['15'][x_key].float(),
            'l_21': self.blur_features['21'][x_key].float(),
            'l_27': self.blur_features['27'][x_key].float(),
            'l_33': self.blur_features['33'][x_key].float(),
            'l_39': self.blur_features['39'][x_key].float(),
            'l_45': self.blur_features['45'][x_key].float(),
            'l_51': self.blur_features['51'][x_key].float(),
            'l_57': self.blur_features['57'][x_key].float(),
            'l_63': self.blur_features['63'][x_key].float(),
        }
        return sample


def filter_by_available_features(eeg_data, index_dict, blur_features, split_name, require_all=False):
    """Keep only EEG samples whose image path has precomputed visual features."""
    available = set(blur_features['1'].keys())
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
                low_freq=0.1, high_freq=50.0, select_period=[0, 250]):
    channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
                'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                'O1', 'Oz', 'O2']
    selected_idx = [channels.index(ch) for ch in select_channels]

    train_blur = torch.load(os.path.join(feature_path, 'MultiBlur_RN50_train.pt'), weights_only=False)
    test_blur  = torch.load(os.path.join(feature_path, 'MultiBlur_RN50_test.pt'),  weights_only=False)

    train_path = os.path.join(base_path, 'Preprocessed_data', 'sub-{:02}'.format(sub), 'train.pt')
    loaded = torch.load(train_path, weights_only=False)
    # eeg shape: [N_images, N_trials, N_channels, N_timepoints]
    eeg = loaded['eeg']
    if isinstance(eeg, np.ndarray):
        eeg = torch.from_numpy(eeg.astype(np.float32))
    else:
        eeg = eeg.float()

    # Select channels and average over trials
    eeg_data   = eeg[:, :, selected_idx, :].mean(dim=1)   # [N, C_sel, T]
    index_dict = loaded['img'][:, 0]                       # [N] first trial's path
    del eeg, loaded

    eeg_data, index_dict = filter_by_available_features(
        eeg_data, index_dict, train_blur, "train", require_all=False
    )

    rng = np.random.permutation(eeg_data.shape[0])
    n_train = int(eeg_data.shape[0] * 0.95)
    train_eeg  = eeg_data[rng[:n_train]]
    train_idx  = index_dict[rng[:n_train]]
    val_eeg    = eeg_data[rng[n_train:]]
    val_idx    = index_dict[rng[n_train:]]

    train_dataset = Pair_dataset(train_eeg, train_blur, train_idx, use_filter, low_freq, high_freq, select_period)
    val_dataset   = Pair_dataset(val_eeg,   train_blur, val_idx,   use_filter, low_freq, high_freq, select_period)

    test_path = os.path.join(base_path, 'Preprocessed_data', 'sub-{:02}'.format(sub), 'test.pt')
    test_loaded = torch.load(test_path, weights_only=False)
    test_eeg = test_loaded['eeg']
    if isinstance(test_eeg, np.ndarray):
        test_eeg = torch.from_numpy(test_eeg.astype(np.float32))
    else:
        test_eeg = test_eeg.float()
    test_eeg_data  = test_eeg[:, :, selected_idx, :].mean(dim=1)
    test_index_dict = test_loaded['img'][:, 0]
    del test_eeg, test_loaded

    test_eeg_data, test_index_dict = filter_by_available_features(
        test_eeg_data, test_index_dict, test_blur, "test", require_all=True
    )

    test_dataset = Pair_dataset(test_eeg_data, test_blur, test_index_dict, use_filter, low_freq, high_freq, select_period)

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


def get_test_accu(model, params, test_dataloader, device):
    total = top1 = top3 = top5 = 0
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            teeg = data['eeg'].to(device)
            img_list = torch.cat([data[k][:, None].to(device) for k in params['blur_level']], 1)
            tfea = model(teeg)
            tfea = tfea / tfea.norm(dim=-1, keepdim=True)
            embed = model.get_image_feature(img_list)
            embed = embed / embed.norm(dim=-1, keepdim=True)
            sim = tfea @ embed.transpose(-1, -2)
            _, indices = sim.topk(5)
            indices = indices.cpu()
            label = torch.arange(teeg.shape[0])
            top1 += (label[:, None] == indices[:, :1]).sum()
            top3 += (label[:, None].expand(-1, 3) == indices[:, :3]).any(1).sum()
            top5 += (label[:, None].expand(-1, 5) == indices).any(1).sum()
            total += teeg.shape[0]
    return float(top1) / total, float(top3) / total, float(top5) / total


def train(params, logger):
    set_seed(params['seed'])
    base_path = params['data_path']

    train_dataset, val_dataset, test_dataset = get_dataset(
        base_path, params['feature_path'], params['sub'], params['select_chs'],
        params['use_filter'], params['low_freq'], params['high_freq'], params['select_period']
    )
    logger.info(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    val_way = len(val_dataset)
    test_way = len(test_dataset)
    logger.info(
        f"Retrieval candidate pools: VAL={val_way}-way, TEST={test_way}-way. "
        "VAL way is the number of validation images ranked together in each epoch."
    )

    train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=len(val_dataset),  shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=len(test_dataset), shuffle=False)

    model = models.__dict__[params['net_name']](
        len(params['select_chs']), 1024,
        params['select_period'][1] - params['select_period'][0]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])
    criterion = ClipLoss()

    best_val_accu  = 0
    best_test_accu = 0
    select_model = best_model = None
    saved_metric = {
        'train_size': len(train_dataset),
        'val_way': val_way,
        'test_way': test_way,
    }
    loss_points = []
    accu_top1 = []
    accu_top3 = []
    accu_top5 = []

    for e in range(params['epoch']):
        model.train()
        all_loss = 0
        steps = 0
        for data in tqdm.tqdm(train_loader, desc=f"Epoch {e}"):
            optimizer.zero_grad()
            x = data['eeg'].to(device)
            img_list = torch.cat([data[k][:, None].to(device) for k in params['blur_level']], 1)

            if params['mixup']:
                rand_idx = np.random.permutation(x.shape[0])
                lam = np.random.beta(0.2, 0.2)
                x = lam * x + (1 - lam) * x[rand_idx]
                img_list = lam * img_list + (1 - lam) * img_list[rand_idx]

            eeg_f = model(x)
            img_f = model.get_image_feature(img_list)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)

            eeg_loss, img_loss = criterion(eeg_f, img_f, 1)
            loss = (eeg_loss.mean() + img_loss.mean()) / 2
            loss.backward()
            optimizer.step()
            all_loss += loss.detach().cpu()
            steps += 1

        train_loss = all_loss / steps
        loss_points.append(float(train_loss))

        val_top1, val_top3, val_top5   = get_test_accu(model, params, val_loader, device)
        test_top1, test_top3, test_top5 = get_test_accu(model, params, test_loader, device)
        accu_top1.append(test_top1)
        accu_top3.append(test_top3)
        accu_top5.append(test_top5)

        if best_val_accu < val_top1:
            select_model = copy.deepcopy(model.state_dict())
            best_val_accu = val_top1
            saved_metric.update({'test_top1_acc': test_top1, 'test_top3_acc': test_top3, 'test_top5_acc': test_top5})

        if best_test_accu < test_top1:
            best_test_accu = test_top1
            best_model = copy.deepcopy(model.state_dict())
            saved_metric.update({'best_test_top1_acc': test_top1, 'best_test_top3_acc': test_top3, 'best_test_top5_acc': test_top5})

        logger.info(f'VAL  ({val_way}-way) epoch:{e} loss:{train_loss:.4f} top1:{val_top1:.4f} top3:{val_top3:.4f} top5:{val_top5:.4f}')
        logger.info(f'TEST ({test_way}-way) epoch:{e} loss:{train_loss:.4f} top1:{test_top1:.4f} top3:{test_top3:.4f} top5:{test_top5:.4f}')

    saved_metric['train_loss_points']    = loss_points
    saved_metric['test_top1_acc_points'] = accu_top1
    saved_metric['test_top3_acc_points'] = accu_top3
    saved_metric['test_top5_acc_points'] = accu_top5

    torch.save(select_model, os.path.join(params['save_path'], f"{params['net_name']}_sub{params['sub']}_seed{params['seed']}_select.pth"))
    torch.save(best_model,   os.path.join(params['save_path'], f"{params['net_name']}_sub{params['sub']}_seed{params['seed']}_best.pth"))

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
    parser.add_argument('--data_path',         type=str,   default=os.path.join(root_dir, 'data', 'things-eeg'))
    parser.add_argument('--feature_path',      type=str,   default=os.path.join(root_dir, 'output', 'Image_feature'))
    parser.add_argument('--output_dir',        type=str,   default=os.path.join(root_dir, 'output', 'logs', 'main_eeg_course'))
    args = parser.parse_args()

    ALL_CHANNELS = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                    'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
                    'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                    'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                    'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                    'O1', 'Oz', 'O2']

    BLUR_LEVELS = ['l_1', 'l_3', 'l_9', 'l_15', 'l_21', 'l_27', 'l_33', 'l_39', 'l_45', 'l_51', 'l_57', 'l_63']

    save_path = os.path.join(args.output_dir, args.net_name, time.strftime('%Y-%m-%d-%H-%M'))
    os.makedirs(save_path, exist_ok=True)

    params = {
        'net_name':         args.net_name,
        'epoch':            args.epoch,
        'train_batch_size': args.train_batch_size,
        'test_batch_size':  args.test_batch_size,
        'lr':               args.lr,
        'mixup':            args.mixup,
        'blur_level':       BLUR_LEVELS,
        'use_filter':       args.use_filter,
        'low_freq':         args.low_freq,
        'high_freq':        args.high_freq,
        'select_chs':       ALL_CHANNELS,
        'select_period':    [0, 250],
        'data_path':        args.data_path,
        'feature_path':     args.feature_path,
        'save_path':        save_path,
        'sub':              1,
    }

    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(params, f, indent=4)

    logger = create_logger(save_path)
    logger.info(f"Saving to: {save_path}")

    all_metrics = []
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))

    for seed in seeds:
        logger.info(f'Training sub-01 seed={seed}')
        params['seed'] = seed
        metric = train(params, logger)
        all_metrics.append({'seed': seed, **metric})

    # Summary
    top1_sel  = [m['test_top1_acc'] for m in all_metrics]
    top5_sel  = [m['test_top5_acc'] for m in all_metrics]
    top1_best = [m['best_test_top1_acc'] for m in all_metrics]
    top5_best = [m['best_test_top5_acc'] for m in all_metrics]

    logger.info('='*60)
    logger.info('Results (val-selected model):')
    logger.info(f'  Top-1: {np.mean(top1_sel):.4f} ± {np.std(top1_sel):.4f}')
    logger.info(f'  Top-5: {np.mean(top5_sel):.4f} ± {np.std(top5_sel):.4f}')
    logger.info('Results (best test model):')
    logger.info(f'  Top-1: {np.mean(top1_best):.4f} ± {np.std(top1_best):.4f}')
    logger.info(f'  Top-5: {np.mean(top5_best):.4f} ± {np.std(top5_best):.4f}')

    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(save_path, "all_metrics.csv"), index=False)
    logger.info(f"Metrics saved to {save_path}/all_metrics.csv")
