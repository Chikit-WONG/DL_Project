# import os
# os.makedirs("./AI_Cache/huggingface", exist_ok=True)
# os.makedirs("./AI_Cache/torch", exist_ok=True)
# os.environ["HF_HOME"] = r"./AI_Cache/huggingface"
# os.environ["TORCH_HOME"] = r"./AI_Cache/torch"
import logging

logging.basicConfig(
    level=logging.INFO,  # 日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",  # 输出格式
    handlers=[
        logging.FileHandler("run.log", mode="w", encoding="utf-8"),  # 输出到文件 run.log
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Literal, Sequence, Union
import warnings
from PIL import Image
import ast
import datasets
import random

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
logging.info("running")

# 与项目根目录 `reproduce.py` 共用 `ATM_S`，避免 `2/` 与根目录两份结构漂移。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# from blocks import CogCapProEEGEncoder as ATM_S
from atm_block import ATM_S

def set_seed(seed=114514):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ====================== 从 EEG Project Sample Code 导入的数据加载函数 ======================
def _selected_channel_indices_from_jsonl(
    selected_channels: Union[str, Sequence[str]],
    eeg_channel_jsonl: Union[str, Path],
) -> List[int]:
    """Map EEG channel names to channel indices."""
    if isinstance(selected_channels, str):
        selected_channels = [selected_channels]
    selected_channels = list(selected_channels)

    channel_names: List[str] = []
    with open(eeg_channel_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            name = item.get("name") or item.get("channel_name") or item.get("label")
            if name is None:
                raise KeyError(
                    "Each JSONL record must contain one of: 'name', 'channel_name', or 'label'."
                )
            channel_names.append(str(name))

    name_to_index = {name: idx for idx, name in enumerate(channel_names)}

    missing = [ch for ch in selected_channels if ch not in name_to_index]
    if missing:
        raise ValueError(f"Unknown EEG channels: {missing}")

    return [name_to_index[ch] for ch in selected_channels]


def load_eeg_dataset(
    *,
    data_directory: Union[str, Path],
    split: Literal["train", "test"],
    avg_trials: bool = True,
    selected_channels: Optional[Union[str, Sequence[str]]] = None,
    eeg_channel_jsonl: Union[str, Path] = "./image-eeg-data/EEG_CHANNELS.jsonl",
) -> datasets.Dataset:
    """Build a Hugging Face dataset for the released EEG data.
    
    Returns dataset with columns:
    - `eeg`: Array2D float32 [C, T]
    - `image_id`: string
    """
    pt_path = Path(data_directory).joinpath(f"{split}.pt")
    loaded = torch.load(str(pt_path), weights_only=False)

    x = torch.as_tensor(loaded["eeg"])  # [N, TRIAL, C, T] or [N, C, T]
    if x.ndim == 4:
        if avg_trials:
            x = x.mean(dim=1)  # [N, C, T]
        else:
            x = x.reshape(-1, *x.shape[2:])  # [N * TRIAL, C, T]
    elif x.ndim != 3:
        raise ValueError(f"Unexpected EEG shape: {tuple(x.shape)} in {pt_path}")

    if selected_channels is not None:
        sel_idx = _selected_channel_indices_from_jsonl(selected_channels, eeg_channel_jsonl)
        x = x[:, sel_idx, :]

    imgs = np.array(loaded["img"])
    if avg_trials:
        if imgs.ndim == 2:
            imgs = imgs[:, 0]
        imgs = imgs.reshape(-1)[: x.shape[0]]
    else:
        imgs = imgs.reshape(-1)

    image_ids = [Path(p).stem for p in imgs.tolist()]
    if len(image_ids) != x.shape[0]:
        raise ValueError(
            f"EEG/image mismatch: {x.shape[0]} vs {len(image_ids)} for {pt_path}"
        )

    x_np = x.float().cpu().numpy()  # [N, C, T]
    C, T = x_np.shape[1], x_np.shape[2]

    features = datasets.Features(
        {
            "eeg": datasets.Array2D(shape=(C, T), dtype="float32"),
            "image_id": datasets.Value("string"),
        }
    )

    ds = datasets.Dataset.from_dict(
        {
            "eeg": list(x_np),
            "image_id": image_ids,
        },
        features=features,
    )
    return ds


def _avg_trials_for_split(split: Literal["train", "test"]) -> bool:
    """与论文数据流对齐：训练保留试次维度；测试对试次取平均以降噪。"""
    if split == "train":
        return True
    if split == "test":
        return True
    raise ValueError(f"Unknown split: {split}")


def _aligned_img_refs_from_loaded(loaded: dict, split: str) -> Tuple[List[str], List[str]]:
    """与 load_eeg_dataset 完全一致的 EEG/图像行对齐，返回 (img_ref 字符串列表, image_id stem 列表)。"""
    avg_trials = _avg_trials_for_split(split)  # type: ignore[arg-type]
    x = torch.as_tensor(loaded["eeg"])
    if x.ndim == 4:
        if avg_trials:
            x = x.mean(dim=1)
        else:
            x = x.reshape(-1, *x.shape[2:])
    elif x.ndim != 3:
        raise ValueError(f"Unexpected EEG shape: {tuple(x.shape)}")

    imgs = np.array(loaded["img"])
    if avg_trials:
        if imgs.ndim == 2:
            imgs = imgs[:, 0]
        imgs = imgs.reshape(-1)[: x.shape[0]]
    else:
        imgs = imgs.reshape(-1)

    if len(imgs) != x.shape[0]:
        raise ValueError(f"EEG/image row mismatch: EEG rows={x.shape[0]} vs img rows={len(imgs)}")

    img_refs = [str(p).strip() for p in imgs.tolist()]
    image_ids = [Path(p).stem for p in img_refs]
    return img_refs, image_ids


def _topk_accuracy(similarity: torch.Tensor, ks: Tuple[int, ...] = (1, 5)) -> Dict[str, float]:
    max_k = min(max(ks), similarity.shape[1])
    labels = torch.arange(similarity.shape[0], device=similarity.device)
    topk_idx = similarity.topk(k=max_k, dim=1).indices
    matches = topk_idx.eq(labels.unsqueeze(1))
    return {
        f"top{k}": matches[:, : min(k, similarity.shape[1])].any(dim=1).float().mean().item() * 100
        for k in ks
    }


@torch.no_grad()
def evaluate_retrieval(model, test_loader, device="cuda", *, verbose: bool = True):
    model_to_eval = model.module if isinstance(model, nn.DataParallel) else model
    model_to_eval.eval()

    all_eeg_embeds = []
    all_clip_embeds = []
    device_obj = torch.device(device) if not isinstance(device, torch.device) else device

    for eeg, clip_feat in test_loader:
        eeg = eeg.to(device_obj, non_blocking=True)
        clip_feat = clip_feat.to(device_obj, non_blocking=True)

        eeg_emb = F.normalize(model_to_eval(eeg).float(), dim=-1)
        clip_emb = F.normalize(clip_feat.float(), dim=-1)
        all_eeg_embeds.append(eeg_emb.cpu())
        all_clip_embeds.append(clip_emb.cpu())

    eeg_embeds = torch.cat(all_eeg_embeds, dim=0)
    clip_embeds = torch.cat(all_clip_embeds, dim=0)

    if verbose:
        logging.info(f"EEG embeds count: {len(all_eeg_embeds)}")
        logging.info(f"CLIP embeds count: {len(all_clip_embeds)}")
        logging.info(f"EEG embeddings shape: {eeg_embeds.shape}")
        logging.info(f"CLIP embeddings shape: {clip_embeds.shape}")

    similarity = torch.matmul(eeg_embeds, clip_embeds.T)
    eeg_to_image = _topk_accuracy(similarity, ks=(1, 5))
    image_to_eeg = _topk_accuracy(similarity.T, ks=(1, 5))

    metrics = {
        "eeg_to_image_top1": eeg_to_image["top1"],
        "eeg_to_image_top5": eeg_to_image["top5"],
        "image_to_eeg_top1": image_to_eeg["top1"],
        "image_to_eeg_top5": image_to_eeg["top5"],
    }

    if verbose:
        logging.info("\n--- 评估结果 ---")
        logging.info(
            f"EEG -> Image | Top-1: {metrics['eeg_to_image_top1']:.2f}% | Top-5: {metrics['eeg_to_image_top5']:.2f}%"
        )
        logging.info(
            f"Image -> EEG | Top-1: {metrics['image_to_eeg_top1']:.2f}% | Top-5: {metrics['image_to_eeg_top5']:.2f}%"
        )
    return metrics


def precompute_clip_for_split(
    data_root: Path,
    split: str = "train",
    open_clip_arch: str = "ViT-H-14",
    open_clip_pretrained: str = "laion2b_s32b_b79k",
    batch_size: int = 128,
    save_path: Optional[str] = None,
):
    """使用 OpenCLIP ViT-H/14 的 **encode_image** 输出（1024 维），与 ATM_S 对齐。"""
    try:
        import open_clip
    except ImportError as e:
        raise ImportError(
            "需要安装 open-clip-torch 才能使用 ViT-H/14：pip install open-clip-torch"
        ) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt_path = data_root / f"{split}.pt"
    data = torch.load(pt_path, weights_only=False, map_location="cpu")
    logging.info(f"Loaded {split}.pt → keys: {list(data.keys())}")

    def parse_image_reference(item) -> str:
        """从item中提取纯文件名，处理各种格式（与 EEG 行对齐后的 ref 一致）。"""
        s = str(item).strip()
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                s = str(parsed[0]).strip()
        except (ValueError, SyntaxError):
            pass
        if len(s) > 500:
            logging.warning(f"检测到异常长的路径字符串 (len={len(s)})")
            if "/" in s or "\\" in s:
                parts = s.replace("\\", "/").split("/")
                for part in reversed(parts):
                    if part and "." in part:
                        return part.strip()
        s_clean = s.replace("\\", "/")
        if "/" in s_clean:
            s = s_clean.split("/")[-1]
        return s.strip()

    raw_refs, image_ids = _aligned_img_refs_from_loaded(data, split)
    img_refs = [parse_image_reference(r) for r in raw_refs]
    image_ids = [Path(r).stem for r in img_refs]

    logging.info(
        f"对齐后样本数={len(img_refs)} (split={split}, avg_trials={_avg_trials_for_split(split)})"
    )
    logging.debug(f"前3个 img_refs: {img_refs[:3]}")

    image_root = data_root / f"{split}_images" / f"{split}_images"
    logging.info(f"✅ 图片根目录: {image_root.absolute()}")

    name_map = {}
    for img_path in image_root.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            name_map.setdefault(img_path.name, []).append(img_path)

    logging.info(f"✅ 找到 {sum(len(v) for v in name_map.values())} 张图片")

    def resolve_image_path(path_str: str) -> Path:
        """使用文件名直接查找，避免路径拼接问题"""
        filename = Path(path_str).name
        logging.debug(f"resolve_image_path: 查找文件名 = {repr(filename)}")
        
        # 首先尝试按文件名直接查找
        basename_matches = name_map.get(filename, [])
        
        if len(basename_matches) == 1:
            logging.debug(f"✓ 按文件名找到: {basename_matches[0]}")
            return basename_matches[0]
        elif len(basename_matches) > 1:
            logging.warning(f"⚠ 文件名 {filename} 有多个匹配，使用第一个")
            return basename_matches[0]
        
        # 备选方案：尝试路径拼接（保留原有逻辑但添加保护）
        try:
            rel_path = Path(path_str.replace("\\", "/"))
            candidates = [
                image_root / rel_path,
                image_root / rel_path.name,
            ]
            for candidate in candidates:
                if candidate is not None and len(str(candidate)) < 4096 and candidate.is_file():
                    logging.debug(f"✓ 通过路径拼接找到: {candidate}")
                    return candidate
        except Exception as e:
            logging.debug(f"路径拼接失败: {e}")
        
        # 都没找到
        logging.error(f"✗ 未找到图片: {path_str} (文件名: {filename})")
        logging.error(f"  可用的类似文件: {[k for k in name_map.keys() if filename[:5] in k][:5]}")
        raise FileNotFoundError(f"未找到图片文件: {path_str}")

    created = open_clip.create_model_and_transforms(
        open_clip_arch,
        pretrained=open_clip_pretrained,
    )
    if len(created) == 3:
        clip_model, _preprocess_train, preprocess = created
    else:
        clip_model, preprocess = created  # type: ignore[misc]
    clip_model = clip_model.to(device).eval().float()

    embeddings = []
    resolved_paths = []
    with torch.no_grad():
        for i in range(0, len(img_refs), batch_size):
            batch_refs = img_refs[i:i + batch_size]
            batch_imgs = []

            for path_str in batch_refs:
                img_path = resolve_image_path(path_str)
                resolved_paths.append(str(img_path))
                with Image.open(img_path) as img:
                    batch_imgs.append(preprocess(img.convert("RGB")))

            batch_tensor = torch.stack(batch_imgs).to(device)
            emb = clip_model.encode_image(batch_tensor).float().cpu()
            embeddings.append(emb)

            processed = i + len(batch_refs)
            if processed % 500 == 0 or processed == len(img_refs):
                logging.info(f"已处理 {processed}/{len(img_refs)}")

    all_emb = torch.cat(embeddings, dim=0)
    payload = {
        "embeddings": all_emb,
        "image_ids": image_ids,
        "image_refs": img_refs,
        "resolved_paths": resolved_paths,
        "model_name": f"open_clip:{open_clip_arch}:{open_clip_pretrained}",
        "open_clip_arch": open_clip_arch,
        "open_clip_pretrained": open_clip_pretrained,
        "feature_type": "open_clip_encode_image",
        "embedding_dim": all_emb.shape[1],
    }

    save_path = save_path or str(data_root / f"{split}_clip_openclip_vith14_1024.pt")
    torch.save(payload, save_path)
    logging.info(f"\n🎉 成功！保存到: {save_path} | 形状: {all_emb.shape} | 特征: {payload['feature_type']}")

    return payload

# ====================== 2. CLIP Loss ======================
def clip_loss(eeg_emb, img_emb, temp=0.07):
    return contrastive_loss(eeg_emb, img_emb, temperature=temp)


def contrastive_loss(eeg_embeds, clip_embeds, temperature=0.07, logit_scale: Optional[torch.Tensor] = None):
    eeg_embeds = F.normalize(eeg_embeds.float(), dim=-1)
    clip_embeds = F.normalize(clip_embeds.float(), dim=-1)

    if logit_scale is None:
        scale = 1.0 / temperature
    else:
        scale = logit_scale.exp().clamp(max=100.0)

    logits = torch.matmul(eeg_embeds, clip_embeds.T) * scale
    labels = torch.arange(eeg_embeds.shape[0], device=eeg_embeds.device)

    loss_eeg = F.cross_entropy(logits, labels)
    loss_clip = F.cross_entropy(logits.T, labels)
    return (loss_eeg + loss_clip) / 2


class EEGProjectDataset(Dataset):
    """Strictly aligned EEG <-> CLIP retrieval dataset."""

    def __init__(
        self,
        data_directory: Union[str, Path],
        split: str = "train",
        clip_pt_path: Optional[str] = None,
        map_location="cpu",
        max_samples: Optional[int] = None,
    ):
        avg_trials = _avg_trials_for_split(split)  # train: False, test: True
        dataset = load_eeg_dataset(
            data_directory=data_directory,
            split=split,
            avg_trials=avg_trials,
            selected_channels=None,
            eeg_channel_jsonl=str(Path(data_directory) / "EEG_CHANNELS.jsonl"),
        )
        logging.info(
            f"EEGProjectDataset split={split} -> avg_trials={avg_trials} (训练保留试次 / 测试试次平均)"
        )

        eeg_list = []
        image_ids = []
        for sample in dataset:
            eeg_list.append(torch.from_numpy(np.array(sample["eeg"])).float())
            image_ids.append(sample["image_id"])

        self.eeg = torch.stack(eeg_list)
        self.image_ids = list(image_ids)

        if not clip_pt_path:
            raise ValueError("clip_pt_path must be provided")

        clip_payload = torch.load(clip_pt_path, map_location=map_location, weights_only=False)
        if not isinstance(clip_payload, dict) or "embeddings" not in clip_payload or "image_ids" not in clip_payload:
            raise ValueError(
                "CLIP 缓存不是新版 payload 格式，请删除旧缓存并重新运行预计算单元。"
            )

        self.clip = torch.as_tensor(clip_payload["embeddings"]).float()
        clip_image_ids = list(clip_payload["image_ids"])
        self.feature_type = clip_payload.get("feature_type", "unknown")
        self.clip_model_name = clip_payload.get("model_name", "unknown")

        if len(self.eeg) != len(self.clip):
            raise ValueError(
                f"EEG size ({len(self.eeg)}) != CLIP size ({len(self.clip)})，请重新生成缓存。"
            )

        if clip_image_ids != self.image_ids:
            mismatch_idx = next(
                idx for idx, (a, b) in enumerate(zip(self.image_ids, clip_image_ids)) if a != b
            )
            raise ValueError(
                "EEG 与 CLIP 的 image_id 顺序不一致，"
                f"首个不一致位置: {mismatch_idx} ({self.image_ids[mismatch_idx]} != {clip_image_ids[mismatch_idx]})"
            )

        if split == "train" and max_samples is not None and max_samples > 0:
            n_orig = len(self.eeg)
            n = min(n_orig, int(max_samples))
            if n < n_orig:
                self.eeg = self.eeg[:n].contiguous()
                self.clip = self.clip[:n].contiguous()
                self.image_ids = self.image_ids[:n]
                logging.info(
                    "训练集截断为前 %s 条（原 %s 条，max_samples=%s）",
                    n,
                    n_orig,
                    max_samples,
                )
            elif n_orig < int(max_samples):
                logging.warning(
                    "训练集仅 %s 条，小于 max_samples=%s，将全部用于训练",
                    n_orig,
                    max_samples,
                )

        logging.info("--- EEG 数据加载完成 ---")
        logging.info(f"EEG 形状: {self.eeg.shape}")
        logging.info(f"CLIP 形状: {self.clip.shape}")
        logging.info(f"CLIP 特征类型: {self.feature_type}")
        logging.info(f"CLIP 模型: {self.clip_model_name}")

    def __getitem__(self, index):
        return self.eeg[index], self.clip[index]

    def __len__(self):
        return len(self.eeg)


def train_atm(
    model,
    train_loader,
    epochs=40,
    lr=3e-4,
    device="cuda",
    checkpoint_path: Optional[Union[str, Path]] = None,
    test_loader: Optional[DataLoader] = None,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = True,  # 混合精度训练
):
    device_obj = torch.device(device) if not isinstance(device, torch.device) else device
    model.to(device_obj)

    # 重要：不使用 DataParallel，改用 DistributedDataParallel 或单卡
    # 如果必须多卡，使用 DDP 而不是 DataParallel
    if device_obj.type == "cuda" and torch.cuda.device_count() > 1:
        logging.warning(f"检测到 {torch.cuda.device_count()} 张 GPU，但建议使用单卡训练以减少显存。")
        logging.warning("如需多卡，请改用 DistributedDataParallel。")
        # 仍然使用单卡
        model = model.to(device_obj)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 清理显存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    history = []
    best_loss = float("inf")
    best_eeg_to_image_top1 = -1.0
    best_epoch = -1
    best_metrics: Optional[Dict[str, float]] = None

    logging.info(f"开始训练，设备: {device_obj}, Epochs: {epochs}, LR: {lr}")
    logging.info(f"梯度累积步数: {gradient_accumulation_steps}, 混合精度: {use_amp}")
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"模型参数量: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    # 打印初始显存
    if device_obj.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logging.info(f"初始显存: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB")

    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)  # 只在 epoch 开始时清零一次
        
        for batch_idx, (eeg, clip_feat) in enumerate(train_loader):
            eeg = eeg.to(device_obj, non_blocking=True)
            clip_feat = clip_feat.to(device_obj, non_blocking=True)

            # 混合精度前向传播
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(eeg)
                    model_ref = model.module if isinstance(model, nn.DataParallel) else model
                    loss = contrastive_loss(outputs, clip_feat, logit_scale=model_ref.logit_scale)
                    loss = loss / gradient_accumulation_steps  # 归一化损失
                
                # 混合精度反向传播
                scaler.scale(loss).backward()
            else:
                outputs = model(eeg)
                model_ref = model.module if isinstance(model, nn.DataParallel) else model
                loss = contrastive_loss(outputs, clip_feat, logit_scale=model_ref.logit_scale)
                loss = loss / gradient_accumulation_steps
                loss.backward()
            
            running_loss += loss.item() * gradient_accumulation_steps
            
            # 梯度累积
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # 定期清理显存（可选）
            if batch_idx % 100 == 0 and device_obj.type == "cuda":
                torch.cuda.empty_cache()
            
            # 可选：打印显存使用情况（调试用）
            if batch_idx == 0 and epoch == 0 and device_obj.type == "cuda":
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logging.info(f"第一个 batch 后显存: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB")

        # 处理剩余未更新的梯度
        if (batch_idx + 1) % gradient_accumulation_steps != 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]

        row: Dict[str, Union[int, float]] = {
            "epoch": int(epoch + 1),
            "loss": float(avg_loss),
            "lr": float(current_lr),
        }

        # 每个 epoch 后清理显存
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()

        if test_loader is not None:
            metrics = evaluate_retrieval(model, test_loader, device=device_obj, verbose=False)
            row["eeg_to_image_top1"] = metrics["eeg_to_image_top1"]
            row["eeg_to_image_top5"] = metrics["eeg_to_image_top5"]
            row["image_to_eeg_top1"] = metrics["image_to_eeg_top1"]
            row["image_to_eeg_top5"] = metrics["image_to_eeg_top5"]
            history.append(row)
            logging.info(
                f"Epoch [{epoch+1}/{epochs}] Loss={avg_loss:.4f} LR={current_lr:.6f} | "
                f"EEG->Img Top-1={metrics['eeg_to_image_top1']:.2f}% Top-5={metrics['eeg_to_image_top5']:.2f}% | "
                f"Img->EEG Top-1={metrics['image_to_eeg_top1']:.2f}% Top-5={metrics['image_to_eeg_top5']:.2f}%"
            )

            if metrics["eeg_to_image_top1"] > best_eeg_to_image_top1 and checkpoint_path is not None:
                best_eeg_to_image_top1 = metrics["eeg_to_image_top1"]
                best_epoch = epoch + 1
                best_metrics = dict(metrics)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                model_ref = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(
                    {
                        "model_state_dict": model_ref.state_dict(),
                        "history": history,
                        "best_eeg_to_image_top1": best_eeg_to_image_top1,
                        "best_epoch": best_epoch,
                        "best_metrics": best_metrics,
                    },
                    checkpoint_path,
                )
                logging.info(
                    f"💾 已更新最佳 checkpoint (EEG->Image Top-1={best_eeg_to_image_top1:.2f}%, epoch={best_epoch}): {checkpoint_path}"
                )
        else:
            history.append(row)
            logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
            if checkpoint_path is not None and avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                model_ref = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(
                    {
                        "model_state_dict": model_ref.state_dict(),
                        "history": history,
                        "best_loss": best_loss,
                    },
                    checkpoint_path,
                )
                logging.info(f"💾 已更新最佳 checkpoint (min Loss): {checkpoint_path}")

    return model, history


# ====================== 5. Retrieval (200-way) ======================
@torch.no_grad()
def retrieval(model: nn.Module, test_eeg: torch.Tensor, test_clip_db: torch.Tensor) -> Tuple[float, float]:
    model_to_eval = model.module if isinstance(model, nn.DataParallel) else model
    model_to_eval.eval()

    device = next(model.parameters()).device
    test_eeg = test_eeg.to(device)
    test_clip_db = test_clip_db.to(device)

    eeg_emb = F.normalize(model_to_eval(test_eeg).float(), dim=-1)
    db_emb = F.normalize(test_clip_db.float(), dim=-1)
    sim = eeg_emb @ db_emb.T
    metrics = _topk_accuracy(sim, ks=(1, 5))
    return metrics["top1"], metrics["top5"]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path("./image-eeg-data")
    # OpenCLIP ViT-H/14：encode_image 为 1024 维（与 ATM_S 输出一致）
    open_clip_arch = "ViT-H-14"
    open_clip_pretrained = "laion2b_s32b_b79k"
    checkpoint_path = Path("../artifacts/atm_s_retrieval_best.pt")

    num_gpus = torch.cuda.device_count()
    logging.info(f"可用GPU数量: {num_gpus}")
    logging.info(f"数据目录: {data_root}")
    logging.info(f"OpenCLIP 图像编码: {open_clip_arch} ({open_clip_pretrained})")

    train_clip_path = str(data_root / "train_clip_openclip_vith14_1024.pt")
    test_clip_path = str(data_root / "test_clip_openclip_vith14_1024.pt")

    if not Path(train_clip_path).exists():
        logging.info("🚀 正在预计算训练集 OpenCLIP ViT-H/14 image embedding...")
        precompute_clip_for_split(
            data_root,
            split="train",
            open_clip_arch=open_clip_arch,
            open_clip_pretrained=open_clip_pretrained,
            save_path=train_clip_path,
        )
    else:
        logging.info(f"训练集 CLIP 缓存已存在: {train_clip_path}")

    if not Path(test_clip_path).exists():
        logging.info("正在预计算测试集 OpenCLIP ViT-H/14 image embedding...")
        precompute_clip_for_split(
            data_root,
            split="test",
            open_clip_arch=open_clip_arch,
            open_clip_pretrained=open_clip_pretrained,
            save_path=test_clip_path,
        )
    else:
        logging.info(f"测试集 CLIP 缓存已存在: {test_clip_path}")

    train_dataset = EEGProjectDataset(str(data_root), split="train", clip_pt_path=train_clip_path)
    test_dataset = EEGProjectDataset(str(data_root), split="test", clip_pt_path=test_clip_path)

    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")

    batch_size = 16
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if num_gpus > 1 else 0,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if num_gpus > 1 else 0,
        pin_memory=True,
        drop_last=False,
    )

    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Num workers: {4 if num_gpus > 1 else 0}")

    clip_dim = train_dataset.clip.shape[1]
    if clip_dim != 1024:
        raise ValueError(
            f"ATM_S 输出为 1024 维，需与 OpenCLIP ViT-H/14 的 encode_image 一致；当前缓存维度为 {clip_dim}。"
            "请删除旧缓存后重新预计算 train/test_clip_openclip_vith14_1024.pt。"
        )
    model = ATM_S(out_dim=clip_dim).to(device)
    logging.info(f"CLIP 特征维度 (OpenCLIP encode_image): {clip_dim}")

    logging.info("\n" + "=" * 60)
    logging.info("开始训练 ATM_S 模型")
    logging.info("=" * 60)
    trained_model, train_history = train_atm(
        model,
        train_loader,
        epochs=40,
        lr=3e-4,
        device=device,
        checkpoint_path=checkpoint_path,
        test_loader=test_loader,
    )
    logging.info(f"最后一轮训练记录: {train_history[-1]}")

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "best_eeg_to_image_top1" in ckpt:
            model_ref = trained_model.module if isinstance(trained_model, nn.DataParallel) else trained_model
            model_ref.load_state_dict(ckpt["model_state_dict"])
            logging.info(
                f"已加载按 EEG->Image Top-1 最优的 checkpoint（epoch {ckpt.get('best_epoch')}，"
                f"Top-1={ckpt['best_eeg_to_image_top1']:.2f}%）用于最终评估。"
            )

    retrieval_metrics = evaluate_retrieval(trained_model, test_loader, device=device)
    logging.info(f"最终检索指标: {retrieval_metrics}")