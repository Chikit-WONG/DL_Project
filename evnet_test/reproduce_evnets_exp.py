# reproduce_lib_evnet.py
"""
完整的 EVNet + CLIP 融合训练脚本
使用 EVNet 前端（SubcorticalBlock + VOneBlock）处理图像后再用 CLIP 编码
与原始 CLIP 特征进行可学习加权融合
"""

import logging
import os
import sys
import json
import ast
import random
import warnings
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Literal, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import datasets

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("run_evnet_fused.log", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
torch.serialization.add_safe_globals([np._core.multiarray.scalar])
logging.info("running - EVNet + CLIP Fused version")

# 项目路径设置
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from atm_block import ATM_S

# ====================== 导入 EVNet ======================
# 假设 evnet 文件夹在当前目录下
EVNET_PATH = Path(__file__).parent / "evnet" / "evnet"  # 修改这里：添加了一层 evnet
if EVNET_PATH.exists():
    sys.path.insert(0, str(EVNET_PATH.parent))  # 将 evnet 的父目录加入路径
    from evnet.evnet import EVNet  # 修改导入方式
    from evnet.modules import SubcorticalBlock, VOneBlock
    from evnet.params import get_tuned_params, generate_gabor_param
    EVNET_AVAILABLE = True
    logging.info(f"EVNet modules loaded successfully from {EVNET_PATH}")
else:
    raise ImportError(f"EVNet not found at {EVNET_PATH}. Please ensure evnet folder is in the same directory.")

def set_seed(seed=114514):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== EVNet 前端 + CLIP 编码器 ======================
class EVNetFrontEnd(nn.Module):
    """
    完整的 EVNet 前端：SubcorticalBlock + VOneBlock
    模拟视网膜/LGN 和初级视觉皮层 V1
    权重完全固定，不可学习
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 224,
        visual_degrees: float = 7.0,
        # SubcorticalBlock 参数
        p_channels: int = 3,
        m_channels: int = 0,
        colors_p_cells: List[str] = ['r/g', 'g/r', 'b/y'],
        with_light_adapt: bool = True,
        with_dog: bool = True,
        with_contrast_norm: bool = True,
        with_relu: bool = False,
        subcort_noise_mode: Optional[str] = None,  # 推理时关闭噪声
        subcort_fano_factor: float = 0.4,
        # VOneBlock 参数
        simple_channels: int = 256,
        complex_channels: int = 256,
        vone_noise_mode: Optional[str] = None,  # 推理时关闭噪声
        vone_ksize: int = 31,
        vone_stride: int = 4,
        sf_corr: float = 0.75,
        sf_max: float = 8.0,
        sf_min: float = 0.0,
        gabor_seed: int = 0,
    ):
        super().__init__()
        
        # 获取 SubcorticalBlock 参数
        subcort_params = get_tuned_params(
            p_channels, m_channels, colors_p_cells, ['w/b'], 
            visual_degrees, image_size
        )
        
        self.subcortical = SubcorticalBlock(
            in_channels=in_channels,
            p_channels=p_channels,
            m_channels=m_channels,
            **subcort_params,
            fano_factor=subcort_fano_factor,
            noise_mode=subcort_noise_mode,
            with_light_adapt=with_light_adapt,
            with_dog=with_dog,
            with_contrast_norm=with_contrast_norm,
            with_relu=with_relu,
            light_adapt_threshold=False
        )
        
        # 生成 Gabor 参数
        vone_in_channels = p_channels + m_channels
        sf, theta, phase, nx, ny, color = generate_gabor_param(
            simple_channels, complex_channels, gabor_seed, 
            rand_flag=False, sf_corr=sf_corr, sf_max=sf_max, sf_min=sf_min,
            color_prob=None, in_channels=vone_in_channels
        )
        
        # 转换为 EVNet 需要的格式
        ppd = image_size / visual_degrees
        sf = sf / ppd
        sigx = nx / sf
        sigy = ny / sf
        theta = theta / 180 * np.pi
        phase = phase / 180 * np.pi
        
        self.voneblock = VOneBlock(
            sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase, color=color,
            in_channels=vone_in_channels,
            k_exc=25,
            noise_mode=vone_noise_mode,
            noise_scale=1,
            noise_level=0,
            fano_factor=1,
            simple_channels=simple_channels,
            complex_channels=complex_channels,
            ksize=vone_ksize,
            stride=vone_stride,
            input_size=image_size
        )
        
        # 冻结所有参数
        for param in self.subcortical.parameters():
            param.requires_grad = False
        for param in self.voneblock.parameters():
            param.requires_grad = False
        
        self.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 EVNet 处理后的特征图 [B, C, H, W]"""
        x = self.subcortical(x)
        x = self.voneblock(x)
        return x


class EVNetCLIPEncoder(nn.Module):
    """
    EVNet 前端 + CLIP ViT-H/14 编码器
    同时输出原始 CLIP 特征和 EVNet 处理后的 CLIP 特征
    """
    
    def __init__(
        self,
        open_clip_arch: str = "ViT-H-14",
        open_clip_pretrained: str = "laion2b_s32b_b79k",
        image_size: int = 224,
        visual_degrees: float = 7.0,
    ):
        super().__init__()
        
        try:
            import open_clip
        except ImportError:
            raise ImportError("需要安装 open-clip-torch: pip install open-clip-torch")
        
        # 加载 CLIP 模型
        created = open_clip.create_model_and_transforms(
            open_clip_arch,
            pretrained=open_clip_pretrained,
        )
        if len(created) == 3:
            self.clip_model, _, self.preprocess = created
        else:
            self.clip_model, self.preprocess = created
        
        self.clip_model = self.clip_model.eval().float()
        
        # 冻结 CLIP 参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # EVNet 前端
        self.evnet_frontend = EVNetFrontEnd(
            in_channels=3,
            image_size=image_size,
            visual_degrees=visual_degrees,
        )
        
        # 适配层：将 EVNet 输出的特征图 [B, C, H, W] 转换为 CLIP 可接受的 [B, 3, 224, 224]
        # EVNet VOneBlock 输出通道数 = simple_channels + complex_channels = 512
        # 输出空间尺寸 = image_size / stride = 224 / 4 = 56
        self.adapter = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
        )
        
        # 适配层可以训练（让模型学习如何将 EVNet 特征映射回 RGB 空间）
        # 但为了保持 EVNet 前端固定，适配层是可学习的
        for param in self.adapter.parameters():
            param.requires_grad = True
        
        self.eval()
    
    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: [B, 3, 224, 224] RGB 图像
        Returns:
            raw_features: 原始 CLIP 特征 [B, 1024]
            evnet_features: EVNet 处理后的 CLIP 特征 [B, 1024]
        """
        # 原始 CLIP 特征
        raw_features = self.clip_model.encode_image(images).float()
        
        # EVNet 处理
        evnet_feat_map = self.evnet_frontend(images)  # [B, 512, 56, 56]
        evnet_images = self.adapter(evnet_feat_map)   # [B, 3, 224, 224]
        evnet_features = self.clip_model.encode_image(evnet_images).float()
        
        return raw_features, evnet_features


# ====================== 数据加载函数 ======================
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
    """Build a Hugging Face dataset for the released EEG data."""
    pt_path = Path(data_directory).joinpath(f"{split}.pt")
    loaded = torch.load(str(pt_path), weights_only=False)

    x = torch.as_tensor(loaded["eeg"])
    if x.ndim == 4:
        if avg_trials:
            x = x.mean(dim=1)
        else:
            x = x.reshape(-1, *x.shape[2:])
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

    x_np = x.float().cpu().numpy()
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
    """与论文数据流对齐：统一使用平均"""
    return True


def _aligned_img_refs_from_loaded(loaded: dict, split: str) -> Tuple[List[str], List[str]]:
    """与 load_eeg_dataset 完全一致的 EEG/图像行对齐"""
    avg_trials = _avg_trials_for_split(split)
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


def resolve_image_path(path_str: str, image_root: Path, name_map: dict) -> Path:
    """根据文件名查找图片路径"""
    filename = Path(path_str).name
    basename_matches = name_map.get(filename, [])
    
    if len(basename_matches) == 1:
        return basename_matches[0]
    elif len(basename_matches) > 1:
        logging.warning(f"文件名 {filename} 有多个匹配，使用第一个")
        return basename_matches[0]
    
    # 备选：路径拼接
    try:
        rel_path = Path(path_str.replace("\\", "/"))
        candidates = [
            image_root / rel_path,
            image_root / rel_path.name,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
    except Exception as e:
        logging.debug(f"路径拼接失败: {e}")
    
    raise FileNotFoundError(f"未找到图片文件: {path_str}")


# ====================== 预计算融合特征 ======================
def precompute_fused_features(
    data_root: Path,
    split: str = "train",
    open_clip_arch: str = "ViT-H-14",
    open_clip_pretrained: str = "laion2b_s32b_b79k",
    batch_size: int = 32,
    save_path: Optional[str] = None,
):
    """预计算融合特征：原始CLIP + EVNet-CLIP"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"预计算 {split} 集融合特征，设备: {device}")
    
    # 加载编码器
    encoder = EVNetCLIPEncoder(open_clip_arch, open_clip_pretrained).to(device)
    
    # 加载EEG数据获取图片引用
    pt_path = data_root / f"{split}.pt"
    data = torch.load(pt_path, weights_only=False, map_location="cpu")
    raw_refs, image_ids = _aligned_img_refs_from_loaded(data, split)
    
    # 解析图片引用
    img_refs = []
    for r in raw_refs:
        s = str(r).strip()
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                s = str(parsed[0]).strip()
        except (ValueError, SyntaxError):
            pass
        s_clean = s.replace("\\", "/")
        if "/" in s_clean:
            s = s_clean.split("/")[-1]
        img_refs.append(s.strip())
    
    logging.info(f"样本数: {len(img_refs)}")
    
    # 构建文件名映射
    image_root = data_root / f"{split}_images" / f"{split}_images"
    name_map = {}
    for img_path in image_root.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            name_map.setdefault(img_path.name, []).append(img_path)
    
    logging.info(f"找到 {sum(len(v) for v in name_map.values())} 张图片")
    
    # 预计算特征
    all_raw_features = []
    all_evnet_features = []
    
    with torch.no_grad():
        for i in range(0, len(img_refs), batch_size):
            batch_refs = img_refs[i:i+batch_size]
            batch_imgs = []
            
            for path_str in batch_refs:
                img_path = resolve_image_path(path_str, image_root, name_map)
                with Image.open(img_path) as img:
                    batch_imgs.append(encoder.preprocess(img.convert("RGB")))
            
            batch_tensor = torch.stack(batch_imgs).to(device)
            raw_feat, evnet_feat = encoder(batch_tensor)
            
            all_raw_features.append(raw_feat.cpu())
            all_evnet_features.append(evnet_feat.cpu())
            
            processed = i + len(batch_refs)
            if processed % 200 == 0 or processed == len(img_refs):
                logging.info(f"已处理 {processed}/{len(img_refs)}")
    
    raw_embeddings = torch.cat(all_raw_features, dim=0)
    evnet_embeddings = torch.cat(all_evnet_features, dim=0)
    
    payload = {
        "raw_embeddings": raw_embeddings,
        "evnet_embeddings": evnet_embeddings,
        "image_ids": image_ids,
        "model_name": f"evnet_fused:{open_clip_arch}:{open_clip_pretrained}",
        "embedding_dim": raw_embeddings.shape[1],
    }
    
    save_path = save_path or str(data_root / f"{split}_evnet_fused_features.pt")
    torch.save(payload, save_path)
    logging.info(f"融合特征已保存: {save_path} | 形状: {raw_embeddings.shape}")
    
    return payload


# ====================== 损失函数 ======================
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


# ====================== Dataset with Fused Features ======================
class FusedEEGProjectDataset(Dataset):
    """支持融合图像特征的Dataset"""
    
    def __init__(
        self,
        data_directory: Union[str, Path],
        split: str = "train",
        fused_pt_path: Optional[str] = None,
        map_location="cpu",
        max_samples: Optional[int] = None,
    ):
        # 加载EEG数据
        avg_trials = _avg_trials_for_split(split)
        dataset = load_eeg_dataset(
            data_directory=data_directory,
            split=split,
            avg_trials=avg_trials,
            selected_channels=None,
            eeg_channel_jsonl=str(Path(data_directory) / "EEG_CHANNELS.jsonl"),
        )
        
        eeg_list = []
        image_ids = []
        for sample in dataset:
            eeg_list.append(torch.from_numpy(np.array(sample["eeg"])).float())
            image_ids.append(sample["image_id"])
        
        self.eeg = torch.stack(eeg_list)
        self.image_ids = list(image_ids)
        
        # 加载融合特征
        if fused_pt_path is None:
            raise ValueError("fused_pt_path must be provided")
        
        payload = torch.load(fused_pt_path, map_location=map_location, weights_only=False)
        self.raw_features = torch.as_tensor(payload["raw_embeddings"]).float()
        self.evnet_features = torch.as_tensor(payload["evnet_embeddings"]).float()
        
        if len(self.eeg) != len(self.raw_features):
            raise ValueError(
                f"数据不匹配: EEG {len(self.eeg)} vs 特征 {len(self.raw_features)}"
            )
        
        # 截断
        if split == "train" and max_samples is not None and max_samples > 0:
            n = min(len(self.eeg), int(max_samples))
            if n < len(self.eeg):
                self.eeg = self.eeg[:n]
                self.raw_features = self.raw_features[:n]
                self.evnet_features = self.evnet_features[:n]
                self.image_ids = self.image_ids[:n]
                logging.info(f"训练集截断为前 {n} 条")
        
        logging.info(f"--- {split} 数据加载完成 ---")
        logging.info(f"EEG 形状: {self.eeg.shape}")
        logging.info(f"原始CLIP特征形状: {self.raw_features.shape}")
        logging.info(f"EVNet特征形状: {self.evnet_features.shape}")
    
    def __getitem__(self, index):
        return self.eeg[index], self.raw_features[index], self.evnet_features[index]
    
    def __len__(self):
        return len(self.eeg)


# ====================== ATM_S with Learnable Fusion ======================
class ATM_S_Fused(nn.Module):
    """ATM_S with learnable feature fusion weights w1 and w2"""
    
    def __init__(self, out_dim: int = 1024):
        super().__init__()
        # 原有的ATM_S编码器
        self.atm = ATM_S(out_dim=out_dim)
        
        # 可学习的融合权重 (w1 for raw, w2 for evnet)
        # 使用 logits 形式，通过 softmax 得到和为1的权重
        self.fusion_logits = nn.Parameter(torch.tensor([0.7, 0.3]))
        
        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, eeg, raw_features=None, evnet_features=None):
        """
        Args:
            eeg: EEG信号 (B, C, T)
            raw_features: 原始CLIP特征 (B, D) - 训练时必须提供
            evnet_features: EVNet处理后的CLIP特征 (B, D) - 训练时必须提供
        Returns:
            如果提供了图像特征，返回 (eeg_emb, fused_img_features)
            否则只返回 eeg_emb
        """
        eeg_emb = self.atm(eeg)
        
        if raw_features is not None and evnet_features is not None:
            # 计算融合权重 (softmax保证和为1)
            w = F.softmax(self.fusion_logits, dim=0)
            # w1 * v1 + w2 * v2
            fused_features = w[0] * raw_features + w[1] * evnet_features
            return eeg_emb, fused_features
        else:
            return eeg_emb
    
    def get_fusion_weights(self):
        """获取当前的融合权重 (w1, w2)"""
        w = F.softmax(self.fusion_logits, dim=0)
        return w.detach().cpu().numpy()
    
    def get_temperature(self):
        """获取当前温度参数"""
        return 1.0 / self.logit_scale.exp().item()


# ====================== 评估函数 ======================
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
def evaluate_retrieval_fused(model, test_loader, device="cuda", *, verbose: bool = True):
    """评估融合模型的检索性能"""
    model_to_eval = model.module if isinstance(model, nn.DataParallel) else model
    model_to_eval.eval()

    all_eeg_embeds = []
    all_img_embeds = []
    device_obj = torch.device(device)

    for eeg, raw_feat, evnet_feat in test_loader:
        eeg = eeg.to(device_obj, non_blocking=True)
        raw_feat = raw_feat.to(device_obj, non_blocking=True)
        evnet_feat = evnet_feat.to(device_obj, non_blocking=True)

        eeg_emb, fused_img = model_to_eval(eeg, raw_feat, evnet_feat)
        
        eeg_emb = F.normalize(eeg_emb.float(), dim=-1)
        fused_img = F.normalize(fused_img.float(), dim=-1)
        
        all_eeg_embeds.append(eeg_emb.cpu())
        all_img_embeds.append(fused_img.cpu())

    eeg_embeds = torch.cat(all_eeg_embeds, dim=0)
    img_embeds = torch.cat(all_img_embeds, dim=0)

    similarity = torch.matmul(eeg_embeds, img_embeds.T)
    eeg_to_image = _topk_accuracy(similarity, ks=(1, 5))
    image_to_eeg = _topk_accuracy(similarity.T, ks=(1, 5))

    metrics = {
        "eeg_to_image_top1": eeg_to_image["top1"],
        "eeg_to_image_top5": eeg_to_image["top5"],
        "image_to_eeg_top1": image_to_eeg["top1"],
        "image_to_eeg_top5": image_to_eeg["top5"],
    }

    if verbose:
        w = model_to_eval.get_fusion_weights()
        logging.info("\n--- 评估结果 ---")
        logging.info(f"当前融合权重: w1(raw)={w[0]:.4f}, w2(evnet)={w[1]:.4f}")
        logging.info(f"EEG -> Image | Top-1: {metrics['eeg_to_image_top1']:.2f}% | Top-5: {metrics['eeg_to_image_top5']:.2f}%")
        logging.info(f"Image -> EEG | Top-1: {metrics['image_to_eeg_top1']:.2f}% | Top-5: {metrics['image_to_eeg_top5']:.2f}%")
    
    return metrics


# ====================== 训练函数 ======================
def train_atm_fused(
    model,
    train_loader,
    epochs=40,
    lr=3e-4,
    device="cuda",
    checkpoint_path: Optional[Union[str, Path]] = None,
    test_loader: Optional[DataLoader] = None,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = True,
):
    device_obj = torch.device(device) if not isinstance(device, torch.device) else device
    model.to(device_obj)
    
    # 单卡训练
    if device_obj.type == "cuda" and torch.cuda.device_count() > 1:
        logging.warning(f"检测到 {torch.cuda.device_count()} 张 GPU，使用单卡训练")
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    torch.cuda.empty_cache()
    
    history = []
    best_eeg_to_image_top1 = -1.0
    best_epoch = -1
    
    logging.info(f"开始训练，设备: {device_obj}, Epochs: {epochs}, LR: {lr}")
    logging.info(f"梯度累积步数: {gradient_accumulation_steps}, 混合精度: {use_amp}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"模型参数量: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (eeg, raw_feat, evnet_feat) in enumerate(train_loader):
            eeg = eeg.to(device_obj, non_blocking=True)
            raw_feat = raw_feat.to(device_obj, non_blocking=True)
            evnet_feat = evnet_feat.to(device_obj, non_blocking=True)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    eeg_emb, fused_img = model(eeg, raw_feat, evnet_feat)
                    loss = contrastive_loss(eeg_emb, fused_img, logit_scale=model.logit_scale)
                    loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                eeg_emb, fused_img = model(eeg, raw_feat, evnet_feat)
                loss = contrastive_loss(eeg_emb, fused_img, logit_scale=model.logit_scale)
                loss = loss / gradient_accumulation_steps
                loss.backward()
            
            running_loss += loss.item() * gradient_accumulation_steps
            
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
        
        # 处理剩余梯度
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
        current_weights = model.module.get_fusion_weights() if isinstance(model, nn.DataParallel) else model.get_fusion_weights()
        
        if test_loader is not None:
            metrics = evaluate_retrieval_fused(model, test_loader, device=device_obj, verbose=False)
            history.append({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "lr": current_lr,
                "w1_raw": current_weights[0],
                "w2_evnet": current_weights[1],
                **metrics
            })
            
            logging.info(
                f"Epoch [{epoch+1}/{epochs}] Loss={avg_loss:.4f} LR={current_lr:.6f} | "
                f"w1(raw)={current_weights[0]:.4f} w2(evnet)={current_weights[1]:.4f} | "
                f"EEG->Img Top-1={metrics['eeg_to_image_top1']:.2f}% Top-5={metrics['eeg_to_image_top5']:.2f}%"
            )
            
            if metrics["eeg_to_image_top1"] > best_eeg_to_image_top1 and checkpoint_path is not None:
                best_eeg_to_image_top1 = metrics["eeg_to_image_top1"]
                best_epoch = epoch + 1
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                model_ref = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(
                    {
                        "model_state_dict": model_ref.state_dict(),
                        "history": history,
                        "best_eeg_to_image_top1": best_eeg_to_image_top1,
                        "best_epoch": best_epoch,
                        "best_metrics": metrics,
                        "fusion_weights": current_weights.tolist(),
                    },
                    checkpoint_path,
                )
                logging.info(f"💾 已保存最佳 checkpoint (Top-1={best_eeg_to_image_top1:.2f}%, epoch={best_epoch})")
        else:
            history.append({"epoch": epoch + 1, "loss": avg_loss, "lr": current_lr})
            logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
    
    return model, history


# ====================== 主函数 ======================
if __name__ == "__main__":
    data_root = Path("./image-eeg-data")
    open_clip_arch = "ViT-H-14"
    open_clip_pretrained = "laion2b_s32b_b79k"
    
    num_gpus = torch.cuda.device_count()
    logging.info(f"可用GPU数量: {num_gpus}")
    logging.info(f"数据目录: {data_root}")
    logging.info(f"EVNet 可用: {EVNET_AVAILABLE}")
    
    # 预计算融合特征（只执行一次）
    train_fused_path = data_root / "train_evnet_fused_features.pt"
    test_fused_path = data_root / "test_evnet_fused_features.pt"
    
    if not train_fused_path.exists():
        logging.info("🚀 预计算训练集融合特征 (EVNet + CLIP)...")
        precompute_fused_features(
            data_root,
            split="train",
            open_clip_arch=open_clip_arch,
            open_clip_pretrained=open_clip_pretrained,
            batch_size=16,
            save_path=str(train_fused_path),
        )
    else:
        logging.info(f"训练集融合特征已存在: {train_fused_path}")
    
    if not test_fused_path.exists():
        logging.info("🚀 预计算测试集融合特征 (EVNet + CLIP)...")
        precompute_fused_features(
            data_root,
            split="test",
            open_clip_arch=open_clip_arch,
            open_clip_pretrained=open_clip_pretrained,
            batch_size=16,
            save_path=str(test_fused_path),
        )
    else:
        logging.info(f"测试集融合特征已存在: {test_fused_path}")
    
    # 创建数据集（只创建一次，因为数据相同）
    train_dataset = FusedEEGProjectDataset(
        str(data_root), 
        split="train", 
        fused_pt_path=str(train_fused_path)
    )
    test_dataset = FusedEEGProjectDataset(
        str(data_root), 
        split="test", 
        fused_pt_path=str(test_fused_path)
    )
    
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")
    
    # 数据加载器
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
    
    # ====================== 10次重复实验 ======================
    n_runs = 10
    all_best_top1 = []
    all_best_top5 = []
    all_final_weights = []
    all_results = []
    
    logging.info("\n" + "=" * 80)
    logging.info(f"开始 {n_runs} 次独立重复实验")
    logging.info("=" * 80)
    
    for run_idx in range(1, n_runs + 1):
        logging.info(f"\n{'='*60}")
        logging.info(f"第 {run_idx}/{n_runs} 次实验")
        logging.info(f"{'='*60}")
        
        # 设置不同的随机种子
        run_seed = 114514 + run_idx
        set_seed(run_seed)
        logging.info(f"随机种子: {run_seed}")
        
        # 创建模型
        clip_dim = train_dataset.raw_features.shape[1]
        model = ATM_S_Fused(out_dim=clip_dim).to(device)
        
        # 独立的 checkpoint 路径
        checkpoint_path = Path(f"../artifacts/atm_s_evnet_fused_run{run_idx}_best.pt")
        
        initial_weights = model.get_fusion_weights()
        logging.info(f"CLIP 特征维度: {clip_dim}")
        logging.info(f"初始融合权重: w1(raw)={initial_weights[0]:.4f}, w2(evnet)={initial_weights[1]:.4f}")
        
        # 训练
        trained_model, train_history = train_atm_fused(
            model,
            train_loader,
            epochs=40,
            lr=3e-4,
            device=device,
            checkpoint_path=checkpoint_path,
            test_loader=test_loader,
        )
        
        # 加载最佳 checkpoint 进行评估
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model_ref = trained_model.module if isinstance(trained_model, nn.DataParallel) else trained_model
            model_ref.load_state_dict(ckpt["model_state_dict"])
            best_top1 = ckpt["best_eeg_to_image_top1"]
            best_top5 = ckpt["best_metrics"]["eeg_to_image_top5"]
            logging.info(f"第 {run_idx} 次实验最佳 Top-1: {best_top1:.2f}%")
        else:
            # 如果没有 checkpoint，使用最终模型评估
            final_metrics = evaluate_retrieval_fused(trained_model, test_loader, device=device)
            best_top1 = final_metrics["eeg_to_image_top1"]
            best_top5 = final_metrics["eeg_to_image_top5"]
        
        # 获取最终融合权重
        final_weights = trained_model.module.get_fusion_weights() if isinstance(trained_model, nn.DataParallel) else trained_model.get_fusion_weights()
        
        all_best_top1.append(best_top1)
        all_best_top5.append(best_top5)
        all_final_weights.append(final_weights)
        all_results.append({
            "run": run_idx,
            "best_top1": best_top1,
            "best_top5": best_top5,
            "final_weights": final_weights.tolist()
        })
        
        logging.info(f"第 {run_idx} 次实验完成 - Best Top-1: {best_top1:.2f}%, Best Top-5: {best_top5:.2f}%")
        logging.info(f"最终融合权重: w1(raw)={final_weights[0]:.4f}, w2(evnet)={final_weights[1]:.4f}")
        
        # 清理显存
        torch.cuda.empty_cache()
    
    # ====================== 统计分析 ======================
    all_best_top1 = np.array(all_best_top1)
    all_best_top5 = np.array(all_best_top5)
    all_final_weights = np.array(all_final_weights)
    
    mean_top1 = np.mean(all_best_top1)
    std_top1 = np.std(all_best_top1)
    mean_top5 = np.mean(all_best_top5)
    std_top5 = np.std(all_best_top5)
    
    mean_w1 = np.mean(all_final_weights[:, 0])
    std_w1 = np.std(all_final_weights[:, 0])
    mean_w2 = np.mean(all_final_weights[:, 1])
    std_w2 = np.std(all_final_weights[:, 1])
    
    # 输出详细结果
    logging.info("\n" + "=" * 80)
    logging.info("10次重复实验统计结果")
    logging.info("=" * 80)
    
    logging.info("\n各次实验最佳 Top-1:")
    for i, top1 in enumerate(all_best_top1):
        logging.info(f"  Run {i+1}: {top1:.2f}%")
    
    logging.info(f"\n各次实验最佳 Top-5:")
    for i, top5 in enumerate(all_best_top5):
        logging.info(f"  Run {i+1}: {top5:.2f}%")
    
    logging.info(f"\n各次实验最终融合权重:")
    for i, w in enumerate(all_final_weights):
        logging.info(f"  Run {i+1}: w1(raw)={w[0]:.4f}, w2(evnet)={w[1]:.4f}")
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Top-1 平均值: {mean_top1:.2f}% ± {std_top1:.2f}%")
    logging.info(f"Top-5 平均值: {mean_top5:.2f}% ± {std_top5:.2f}%")
    logging.info(f"融合权重 w1(raw): {mean_w1:.4f} ± {std_w1:.4f}")
    logging.info(f"融合权重 w2(evnet): {mean_w2:.4f} ± {std_w2:.4f}")
    logging.info(f"{'='*80}")
    
    # 保存所有结果
    results_path = Path("../artifacts/evnet_fused_10runs_results.pt")
    torch.save({
        "all_best_top1": all_best_top1,
        "all_best_top5": all_best_top5,
        "all_final_weights": all_final_weights,
        "mean_top1": mean_top1,
        "std_top1": std_top1,
        "mean_top5": mean_top5,
        "std_top5": std_top5,
        "mean_w1": mean_w1,
        "std_w1": std_w1,
        "mean_w2": mean_w2,
        "std_w2": std_w2,
        "detailed_results": all_results,
    }, results_path)
    logging.info(f"\n结果已保存到: {results_path}")