"""
Generate multi-blur CLIP (RN50) image features for course data.
Added EVNet frontend processing option.
Adapted from process_image.py: Linux paths, course image dirs.
"""
import torch
import cv2
from PIL import Image
import numpy as np
import torch.nn.functional as F
import os
import argparse
from torchvision import transforms
import open_clip
import torch.nn as nn
import tqdm
import sys
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DL_PROJECT_ROOT = os.path.dirname(REPO_DIR)
_AUTO_BASE_PATH = os.path.join(
    _DL_PROJECT_ROOT, "image-eeg-data", "converted_for_cogcappro", "ThingsEEG", "Image_set_Resize"
)
BASE_PATH = _AUTO_BASE_PATH if os.path.isdir(_AUTO_BASE_PATH) else os.path.join(REPO_DIR, "data", "things-eeg", "Image_set")
DEFAULT_RN50_WEIGHTS = os.environ.get(
    "VED_RN50_WEIGHTS",
    os.path.join(REPO_DIR, "data", "weights", "open_clip_pytorch_model.bin"),
)
DEFAULT_SAVE_DIR = os.path.join(REPO_DIR, "output", "Image_feature")

# ====================== 导入 EVNet ======================
EVNET_AVAILABLE = False
try:
    # 假设 evnet 文件夹在当前目录或项目根目录
    evnet_paths = [
        # Path(__file__).parent / "evnet",           # 同级目录下的 evnet 文件夹
        # Path(__file__).parent.parent / "evnet",    # 上级目录下的 evnet 文件夹
        Path("./evnet/evnet"),                     # 绝对路径（根据需要添加）
    ]

    for evnet_path in evnet_paths:
        if evnet_path.exists():
            # 将 evnet 的父目录添加到 sys.path，而不是 evnet 本身
            sys.path.insert(0, str(evnet_path.parent))
            print(f"Adding to path: {evnet_path.parent}")
            
            # 然后使用包形式的导入
            from evnet import EVNet
            from evnet.modules import SubcorticalBlock, VOneBlock
            from evnet.params import get_tuned_params, generate_gabor_param
            
            EVNET_AVAILABLE = True
            print(f"EVNet loaded from: {evnet_path}")
            break
        
    if not EVNET_AVAILABLE:
        raise ImportError("EVNet not found")
except ImportError as e:
    print(f"Warning: EVNet not available: {e}")
    print("EVNet features will not be generated. Only standard blur features will be available.")


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
        colors_p_cells: list = None,
        with_light_adapt: bool = True,
        with_dog: bool = True,
        with_contrast_norm: bool = True,
        with_relu: bool = False,
        subcort_noise_mode: str = None,
        subcort_fano_factor: float = 0.4,
        # VOneBlock 参数
        simple_channels: int = 256,
        complex_channels: int = 256,
        vone_noise_mode: str = None,
        vone_ksize: int = 31,
        vone_stride: int = 4,
        sf_corr: float = 0.75,
        sf_max: float = 8.0,
        sf_min: float = 0.0,
        gabor_seed: int = 0,
    ):
        super().__init__()
        
        if colors_p_cells is None:
            colors_p_cells = ['r/g', 'g/r', 'b/y']
        
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
        
        # 生成 Gabor 参数 - 注意参数名是 rand_flag 不是 rand_param
        vone_in_channels = p_channels + m_channels
        sf, theta, phase, nx, ny, color = generate_gabor_param(
            simple_channels,                           # n_sc
            complex_channels,                          # n_cc
            seed=gabor_seed,                          # seed
            rand_flag=False,                          # rand_flag (不是 rand_param)
            sf_corr=sf_corr,
            sf_max=sf_max,
            sf_min=sf_min,
            diff_n=False,
            dnstd=0.22,
            in_channels=vone_in_channels,
            color_prob=None,                          # color_prob
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
        
        # 添加一个适配层，将 EVNet 输出特征图转换为 ResNet50 期望的输入格式
        # EVNet 输出: [B, simple_channels+complex_channels, H/stride, W/stride] = [B, 512, 56, 56]
        # 需要转换为 [B, 3, 224, 224] 给 ResNet50
        self.adapter = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
        )
        
        # 适配层固定（也可以设置为可学习，但这里为了保持前端固定，冻结它）
        for param in self.adapter.parameters():
            param.requires_grad = False
        
        self.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 EVNet 处理并适配后的图像 [B, 3, 224, 224]"""
        x = self.subcortical(x)
        x = self.voneblock(x)
        x = self.adapter(x)
        return x


class EVNetFrontEnd_Xavier(EVNetFrontEnd):
    """EVNetFrontEnd with Xavier uniform initialization for adapter Conv2d layers (still frozen)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for m in self.adapter.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for param in self.adapter.parameters():
            param.requires_grad = False
        self.eval()


class EVNetGAPFrontEnd(nn.Module):
    """EVNet: SubcorticalBlock + VOneBlock + GlobalAvgPool + Linear(512→out_dim).
    Bypasses RN50 entirely. Outputs out_dim-dim feature vector. All weights frozen (Linear: Xavier init)."""

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 224,
        visual_degrees: float = 7.0,
        p_channels: int = 3,
        m_channels: int = 0,
        colors_p_cells: list = None,
        with_light_adapt: bool = True,
        with_dog: bool = True,
        with_contrast_norm: bool = True,
        with_relu: bool = False,
        subcort_noise_mode: str = None,
        subcort_fano_factor: float = 0.4,
        simple_channels: int = 256,
        complex_channels: int = 256,
        vone_noise_mode: str = None,
        vone_ksize: int = 31,
        vone_stride: int = 4,
        sf_corr: float = 0.75,
        sf_max: float = 8.0,
        sf_min: float = 0.0,
        gabor_seed: int = 0,
        out_dim: int = 1024,
    ):
        super().__init__()

        if colors_p_cells is None:
            colors_p_cells = ['r/g', 'g/r', 'b/y']

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

        vone_in_channels = p_channels + m_channels
        sf, theta, phase, nx, ny, color = generate_gabor_param(
            simple_channels, complex_channels,
            seed=gabor_seed, rand_flag=False,
            sf_corr=sf_corr, sf_max=sf_max, sf_min=sf_min,
            diff_n=False, dnstd=0.22,
            in_channels=vone_in_channels, color_prob=None,
        )

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
            noise_scale=1, noise_level=0, fano_factor=1,
            simple_channels=simple_channels,
            complex_channels=complex_channels,
            ksize=vone_ksize,
            stride=vone_stride,
            input_size=image_size
        )

        for param in self.subcortical.parameters():
            param.requires_grad = False
        for param in self.voneblock.parameters():
            param.requires_grad = False

        evnet_out_channels = simple_channels + complex_channels  # 512
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(evnet_out_channels, out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        for param in self.proj.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, out_dim] feature vector."""
        x = self.subcortical(x)
        x = self.voneblock(x)
        x = self.gap(x).flatten(1)
        x = self.proj(x)
        return x


class EVNetGAPEncoder:
    """Encodes images using EVNetGAPFrontEnd (no RN50). Outputs out_dim-dim features."""

    def __init__(self, evnet_gap_frontend, process_transform, batch_size=128):
        self.evnet_gap = evnet_gap_frontend
        self.process_transform = process_transform
        self.batch_size = batch_size
        self.device = next(evnet_gap_frontend.parameters()).device

    @torch.no_grad()
    def encode_images(self, image_paths):
        """Encode images using EVNet+GAP+Linear pipeline."""
        self.evnet_gap.eval()
        image_features_list = []

        for i in tqdm.tqdm(range(0, len(image_paths), self.batch_size), desc="EVNet+GAP encoding"):
            batch = image_paths[i:i + self.batch_size]
            imgs = []
            for rel_path in batch:
                full_path = os.path.join(BASE_PATH, rel_path)
                img = Image.open(full_path).convert("RGB").resize((224, 224))
                img_tensor = self.process_transform(img).unsqueeze(0).to(self.device)
                imgs.append(img_tensor)
            inputs = torch.cat(imgs, dim=0)
            feats = self.evnet_gap(inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            image_features_list.append(feats.float().cpu())

        image_features = torch.cat(image_features_list, dim=0)
        return {image_paths[i]: image_features[i] for i in range(len(image_paths))}


class EVNetPreprocessedImageEncoder:
    """
    使用 EVNet 前端预处理图像，然后用 RN50 编码
    """
    def __init__(self, vlmodel, evnet_frontend, process_transform, batch_size=128):
        self.vlmodel = vlmodel
        self.evnet_frontend = evnet_frontend
        self.process_transform = process_transform
        self.batch_size = batch_size
        
        # 确保模型在正确的设备上
        if hasattr(vlmodel, 'parameters'):
            self.device = next(vlmodel.parameters()).device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @torch.no_grad()
    def encode_images(self, image_paths):
        """对图像应用 EVNet 前端，然后用 RN50 编码"""
        self.vlmodel.eval()
        self.evnet_frontend.eval()
        
        image_features_list = []
        
        for i in tqdm.tqdm(range(0, len(image_paths), self.batch_size), desc="EVNet+RN50 encoding"):
            batch = image_paths[i:i + self.batch_size]
            imgs = []
            
            for rel_path in batch:
                full_path = os.path.join(BASE_PATH, rel_path)
                # 加载原始图像
                img = Image.open(full_path).convert("RGB").resize((224, 224))
                img_tensor = self.process_transform(img).unsqueeze(0).to(self.device)
                imgs.append(img_tensor)
            
            inputs = torch.cat(imgs, dim=0)
            
            # 通过 EVNet 前端处理
            evnet_processed = self.evnet_frontend(inputs)
            
            # 通过 RN50 编码
            feats = self.vlmodel.encode_image(evnet_processed)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            image_features_list.append(feats.float().cpu())
        
        image_features = torch.cat(image_features_list, dim=0)
        return {image_paths[i]: image_features[i] for i in range(len(image_paths))}


class BlurringPipeline:
    def __init__(self, blur_kernel_size):
        self.blur_kernel_size = blur_kernel_size

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)
        img_np = np.array(img)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_blur = cv2.GaussianBlur(img_np, (self.blur_kernel_size, self.blur_kernel_size), 0)
        img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_blur)


BACKBONE_MODEL_NAME = {
    'rn50':     'RN50',
    'vit_h_14': 'ViT-H-14',
}


class Make_dataset(nn.Module):
    def __init__(self, checkpoint_path, batch_size=128, use_evnet=True, evnet_mode='random', backbone='rn50'):
        super().__init__()
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Missing checkpoint. Tried: {checkpoint_path}"
            )

        self.batch_size = batch_size
        model_name = BACKBONE_MODEL_NAME.get(backbone, 'RN50')
        self.vlmodel, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=checkpoint_path
        )
        self.vlmodel.eval()
        
        # 将模型移到设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vlmodel = self.vlmodel.to(self.device)

        # 模糊变换
        self.blur_transform = {}
        blur_kernels = [1, 3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63]
        blur_tags = ['1', '3', '9', '15', '21', '27', '33', '39', '45', '51', '57', '63']
        
        for kernel, tag in zip(blur_kernels, blur_tags):
            self.blur_transform[tag] = BlurringPipeline(kernel)

        # 图像预处理：转为 tensor + 归一化
        process_terms = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ]
        self.process_transform = transforms.Compose(process_terms)
        
        # EVNet 前端编码器
        self.use_evnet = use_evnet and EVNET_AVAILABLE
        if self.use_evnet:
            print(f"Initializing EVNet frontend (mode={evnet_mode})...")
            if evnet_mode == 'gap':
                evnet_frontend = EVNetGAPFrontEnd(
                    in_channels=3, image_size=224, visual_degrees=7.0,
                ).to(self.device)
                self.evnet_encoder = EVNetGAPEncoder(
                    evnet_frontend, self.process_transform, batch_size
                )
            elif evnet_mode == 'xavier':
                evnet_frontend = EVNetFrontEnd_Xavier(
                    in_channels=3, image_size=224, visual_degrees=7.0,
                ).to(self.device)
                self.evnet_encoder = EVNetPreprocessedImageEncoder(
                    self.vlmodel, evnet_frontend, self.process_transform, batch_size
                )
            else:
                evnet_frontend = EVNetFrontEnd(
                    in_channels=3, image_size=224, visual_degrees=7.0,
                ).to(self.device)
                self.evnet_encoder = EVNetPreprocessedImageEncoder(
                    self.vlmodel, evnet_frontend, self.process_transform, batch_size
                )
            print(f"EVNet frontend initialized (mode={evnet_mode}).")
        else:
            print("EVNet not available or disabled. Only standard blur features will be generated.")
            self.evnet_encoder = None

    @torch.no_grad()
    def ImageEncoder(self, image_paths, blur_transform):
        """标准 blur + RN50 编码"""
        self.vlmodel.eval()
        image_features_list = []
        dev = self.device

        for i in tqdm.tqdm(range(0, len(image_paths), self.batch_size), desc="Standard blur encoding"):
            batch = image_paths[i:i + self.batch_size]
            imgs = []
            for rel_path in batch:
                full_path = os.path.join(BASE_PATH, rel_path)
                img = Image.open(full_path).convert("RGB").resize((224, 224))
                imgs.append(self.process_transform(blur_transform(img)))
            inputs = torch.stack(imgs).to(dev)
            feats = self.vlmodel.encode_image(inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            image_features_list.append(feats.float().cpu())

        image_features = torch.cat(image_features_list, dim=0)
        return {image_paths[i]: image_features[i] for i in range(len(image_paths))}
    
    @torch.no_grad()
    def EvnetImageEncoder(self, image_paths):
        """EVNet 前端 + RN50 编码（无模糊，因为 EVNet 本身有生物视觉处理）"""
        if not self.use_evnet:
            raise RuntimeError("EVNet is not available. Cannot encode with EVNet.")
        return self.evnet_encoder.encode_images(image_paths)


def collect_image_paths(split_dir):
    """返回 Image_set/ 中 split_dir 下的相对路径列表（排序后）"""
    rel_paths = []
    split_full = os.path.join(BASE_PATH, split_dir)
    if not os.path.exists(split_full):
        print(f"Warning: Directory not found: {split_full}")
        return rel_paths
    
    for cat in sorted(os.listdir(split_full)):
        cat_dir = os.path.join(split_full, cat)
        if not os.path.isdir(cat_dir):
            continue
        for fname in sorted(os.listdir(cat_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                rel_paths.append(os.path.join(split_dir, cat, fname))
    return rel_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_checkpoint",
        default=DEFAULT_RN50_WEIGHTS,
        help="Path to OpenCLIP RN50 checkpoint, usually open_clip_pytorch_model.bin.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--save_dir",
        default=DEFAULT_SAVE_DIR,
        help="Directory for generated MultiBlur_RN50_train/test.pt files.",
    )
    parser.add_argument(
        "--no_evnet",
        action="store_true",
        help="Disable EVNet frontend processing (only generate blur features).",
    )
    parser.add_argument(
        "--evnet_mode",
        type=str,
        default="random",
        choices=["random", "xavier", "gap"],
        help="EVNet adapter mode: 'random' (Kaiming Conv2d+backbone), 'xavier' (Xavier Conv2d+backbone), 'gap' (GlobalAvgPool+Linear, no backbone).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="rn50",
        choices=["rn50", "vit_h_14"],
        help="Visual backbone: 'rn50' (ResNet50, 1024-dim) or 'vit_h_14' (ViT-H/14, 1024-dim).",
    )
    parser.add_argument(
        "--image_set_dir",
        default=None,
        help="Override the image set root directory (BASE_PATH). "
             "Use this for grey background or other preprocessed image variants. "
             "The directory must contain train_images/ and test_images/ subdirectories.",
    )
    args = parser.parse_args()

    if args.image_set_dir:
        BASE_PATH = os.path.abspath(args.image_set_dir)
        print(f"Image set dir overridden: {BASE_PATH}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 根据 backbone 和 evnet_mode 确定输出文件前缀
    backbone_tag = {'rn50': 'RN50', 'vit_h_14': 'ViTH14'}[args.backbone]
    blur_prefix = f'MultiBlur_{backbone_tag}'
    evnet_prefix_map = {
        'random': f'EVNet_{backbone_tag}',
        'xavier': f'EVNet_xavier_{backbone_tag}',
        'gap':    'EVNet_gap',
    }
    evnet_prefix = evnet_prefix_map[args.evnet_mode]

    train_save = os.path.join(args.save_dir, f"{blur_prefix}_train.pt")
    test_save  = os.path.join(args.save_dir, f"{blur_prefix}_test.pt")
    train_evnet_save = os.path.join(args.save_dir, f"{evnet_prefix}_train.pt")
    test_evnet_save  = os.path.join(args.save_dir, f"{evnet_prefix}_test.pt")

    print(f"Using checkpoint: {args.clip_checkpoint}")
    print(f"Backbone: {args.backbone}  blur_prefix: {blur_prefix}  evnet_prefix: {evnet_prefix}")
    print(f"Saving image features to: {args.save_dir}")
    print(f"EVNet enabled: {not args.no_evnet and EVNET_AVAILABLE}, mode: {args.evnet_mode}")

    # 创建模型实例
    model = Make_dataset(args.clip_checkpoint, args.batch_size,
                         use_evnet=(not args.no_evnet), evnet_mode=args.evnet_mode,
                         backbone=args.backbone)
    model = model.to(device)
    
    # 测试模型是否工作
    print(f"Model device: {next(model.vlmodel.parameters()).device}")
    test_input = torch.randn(1, 3, 224, 224).to(device)
    test_output = model.vlmodel.encode_image(test_input)
    print(f"Test encoding shape: {test_output.shape}")

    blur_keys = ['1', '3', '9', '15', '21', '27', '33', '39', '45', '51', '57', '63']

    # ==================== 训练集特征 ====================
    train_paths = None
    
    # 标准模糊特征
    if os.path.exists(train_save):
        print(f"Train features already exist: {train_save}")
    else:
        print("Collecting training image paths...")
        train_paths = collect_image_paths("train_images")
        print(f"Found {len(train_paths)} training images")
        train_features = {}
        for key in blur_keys:
            print(f"Encoding blur level {key}...")
            train_features[key] = model.ImageEncoder(train_paths, model.blur_transform[key])
        torch.save(train_features, train_save)
        print(f"Saved: {train_save}")
    
    # EVNet 特征（不需要模糊，因为 EVNet 本身有生物视觉处理）
    if model.use_evnet:
        if os.path.exists(train_evnet_save):
            print(f"Train EVNet features already exist: {train_evnet_save}")
        else:
            if train_paths is None:
                train_paths = collect_image_paths("train_images")
                print(f"Found {len(train_paths)} training images")
            print("Encoding with EVNet frontend + RN50...")
            evnet_features = model.EvnetImageEncoder(train_paths)
            torch.save(evnet_features, train_evnet_save)
            print(f"Saved EVNet features: {train_evnet_save}")

    # ==================== 测试集特征 ====================
    test_paths = None
    
    # 标准模糊特征
    if os.path.exists(test_save):
        print(f"Test features already exist: {test_save}")
    else:
        print("Collecting test image paths...")
        test_paths = collect_image_paths("test_images")
        print(f"Found {len(test_paths)} test images")
        test_features = {}
        for key in blur_keys:
            print(f"Encoding blur level {key}...")
            test_features[key] = model.ImageEncoder(test_paths, model.blur_transform[key])
        torch.save(test_features, test_save)
        print(f"Saved: {test_save}")
    
    # EVNet 特征
    if model.use_evnet:
        if os.path.exists(test_evnet_save):
            print(f"Test EVNet features already exist: {test_evnet_save}")
        else:
            if test_paths is None:
                test_paths = collect_image_paths("test_images")
                print(f"Found {len(test_paths)} test images")
            print("Encoding test set with EVNet frontend + RN50...")
            evnet_features = model.EvnetImageEncoder(test_paths)
            torch.save(evnet_features, test_evnet_save)
            print(f"Saved EVNet features: {test_evnet_save}")

    print("All done.")