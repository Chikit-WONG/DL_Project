import torch
import torch.nn as nn


class ATM_S(nn.Module):
    """EEG ShallowNet + 通道自注意力，对齐 CLIP 1024 维隐藏空间 (ViT-L/14 ln_post 输出)。"""

    def __init__(
        self,
        num_channels: int = 63,
        time_len: int = 250,
        out_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        if time_len % 5 != 0:
            raise ValueError("time_len must be divisible by num_heads=5 for MultiheadAttention.")
        # Layer 1: [B,63,250] 视为 seq=63, embed=250
        self.chan_attn = nn.MultiheadAttention(
            embed_dim=time_len, num_heads=5, batch_first=True
        )
        # Layer 2: Temporal-spatial ShallowNet
        self.conv_time = nn.Conv2d(1, 40, kernel_size=(1, 25))
        self.conv_spatial = nn.Conv2d(40, 40, kernel_size=(num_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        # Layer 3: 40*11=440 -> 1024
        self.proj = nn.Sequential(
            nn.Linear(40 * 11, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07, dtype=torch.float32)))

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        # eeg: [B, 63, 250]
        attn_out, _ = self.chan_attn(eeg, eeg, eeg, need_weights=False)
        x = eeg + attn_out
        x = x.unsqueeze(1)
        x = self.conv_time(x)
        x = self.conv_spatial(x)
        x = self.bn(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = x.flatten(1)
        return self.proj(x)