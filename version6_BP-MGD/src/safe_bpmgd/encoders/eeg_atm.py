from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.net(x.mean(dim=-1)).unsqueeze(-1)
        return x * weights


class ATMEEGEncoder(nn.Module):
    """Compact ATM-style EEG encoder for [B, 63, 250] signals."""

    def __init__(self, channels: int = 63, timesteps: int = 250, hidden_dim: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=7, padding=3, groups=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 384, kernel_size=5, padding=2, groups=4),
            nn.BatchNorm1d(384),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(384),
            nn.Linear(384, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected EEG [B,C,T], got {tuple(x.shape)}")
        x = self.channel_attention(x.float())
        h = self.temporal(x)
        return self.project(self.pool(h))
