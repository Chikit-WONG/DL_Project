from __future__ import annotations

from torch import nn

from safe_bpmgd.encoders.eeg_atm import ATMEEGEncoder
from safe_bpmgd.encoders.heads import MultiModalHeads


class SafeBPMGDEEGModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        model_cfg = cfg.get("model", cfg)
        self.encoder = ATMEEGEncoder(
            channels=model_cfg.get("channels", 63),
            timesteps=model_cfg.get("timesteps", 250),
            hidden_dim=model_cfg.get("hidden_dim", 1024),
            dropout=model_cfg.get("dropout", 0.1),
        )
        self.heads = MultiModalHeads(
            hidden_dim=model_cfg.get("hidden_dim", 1024),
            semantic_dim=model_cfg.get("semantic_dim", 1024),
            struct_dim=model_cfg.get("struct_dim", 256),
            edge_dim=model_cfg.get("edge_dim", 512),
            depth_dim=model_cfg.get("depth_dim", 512),
            vae_dim=model_cfg.get("vae_dim", 512),
            num_blur_levels=model_cfg.get("num_blur_levels", 6),
        )

    def forward(self, eeg):
        return self.heads(self.encoder(eeg))
