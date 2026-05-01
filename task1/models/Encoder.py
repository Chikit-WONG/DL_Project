# Encoder.py
import torch.nn as nn
import torch
import scipy.signal as signal
import numpy as np
# from utils import *


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class Conv2dWithAbs(nn.Conv2d):

    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithAbs, self).__init__(*args, **kwargs)

    def forward(self, x):
        
        if self.doWeightNorm: 
            # self.weight.data = torch.renorm(
            #     self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            # )

            self.weight.data = torch.abs(
                self.weight.data
            )
        return super(Conv2dWithAbs, self).forward(x)


class Brain_Visual_Encoder_EEG(nn.Module):
    """支持 EVNet 特征融合的 EEG 编码器"""
    def __init__(self, channels=63, proj_dim=1152, temporal_len=250, use_evnet=False):
        super(Brain_Visual_Encoder_EEG, self).__init__()

        self.temporal_len   = temporal_len
        self.embed_channels = channels
        self.embed_dim      = 200
        self.use_evnet = use_evnet
        self.proj_dim = proj_dim

        self.eeg_encoder = nn.Sequential(
            nn.Linear(self.temporal_len, self.embed_dim),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Linear(self.embed_dim,   self.embed_dim),
            nn.ELU(),
            nn.Dropout(0.65),
        )

        self.spatial_conv = Conv2dWithAbs(1, 25, kernel_size=(channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(25)
    
        # 图像适配器（用于多尺度模糊特征）
        self.img_adapter = nn.Sequential(
            nn.Linear(proj_dim, 768),
            nn.ELU(),
            nn.Dropout(0.85),
            nn.Linear(768, proj_dim),
        )
        
        # EVNet 特征的适配器（可选）
        if use_evnet:
            self.evnet_adapter = nn.Sequential(
                nn.Linear(proj_dim, 768),
                nn.ELU(),
                nn.Dropout(0.85),
                nn.Linear(768, proj_dim),
            )
            # 可学习的融合权重
            self.fusion_logits = nn.Parameter(torch.tensor([0.7, 0.3]))

        self.eeg_adapter = nn.Sequential(
            nn.Linear(25 * self.embed_dim, proj_dim),
        )

        self.fusion_adapter = nn.Sequential(
            nn.Linear(proj_dim, 768),
            nn.ELU(),
            nn.Dropout(0.85),
            nn.Linear(768, proj_dim),
        )

        self.learned_scale = nn.Parameter(torch.rand([1, 50, proj_dim]), requires_grad=True)
        self.default_feature = nn.Parameter(torch.zeros([1, 4, proj_dim]), requires_grad=True)
        self.softplus = nn.Softplus()

    def get_image_feature(self, imgs, evnet_feat=None):
        """
        获取融合后的图像特征
        imgs: [B, num_blur_levels, D] 多尺度模糊特征
        evnet_feat: [B, D] EVNet 处理后的特征（可选）
        """
        # 处理多尺度模糊特征
        imgs = torch.cat([imgs, self.default_feature.expand(imgs.shape[0], -1, -1)], 1)
        rates = torch.softmax(self.learned_scale[:, :imgs.shape[1]], -2)
        blur_agg = torch.sum(imgs * rates, 1)  # [B, proj_dim]
        
        if self.use_evnet and evnet_feat is not None:
            # 先融合 blur_agg 和 evnet_feat（融合前都不经过各自的 MLP）
            w = torch.softmax(self.fusion_logits, dim=0)
            fused_agg = w[0] * blur_agg + w[1] * evnet_feat  # [B, proj_dim]
            # 融合后再统一过一个 MLP
            fused_feat = self.fusion_adapter(fused_agg)
            return fused_feat
        else:
            # 没有 EVNet 时，blur_agg 过原来的 img_adapter
            blur_feat = self.img_adapter(blur_agg)
            return blur_feat

    def get_fusion_weights(self):
        """获取当前融合权重（仅当 use_evnet=True 时）"""
        if self.use_evnet:
            w = torch.softmax(self.fusion_logits, dim=0)
            return w.detach().cpu().numpy()
        else:
            return np.array([1.0, 0.0])

    def forward(self, x):
        x = self.spatial_conv(x[:, None])
        x = self.bn(x)
        x = self.eeg_encoder(x)
        x = self.eeg_adapter(x.flatten(1))
        return x

    
class Brain_Visual_Encoder_MEG(nn.Module):
    def __init__(self, channels=63, proj_dim=1152, temporal_len=250, use_evnet=False):
        super(Brain_Visual_Encoder_MEG, self).__init__()

        self.temporal_len   = temporal_len
        self.embed_channels = channels
        self.embed_dim      = 150
        self.use_evnet = use_evnet
        self.proj_dim = proj_dim

        self.eeg_encoder = nn.Sequential(
            nn.Linear(self.temporal_len, self.embed_dim),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ELU(),
            nn.Dropout(0.65),
        )

        self.spatial_conv = nn.Conv2d(1, 50, kernel_size=(channels, 1), bias=False)

        self.eeg_adapter = nn.Sequential(
            nn.Linear(50 * self.embed_dim, proj_dim),
        )

        self.img_adapter = nn.Sequential(
            nn.Linear(proj_dim, 1024),
            nn.Dropout(0.85),
            nn.ELU(),
            nn.Linear(1024, proj_dim),
        )
        
        # EVNet 特征的适配器
        if use_evnet:
            self.evnet_adapter = nn.Sequential(
                nn.Linear(proj_dim, 1024),
                nn.Dropout(0.85),
                nn.ELU(),
                nn.Linear(1024, proj_dim),
            )
            self.fusion_logits = nn.Parameter(torch.tensor([0.5, 0.5]))

        self.learned_scale = nn.Parameter(torch.rand([1, 50, proj_dim]), requires_grad=True)
        self.default_feature = nn.Parameter(torch.zeros([1, 3, proj_dim]), requires_grad=True)

    def get_image_feature(self, imgs, evnet_feat=None):
        """获取融合后的图像特征"""
        imgs = torch.cat([imgs, self.default_feature.expand(imgs.shape[0], -1, -1)], 1)
        rates = torch.softmax(self.learned_scale[:, :imgs.shape[1]], -2)
        blur_agg = torch.sum(imgs * rates, 1)
        blur_feat = self.img_adapter(blur_agg)
        
        if self.use_evnet and evnet_feat is not None:
            evnet_feat = self.evnet_adapter(evnet_feat)
            w = torch.softmax(self.fusion_logits, dim=0)
            return w[0] * blur_feat + w[1] * evnet_feat
        else:
            return blur_feat
    
    def get_fusion_weights(self):
        if self.use_evnet:
            w = torch.softmax(self.fusion_logits, dim=0)
            return w.detach().cpu().numpy()
        else:
            return np.array([1.0, 0.0])

    def forward(self, x):
        x = self.spatial_conv(x[:, None])
        x = self.eeg_encoder(x)
        x = self.eeg_adapter(x.flatten(1))
        return x


if __name__ == "__main__":
    x = torch.rand(3, 17, 250)
    net = Brain_Visual_Encoder_MEG(17, 1024, use_evnet=True)

    img_1 = torch.rand(4, 1, 1024)
    img_2 = torch.rand(4, 1, 1024)
    imgs = torch.cat([img_1, img_2], 1)
    evnet_feat = torch.rand(4, 1024)

    y = net(x)
    print(f"EEG output shape: {y.shape}")
    
    img_feat = net.get_image_feature(imgs, evnet_feat)
    print(f"Fused image feature shape: {img_feat.shape}")
    print(f"Fusion weights: {net.get_fusion_weights()}")