import torch
import torch.nn as nn

from .backbone import ConvBNAct, ResidualTextureBlock


class TraceFusionBlock(nn.Module):
    """Fuse RGB features with residual/frequency features using a learned gate."""

    def __init__(self, channels):
        super().__init__()
        self.rgb_proj = ConvBNAct(channels, channels, kernel_size=1, padding=0)
        self.aux_proj = ConvBNAct(channels, channels, kernel_size=1, padding=0)
        self.mix = nn.Sequential(
            ConvBNAct(channels * 2, channels, kernel_size=3),
            ResidualTextureBlock(channels),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, rgb_feature, aux_feature):
        # rgb_feature: [B, C, Hs, Ws]
        # aux_feature: [B, C, Hs, Ws]
        rgb = self.rgb_proj(rgb_feature)
        # rgb: [B, C, Hs, Ws]
        aux = self.aux_proj(aux_feature)
        # aux: [B, C, Hs, Ws]
        joint = torch.cat([rgb, aux], dim=1)
        # joint: [B, 2C, Hs, Ws]
        gate = self.gate(joint)
        # gate: [B, C, Hs, Ws]
        trace = self.mix(joint)
        # trace: [B, C, Hs, Ws]
        return rgb_feature + gate * trace


class FeatureFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.ModuleList([TraceFusionBlock(channel) for channel in channels])

    def forward(self, rgb_features, aux_features=None):
        # rgb_features: list of four tensors at [H/2, H/4, H/8, H/16]
        if aux_features is None:
            return rgb_features

        fused_features = []
        for block, rgb_feature, aux_feature in zip(self.blocks, rgb_features, aux_features):
            fused_features.append(block(rgb_feature, aux_feature))
        # fused_features: same shapes as rgb_features
        return fused_features
