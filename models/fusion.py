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


class CrossDomainTraceFusionBlock(nn.Module):
    """Lightweight cross-domain attention fusion for RGB and residual traces.

    RGB features provide texture and semantic context. Residual/frequency
    features provide noise and high-frequency anomalies. This block learns
    which residual traces are reliable through channel attention, spatial
    attention, and a residual confidence gate without using expensive global
    attention.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.rgb_proj = ConvBNAct(channels, channels, kernel_size=1, padding=0)
        self.aux_proj = ConvBNAct(channels, channels, kernel_size=1, padding=0)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels * 2, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            ConvBNAct(channels * 2, channels, kernel_size=3),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.residual_confidence = nn.Sequential(
            ConvBNAct(channels * 2, channels, kernel_size=1, padding=0),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True),
            nn.Sigmoid(),
        )
        self.cross_mix = nn.Sequential(
            ConvBNAct(channels * 2, channels, kernel_size=3),
            ResidualTextureBlock(channels),
        )
        self.out_proj = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=1, padding=0),
            ResidualTextureBlock(channels),
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
        channel_weight = self.channel_attention(joint)
        # channel_weight: [B, 2C, 1, 1]
        rgb_channel_weight, aux_channel_weight = channel_weight.chunk(2, dim=1)
        # rgb_channel_weight/aux_channel_weight: [B, C, 1, 1]

        rgb_attended = rgb * rgb_channel_weight
        # rgb_attended: [B, C, Hs, Ws]
        aux_attended = aux * aux_channel_weight
        # aux_attended: [B, C, Hs, Ws]

        attended_joint = torch.cat([rgb_attended, aux_attended], dim=1)
        # attended_joint: [B, 2C, Hs, Ws]
        spatial_weight = self.spatial_attention(attended_joint)
        # spatial_weight: [B, 1, Hs, Ws]
        residual_confidence = self.residual_confidence(attended_joint)
        # residual_confidence: [B, C, Hs, Ws]

        trusted_aux = aux_attended * spatial_weight * residual_confidence
        # trusted_aux: [B, C, Hs, Ws]
        cross_trace = self.cross_mix(torch.cat([rgb_attended, trusted_aux], dim=1))
        # cross_trace: [B, C, Hs, Ws]

        fused = rgb_feature + self.out_proj(cross_trace)
        # fused: [B, C, Hs, Ws]
        return fused


def build_fusion_block(channels, fusion_type):
    fusion_type = fusion_type.lower()
    if fusion_type == "gate":
        return TraceFusionBlock(channels)
    if fusion_type == "cross_attention":
        return CrossDomainTraceFusionBlock(channels)
    raise ValueError(f"Unsupported fusion_type: {fusion_type}")


class FeatureFusion(nn.Module):
    def __init__(self, channels, fusion_type="cross_attention"):
        super().__init__()
        self.fusion_type = fusion_type
        self.blocks = nn.ModuleList([build_fusion_block(channel, fusion_type) for channel in channels])

    def forward(self, rgb_features, aux_features=None):
        # rgb_features: list of four tensors at [H/2, H/4, H/8, H/16]
        if aux_features is None:
            return rgb_features

        fused_features = []
        for block, rgb_feature, aux_feature in zip(self.blocks, rgb_features, aux_features):
            fused_features.append(block(rgb_feature, aux_feature))
        # fused_features: same shapes as rgb_features
        return fused_features
