import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import LocalTextureBackbone


class FixedSRMHighPass(nn.Module):
    """Fixed SRM-style high-pass filters for noise and residual artifacts."""

    def __init__(self, channels=3):
        super().__init__()
        kernels = torch.tensor(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, -1, 2, -1, 0],
                    [0, 2, -4, 2, 0],
                    [0, -1, 2, -1, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, -2, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
            ],
            dtype=torch.float32,
        )
        kernels[0] = kernels[0] / 4.0
        kernels[1] = kernels[1] / 12.0
        kernels[2] = kernels[2] / 2.0
        weight = kernels[:, None, :, :].repeat(channels, 1, 1, 1)
        self.register_buffer("weight", weight)
        self.channels = channels
        self.num_filters = kernels.shape[0]

    @property
    def out_channels(self):
        return self.channels * self.num_filters

    def forward(self, x):
        # x: [B, 3, H, W]
        residual = F.conv2d(x, self.weight, padding=2, groups=self.channels)
        # residual: [B, 3 * num_filters, H, W]
        return torch.tanh(residual)


class FrequencyBranch(nn.Module):
    """High-frequency branch for SRM/noise residual cues."""

    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.high_pass = FixedSRMHighPass(channels=in_channels)
        self.backbone = LocalTextureBackbone(
            in_channels=self.high_pass.out_channels,
            base_channels=base_channels,
            blocks_per_stage=(1, 1, 1, 1),
        )
        self.out_channels = self.backbone.out_channels

    def forward(self, x):
        # x: [B, 3, H, W]
        residual = self.high_pass(x)
        # residual: [B, 9, H, W] when RGB input and 3 SRM filters are used.
        features = self.backbone(residual)
        # features:
        #   f1: [B, C, H/2, W/2]
        #   f2: [B, 2C, H/4, W/4]
        #   f3: [B, 4C, H/8, W/8]
        #   f4: [B, 8C, H/16, W/16]
        return features


FixedHighPassFilter = FixedSRMHighPass
