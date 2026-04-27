import torch.nn as nn
import torch.nn.functional as F

import torch

from .backbone import ConvBNAct, LocalTextureBackbone, ResidualTextureBlock


class FixedSRMHighPass(nn.Module):
    """Fixed SRM-style high-pass filters for traditional forensic priors."""

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


class LearnableHighPass(nn.Module):
    """Lightweight learnable residual extractor for complex tampering traces.

    Fixed SRM filters preserve hand-crafted forensic priors, while this path
    adapts to data-specific artifacts such as AI edits, inpainting, heavy JPEG
    recompression, resizing, and splicing boundaries.
    """

    def __init__(self, in_channels=3, out_channels=9):
        super().__init__()
        self.out_channels = out_channels
        self.low_pass = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=5,
            padding=2,
            groups=in_channels,
            bias=False,
        )
        self.detail = nn.Sequential(
            ConvBNAct(in_channels, out_channels, kernel_size=3),
            ResidualTextureBlock(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )
        self._init_low_pass()

    def _init_low_pass(self):
        with torch.no_grad():
            kernel = torch.full_like(self.low_pass.weight, 1.0 / 25.0)
            self.low_pass.weight.copy_(kernel)

    def forward(self, x):
        # x: [B, 3, H, W]
        local_mean = self.low_pass(x)
        # local_mean: [B, 3, H, W]
        high_pass_input = x - local_mean
        # high_pass_input: [B, 3, H, W]
        residual = self.detail(high_pass_input)
        # residual: [B, learnable_C, H, W]
        return torch.tanh(residual)


class FrequencyBranch(nn.Module):
    """Fixed SRM branch for noise residual cues.

    This original branch is kept for backward compatibility. It only uses
    fixed SRM filters before the local texture backbone.
    """

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


class HybridFrequencyBranch(nn.Module):
    """Hybrid residual trace extraction branch.

    Fixed SRM keeps the traditional high-frequency forensic prior. Learnable
    residual extraction adapts to complex manipulation traces. Concatenating
    both and compressing them strengthens residual/frequency representation
    while keeping the output feature scales aligned with the RGB backbone.
    """

    def __init__(self, in_channels=3, base_channels=32, residual_channels=9):
        super().__init__()
        self.fixed_srm = FixedSRMHighPass(channels=in_channels)
        self.high_pass = self.fixed_srm
        self.learnable_residual = LearnableHighPass(
            in_channels=in_channels,
            out_channels=residual_channels,
        )
        fused_residual_channels = self.fixed_srm.out_channels + self.learnable_residual.out_channels
        compressed_channels = self.fixed_srm.out_channels
        self.compress = nn.Sequential(
            ConvBNAct(fused_residual_channels, compressed_channels, kernel_size=1, padding=0),
            ResidualTextureBlock(compressed_channels),
        )
        self.backbone = LocalTextureBackbone(
            in_channels=compressed_channels,
            base_channels=base_channels,
            blocks_per_stage=(1, 1, 1, 1),
        )
        self.out_channels = self.backbone.out_channels
        self.residual_channels = compressed_channels

    def extract_residual(self, x):
        # x: [B, 3, H, W]
        fixed_residual = self.fixed_srm(x)
        # fixed_residual: [B, 9, H, W] for RGB input.
        learned_residual = self.learnable_residual(x)
        # learned_residual: [B, learnable_C, H, W]
        residual = torch.cat([fixed_residual, learned_residual], dim=1)
        # residual: [B, 9 + learnable_C, H, W]
        residual = self.compress(residual)
        # residual: [B, 9, H, W]
        return residual

    def forward(self, x):
        # x: [B, 3, H, W]
        residual = self.extract_residual(x)
        # residual: [B, 9, H, W]
        features = self.backbone(residual)
        # features:
        #   f1: [B, C, H/2, W/2]
        #   f2: [B, 2C, H/4, W/4]
        #   f3: [B, 4C, H/8, W/8]
        #   f4: [B, 8C, H/16, W/16]
        return features


def build_frequency_branch(branch_type="hybrid", in_channels=3, base_channels=32):
    branch_type = branch_type.lower()
    if branch_type == "fixed_srm":
        return FrequencyBranch(in_channels=in_channels, base_channels=base_channels)
    if branch_type == "hybrid":
        return HybridFrequencyBranch(in_channels=in_channels, base_channels=base_channels)
    raise ValueError(f"Unsupported frequency_branch_type: {branch_type}")


FixedHighPassFilter = FixedSRMHighPass
