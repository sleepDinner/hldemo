import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualTextureBlock(nn.Module):
    """Local texture block with depthwise spatial filtering and residual path."""

    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden_channels = channels * expansion
        self.block = nn.Sequential(
            ConvBNAct(channels, hidden_channels, kernel_size=1, padding=0),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, groups=hidden_channels),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, C, H, W]
        return self.act(x + self.block(x))


class LocalTextureBackbone(nn.Module):
    """Four-stage CNN backbone for local texture and boundary artifacts."""

    def __init__(self, in_channels=3, base_channels=32, blocks_per_stage=(1, 1, 2, 2)):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        self.out_channels = [c1, c2, c3, c4]

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, c1, kernel_size=3, stride=2),
            ResidualTextureBlock(c1),
        )
        self.stage2 = self._make_stage(c1, c2, blocks_per_stage[1])
        self.stage3 = self._make_stage(c2, c3, blocks_per_stage[2])
        self.stage4 = self._make_stage(c3, c4, blocks_per_stage[3])

    @staticmethod
    def _make_stage(in_channels, out_channels, num_blocks):
        layers = [ConvBNAct(in_channels, out_channels, kernel_size=3, stride=2)]
        layers.extend(ResidualTextureBlock(out_channels) for _ in range(num_blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input x: [B, 3, H, W]
        x1 = self.stem(x)
        # x1: [B, C1, H/2, W/2]
        x2 = self.stage2(x1)
        # x2: [B, C2, H/4, W/4]
        x3 = self.stage3(x2)
        # x3: [B, C3, H/8, W/8]
        x4 = self.stage4(x3)
        # x4: [B, C4, H/16, W/16]
        return [x1, x2, x3, x4]


ConvBlock = ConvBNAct
TinyCNNBackbone = LocalTextureBackbone
