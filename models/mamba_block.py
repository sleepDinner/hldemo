import torch
import torch.nn as nn


class DirectionalSSMBlock(nn.Module):
    """Lightweight Mamba-like block using four directional cumulative scans.

    This is not a dependency on the official Mamba implementation. It is a
    practical global-context block with a selective gate and directional state
    aggregation, designed to be easy to run and replace later.
    """

    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden_channels = channels * expansion
        self.in_proj = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(hidden_channels)
        self.out_proj = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        self.out_norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, C, H, W]
        residual = x

        projected = self.in_proj(x)
        # projected: [B, 2 * hidden_C, H, W]
        value, gate = projected.chunk(2, dim=1)
        # value: [B, hidden_C, H, W], gate: [B, hidden_C, H, W]

        value = self.act(self.norm(self.dwconv(value)))
        # value: [B, hidden_C, H, W]

        scanned = self._four_direction_scan(value)
        # scanned: [B, hidden_C, H, W]

        gated = scanned * torch.sigmoid(gate)
        # gated: [B, hidden_C, H, W]

        out = self.out_norm(self.out_proj(gated))
        # out: [B, C, H, W]
        return self.act(residual + out)

    @staticmethod
    def _four_direction_scan(x):
        # x: [B, C, H, W]
        height = x.shape[-2]
        width = x.shape[-1]

        width_denominator = torch.arange(1, width + 1, device=x.device, dtype=x.dtype).view(1, 1, 1, width)
        height_denominator = torch.arange(1, height + 1, device=x.device, dtype=x.dtype).view(1, 1, height, 1)

        left_to_right = torch.cumsum(x, dim=3) / width_denominator
        # left_to_right: [B, C, H, W]
        right_to_left = torch.flip(
            torch.cumsum(torch.flip(x, dims=[3]), dim=3) / width_denominator,
            dims=[3],
        )
        # right_to_left: [B, C, H, W]
        top_to_bottom = torch.cumsum(x, dim=2) / height_denominator
        # top_to_bottom: [B, C, H, W]
        bottom_to_top = torch.flip(
            torch.cumsum(torch.flip(x, dims=[2]), dim=2) / height_denominator,
            dims=[2],
        )
        # bottom_to_top: [B, C, H, W]

        return 0.25 * (left_to_right + right_to_left + top_to_bottom + bottom_to_top)


MambaLikeBlock = DirectionalSSMBlock
