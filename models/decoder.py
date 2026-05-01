import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ConvBNAct, ResidualTextureBlock


class BoundaryRefinementDecoder(nn.Module):
    """FPN-style decoder with a boundary branch for sharper masks."""

    def __init__(self, in_channels, decoder_channels=64, use_boundary_head=True, dropout=0.0):
        super().__init__()
        self.use_boundary_head = use_boundary_head
        self.dropout = nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(channels, decoder_channels, kernel_size=1, bias=False) for channels in in_channels]
        )
        self.refine_blocks = nn.ModuleList(
            [ResidualTextureBlock(decoder_channels) for _ in in_channels]
        )
        self.aggregate = nn.Sequential(
            ConvBNAct(decoder_channels * len(in_channels), decoder_channels, kernel_size=3),
            ResidualTextureBlock(decoder_channels),
        )
        self.boundary_refine = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels // 2, kernel_size=3),
            ResidualTextureBlock(decoder_channels // 2),
        )
        self.boundary_head = (
            nn.Conv2d(decoder_channels // 2, 1, kernel_size=1) if use_boundary_head else None
        )
        mask_in_channels = decoder_channels + (decoder_channels // 2 if use_boundary_head else 0)
        self.mask_head = nn.Sequential(
            ConvBNAct(mask_in_channels, decoder_channels, kernel_size=3),
            nn.Conv2d(decoder_channels, 1, kernel_size=1),
        )
        self.coarse_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)

    def forward(self, features, output_size):
        # features[0]: [B, C1, H/2, W/2]
        # features[1]: [B, C2, H/4, W/4]
        # features[2]: [B, C3, H/8, W/8]
        # features[3]: [B, C4, H/16, W/16]
        lateral = [conv(feature) for conv, feature in zip(self.lateral_convs, features)]
        # each lateral[i]: [B, decoder_C, Hi, Wi]

        pyramid = [None] * len(lateral)
        pyramid[-1] = self.refine_blocks[-1](lateral[-1])
        # pyramid[3]: [B, decoder_C, H/16, W/16]

        for index in range(len(lateral) - 2, -1, -1):
            upsampled = F.interpolate(
                pyramid[index + 1],
                size=lateral[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            # upsampled: [B, decoder_C, Hi, Wi]
            pyramid[index] = self.refine_blocks[index](lateral[index] + upsampled)
            # pyramid[index]: [B, decoder_C, Hi, Wi]

        finest_size = pyramid[0].shape[-2:]
        multi_scale = [
            F.interpolate(feature, size=finest_size, mode="bilinear", align_corners=False)
            for feature in pyramid
        ]
        # multi_scale tensors: all [B, decoder_C, H/2, W/2]

        decoded = self.aggregate(torch.cat(multi_scale, dim=1))
        decoded = self.dropout(decoded)
        # decoded: [B, decoder_C, H/2, W/2]

        coarse_mask_logits = self.coarse_head(decoded)
        # coarse_mask_logits: [B, 1, H/2, W/2]

        boundary_logits = None
        if self.boundary_head is not None:
            boundary_feature = self.boundary_refine(decoded)
            # boundary_feature: [B, decoder_C/2, H/2, W/2]
            boundary_logits = self.boundary_head(boundary_feature)
            # boundary_logits: [B, 1, H/2, W/2]
            mask_feature = torch.cat([decoded, boundary_feature], dim=1)
            # mask_feature: [B, decoder_C + decoder_C/2, H/2, W/2]
        else:
            mask_feature = decoded
            # mask_feature: [B, decoder_C, H/2, W/2]

        mask_logits = self.mask_head(mask_feature)
        # mask_logits: [B, 1, H/2, W/2]

        mask_logits = F.interpolate(mask_logits, size=output_size, mode="bilinear", align_corners=False)
        # mask_logits: [B, 1, H, W]
        coarse_mask_logits = F.interpolate(
            coarse_mask_logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        # coarse_mask_logits: [B, 1, H, W]
        if boundary_logits is not None:
            boundary_logits = F.interpolate(
                boundary_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            # boundary_logits: [B, 1, H, W]

        return mask_logits, boundary_logits, coarse_mask_logits


MultiScaleDecoder = BoundaryRefinementDecoder
