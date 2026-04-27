import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ConvBNAct, ResidualTextureBlock


class HighResolutionRefinement(nn.Module):
    """Full-resolution mask refinement for small regions and sharp boundaries.

    The main decoder predicts masks from H/2 features and upsamples them to
    H x W. This module compensates for boundary detail loss caused by early
    downsampling by re-injecting full-resolution RGB details and optional SRM
    residual details before producing the final mask logits.
    """

    def __init__(self, image_channels=3, residual_channels=0, refine_channels=32):
        super().__init__()
        self.use_residual = residual_channels > 0

        self.image_stem = nn.Sequential(
            ConvBNAct(image_channels, refine_channels, kernel_size=3),
            ResidualTextureBlock(refine_channels),
        )
        self.mask_stem = nn.Sequential(
            ConvBNAct(1, refine_channels, kernel_size=3),
            ResidualTextureBlock(refine_channels),
        )
        self.residual_stem = (
            nn.Sequential(
                ConvBNAct(residual_channels, refine_channels, kernel_size=3),
                ResidualTextureBlock(refine_channels),
            )
            if self.use_residual
            else None
        )

        fusion_channels = refine_channels * (3 if self.use_residual else 2)
        self.fuse = nn.Sequential(
            ConvBNAct(fusion_channels, refine_channels, kernel_size=1, padding=0),
            ResidualTextureBlock(refine_channels),
            ConvBNAct(refine_channels, refine_channels, kernel_size=3),
        )
        self.delta_head = nn.Conv2d(refine_channels, 1, kernel_size=3, padding=1)

    def forward(self, image, mask_logits, residual=None):
        # image: [B, 3, H, W]
        # mask_logits: [B, 1, H, W] or lower-resolution logits to be resized.
        output_size = image.shape[-2:]
        if mask_logits.shape[-2:] != output_size:
            mask_logits = F.interpolate(mask_logits, size=output_size, mode="bilinear", align_corners=False)
        # mask_logits: [B, 1, H, W]

        image_feature = self.image_stem(image)
        # image_feature: [B, refine_C, H, W]
        mask_feature = self.mask_stem(mask_logits)
        # mask_feature: [B, refine_C, H, W]

        features = [image_feature, mask_feature]
        if self.residual_stem is not None:
            if residual is None:
                residual_feature = torch.zeros_like(image_feature)
            else:
                if residual.shape[-2:] != output_size:
                    residual = F.interpolate(residual, size=output_size, mode="bilinear", align_corners=False)
                # residual: [B, residual_C, H, W]
                residual_feature = self.residual_stem(residual)
                # residual_feature: [B, refine_C, H, W]
            features.append(residual_feature)

        refined_feature = self.fuse(torch.cat(features, dim=1))
        # refined_feature: [B, refine_C, H, W]
        delta_logits = self.delta_head(refined_feature)
        # delta_logits: [B, 1, H, W]

        refined_mask_logits = mask_logits + delta_logits
        # refined_mask_logits: [B, 1, H, W]
        return refined_mask_logits
