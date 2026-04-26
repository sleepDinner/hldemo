import torch.nn as nn

from .backbone import LocalTextureBackbone
from .decoder import BoundaryRefinementDecoder
from .frequency_branch import FrequencyBranch
from .fusion import FeatureFusion
from .mamba_block import DirectionalSSMBlock


class TamperNet(nn.Module):
    """RFB-TraceFormer style network for image manipulation localization.

    Components:
    - RGB local texture backbone.
    - SRM/high-frequency residual branch.
    - Gated RGB-residual feature fusion.
    - Directional SSM global-context blocks.
    - FPN decoder with boundary refinement.
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=32,
        decoder_channels=64,
        use_frequency_branch=True,
        use_global_block=True,
        use_boundary_head=True,
        global_blocks=2,
    ):
        super().__init__()
        self.use_frequency_branch = use_frequency_branch
        self.use_global_block = use_global_block

        self.rgb_backbone = LocalTextureBackbone(in_channels=in_channels, base_channels=base_channels)
        self.frequency_branch = (
            FrequencyBranch(in_channels=in_channels, base_channels=base_channels)
            if use_frequency_branch
            else None
        )
        self.fusion = FeatureFusion(self.rgb_backbone.out_channels) if use_frequency_branch else None

        deep_channels = self.rgb_backbone.out_channels[-1]
        if use_global_block:
            self.global_blocks = nn.Sequential(
                *[DirectionalSSMBlock(deep_channels) for _ in range(global_blocks)]
            )
        else:
            self.global_blocks = nn.Identity()

        self.decoder = BoundaryRefinementDecoder(
            in_channels=self.rgb_backbone.out_channels,
            decoder_channels=decoder_channels,
            use_boundary_head=use_boundary_head,
        )
        self.image_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(deep_channels, 1),
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        output_size = x.shape[-2:]
        # output_size: [H, W]

        rgb_features = self.rgb_backbone(x)
        # rgb_features[0]: [B, C, H/2, W/2]
        # rgb_features[1]: [B, 2C, H/4, W/4]
        # rgb_features[2]: [B, 4C, H/8, W/8]
        # rgb_features[3]: [B, 8C, H/16, W/16]

        if self.frequency_branch is not None:
            freq_features = self.frequency_branch(x)
            # freq_features have the same shapes as rgb_features.
            fused_features = self.fusion(rgb_features, freq_features)
            # fused_features[i]: [B, {C,2C,4C,8C}, H/(2**(i+1)), W/(2**(i+1))]
        else:
            fused_features = rgb_features
            # fused_features are RGB-only multi-scale features.

        fused_features[-1] = self.global_blocks(fused_features[-1])
        # fused_features[-1]: [B, 8C, H/16, W/16] after global context modeling.

        mask_logits, boundary_logits, coarse_mask_logits = self.decoder(
            fused_features,
            output_size=output_size,
        )
        # mask_logits: [B, 1, H, W]
        # boundary_logits: [B, 1, H, W] if boundary head is enabled, else None.
        # coarse_mask_logits: [B, 1, H, W]

        image_logits = self.image_head(fused_features[-1])
        # image_logits: [B, 1]

        return {
            "mask_logits": mask_logits,
            "boundary_logits": boundary_logits,
            "coarse_mask_logits": coarse_mask_logits,
            "image_logits": image_logits,
        }


def build_model(config):
    model_config = config.get("model", {})
    return TamperNet(
        in_channels=model_config.get("in_channels", 3),
        base_channels=model_config.get("base_channels", 32),
        decoder_channels=model_config.get("decoder_channels", 64),
        use_frequency_branch=model_config.get("use_frequency_branch", True),
        use_global_block=model_config.get("use_global_block", True),
        use_boundary_head=model_config.get("use_boundary_head", True),
        global_blocks=model_config.get("global_blocks", 2),
    )
