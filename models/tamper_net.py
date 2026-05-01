import torch.nn as nn

from .backbone import LocalTextureBackbone
from .decoder import BoundaryRefinementDecoder
from .frequency_branch import build_frequency_branch
from .fusion import FeatureFusion
from .hr_refinement import HighResolutionRefinement
from .mamba_block import DirectionalSSMBlock


class TamperNet(nn.Module):
    """RFB-TraceFormer style network for image manipulation localization.

    Components:
    - RGB local texture backbone.
    - SRM/high-frequency residual branch.
    - Gated RGB-residual feature fusion.
    - Directional SSM global-context blocks.
    - FPN decoder with boundary refinement.
    - Optional high-resolution refinement before final mask output.
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
        use_hr_refine=True,
        hr_refine_channels=32,
        frequency_branch_type="hybrid",
        fusion_type="cross_attention",
        decoder_dropout=0.0,
        feature_dropout=0.0,
        image_dropout=0.0,
        use_image_head=True,
    ):
        super().__init__()
        self.use_frequency_branch = use_frequency_branch
        self.use_global_block = use_global_block
        self.use_hr_refine = use_hr_refine
        self.use_image_head = use_image_head
        self.frequency_branch_type = frequency_branch_type
        self.fusion_type = fusion_type

        self.rgb_backbone = LocalTextureBackbone(in_channels=in_channels, base_channels=base_channels)
        self.frequency_branch = (
            build_frequency_branch(
                branch_type=frequency_branch_type,
                in_channels=in_channels,
                base_channels=base_channels,
            )
            if use_frequency_branch
            else None
        )
        self.fusion = (
            FeatureFusion(self.rgb_backbone.out_channels, fusion_type=fusion_type)
            if use_frequency_branch
            else None
        )

        deep_channels = self.rgb_backbone.out_channels[-1]
        if use_global_block:
            self.global_blocks = nn.Sequential(
                *[DirectionalSSMBlock(deep_channels) for _ in range(global_blocks)]
            )
        else:
            self.global_blocks = nn.Identity()
        self.feature_dropout = nn.Dropout2d(float(feature_dropout)) if float(feature_dropout) > 0 else nn.Identity()

        self.decoder = BoundaryRefinementDecoder(
            in_channels=self.rgb_backbone.out_channels,
            decoder_channels=decoder_channels,
            use_boundary_head=use_boundary_head,
            dropout=decoder_dropout,
        )
        residual_channels = (
            self.frequency_branch.high_pass.out_channels if self.frequency_branch is not None else 0
        )
        self.hr_refine = (
            HighResolutionRefinement(
                image_channels=in_channels,
                residual_channels=residual_channels,
                refine_channels=hr_refine_channels,
            )
            if use_hr_refine
            else None
        )
        self.image_head = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(float(image_dropout)) if float(image_dropout) > 0 else nn.Identity(),
                nn.Linear(deep_channels, 1),
            )
            if use_image_head
            else None
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

        hr_residual = None
        if self.frequency_branch is not None:
            freq_features, hr_residual = self._run_frequency_branch(x)
            # freq_features have the same shapes as rgb_features.
            fused_features = self.fusion(rgb_features, freq_features)
            # fused_features[i]: [B, {C,2C,4C,8C}, H/(2**(i+1)), W/(2**(i+1))]
        else:
            fused_features = rgb_features
            # fused_features are RGB-only multi-scale features.

        fused_features[-1] = self.feature_dropout(self.global_blocks(fused_features[-1]))
        # fused_features[-1]: [B, 8C, H/16, W/16] after global context modeling.

        mask_logits, boundary_logits, coarse_mask_logits = self.decoder(
            fused_features,
            output_size=output_size,
        )
        # mask_logits: [B, 1, H, W]
        # boundary_logits: [B, 1, H, W] if boundary head is enabled, else None.
        # coarse_mask_logits: [B, 1, H, W]

        if self.hr_refine is not None:
            # Full-resolution residual details help the refinement branch
            # recover small tampered regions and sharp edges. The current
            # default path uses the compressed hybrid fixed+learnable residual
            # trace; legacy fixed SRM remains supported for baseline studies.
            # hr_residual: [B, 9, H, W] when frequency branch is enabled, else None.
            mask_logits = self.hr_refine(x, mask_logits, residual=hr_residual)
            # mask_logits: [B, 1, H, W] refined at full image resolution.

        outputs = {
            "mask_logits": mask_logits,
            "boundary_logits": boundary_logits,
            "coarse_mask_logits": coarse_mask_logits,
        }
        if self.image_head is not None:
            image_logits = self.image_head(fused_features[-1])
            # image_logits: [B, 1]
            outputs["image_logits"] = image_logits
        return outputs

    def _run_frequency_branch(self, x):
        if self.frequency_branch is None:
            return None, None
        if hasattr(self.frequency_branch, "extract_residual"):
            residual = self.frequency_branch.extract_residual(x)
            features = self.frequency_branch.backbone(residual)
            return features, residual

        residual = self.frequency_branch.high_pass(x)
        features = self.frequency_branch.backbone(residual)
        return features, residual


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
        use_hr_refine=model_config.get("use_hr_refine", True),
        hr_refine_channels=model_config.get("hr_refine_channels", 32),
        frequency_branch_type=model_config.get("frequency_branch_type", "hybrid"),
        fusion_type=model_config.get("fusion_type", "cross_attention"),
        decoder_dropout=model_config.get("decoder_dropout", 0.0),
        feature_dropout=model_config.get("feature_dropout", 0.0),
        image_dropout=model_config.get("image_dropout", 0.0),
        use_image_head=model_config.get("use_image_head", True),
    )
