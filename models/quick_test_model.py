import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.tamper_net import TamperNet


def run_case(device, explicit_new_config=False):
    kwargs = {}
    if explicit_new_config:
        kwargs = {
            "frequency_branch_type": "hybrid",
            "fusion_type": "cross_attention",
            "use_hr_refine": True,
        }

    model = TamperNet(
        in_channels=3,
        base_channels=16,
        decoder_channels=32,
        use_frequency_branch=True,
        use_global_block=True,
        use_boundary_head=True,
        global_blocks=1,
        hr_refine_channels=16,
        **kwargs,
    ).to(device)
    model.train()

    x = torch.randn(2, 3, 256, 256, device=device)
    # x: [B=2, C=3, H=256, W=256]

    outputs = model(x)
    # outputs["mask_logits"]: [2, 1, 256, 256]
    # outputs["boundary_logits"]: [2, 1, 256, 256]
    # outputs["coarse_mask_logits"]: [2, 1, 256, 256]
    # outputs["image_logits"]: [2, 1]

    assert outputs["mask_logits"].shape == (2, 1, 256, 256)
    assert outputs["boundary_logits"].shape == (2, 1, 256, 256)
    assert outputs["coarse_mask_logits"].shape == (2, 1, 256, 256)
    assert outputs["image_logits"].shape == (2, 1)

    loss = (
        outputs["mask_logits"].mean()
        + outputs["boundary_logits"].mean()
        + outputs["coarse_mask_logits"].mean()
        + outputs["image_logits"].mean()
    )
    loss.backward()

    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"explicit_new_config: {explicit_new_config}")
    print(f"frequency_branch_type: {model.frequency_branch_type}")
    print(f"fusion_type: {model.fusion_type}")
    print(f"use_hr_refine: {model.use_hr_refine}")
    print(f"parameters: {num_params:,}")
    for name, value in outputs.items():
        print(f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}")


def main():
    torch.manual_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    run_case(device, explicit_new_config=False)
    run_case(device, explicit_new_config=True)
    print("quick model test passed")


if __name__ == "__main__":
    main()
