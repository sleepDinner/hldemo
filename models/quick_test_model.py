import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.tamper_net import TamperNet


def main():
    torch.manual_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TamperNet(
        in_channels=3,
        base_channels=16,
        decoder_channels=32,
        use_frequency_branch=True,
        use_global_block=True,
        use_boundary_head=True,
        global_blocks=1,
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
    print(f"device: {device}")
    print(f"parameters: {num_params:,}")
    for name, value in outputs.items():
        print(f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}")
    print("quick model test passed")


if __name__ == "__main__":
    main()
