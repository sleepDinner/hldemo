import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.seed import set_seed
from utils.visualization import save_mask, save_prediction_visualization
from utils.warnings import suppress_pil_exif_warnings


suppress_pil_exif_warnings()


def parse_args():
    parser = argparse.ArgumentParser(description="Run tamper localization inference.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override.")
    parser.add_argument("--output-dir", default=None, help="Directory for predicted masks.")
    parser.add_argument("--threshold", type=float, default=None, help="Binary mask threshold override.")
    return parser.parse_args()


def get_device(config):
    accelerator = config.get("device", {}).get("accelerator", "cuda")
    if accelerator == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(config, checkpoint_path, device):
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image, config):
    input_h, input_w = config["data"]["input_size"]
    resized = image.resize((input_w, input_h), Image.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(image_array).unsqueeze(0)
    normalize = config["data"].get("normalize", {})
    mean = torch.tensor(normalize.get("mean", [0.0, 0.0, 0.0]), dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(normalize.get("std", [1.0, 1.0, 1.0]), dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean) / std


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 2026))

    checkpoint_path = args.checkpoint or config.get("checkpoint", {}).get("path")
    if not checkpoint_path:
        raise ValueError("A checkpoint path is required. Set checkpoint.path in YAML or pass --checkpoint.")

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(config["experiment"]["output_dir"]) / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = args.threshold if args.threshold is not None else config.get("metrics", {}).get("threshold", 0.5)

    device = get_device(config)
    model = load_model(config, checkpoint_path, device)

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    original_w, original_h = image.size
    input_tensor = preprocess_image(image, config).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        prob = torch.sigmoid(outputs["mask_logits"])
        prob = F.interpolate(prob, size=(original_h, original_w), mode="bilinear", align_corners=False)

    prob_map = prob[0, 0].detach().cpu().numpy()
    binary_map = (prob_map >= threshold).astype(np.float32)
    image_array = np.asarray(image, dtype=np.uint8)

    stem = image_path.stem
    prob_path = output_dir / f"{stem}_prob.png"
    binary_path = output_dir / f"{stem}_binary.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    save_mask(prob_map, prob_path)
    save_mask(binary_map, binary_path)
    save_prediction_visualization(image_array, prob_map, overlay_path)

    print(f"probability_mask: {prob_path}")
    print(f"binary_mask: {binary_path}")
    print(f"overlay: {overlay_path}")


if __name__ == "__main__":
    main()
