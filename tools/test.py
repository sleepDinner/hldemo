import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.tamper_dataset import build_dataset_from_split_config
from datasets.transforms import build_transforms
from metrics import RunningBinaryMetrics
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config, save_config
from utils.logger import setup_logger
from utils.seed import set_seed
from utils.visualization import save_mask, save_prediction_visualization


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate tamper localization model.")
    parser.add_argument("--config", required=True, help="Path to a YAML test config.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override.")
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


def get_test_sets(config):
    data_config = config["data"]
    if "test_sets" in data_config:
        return data_config["test_sets"]
    if "test" in data_config:
        return [data_config["test"]]
    raise KeyError("Test config must define data.test_sets or data.test.")


def denormalize_images(images, config):
    mean = torch.tensor(config["data"].get("normalize", {}).get("mean", [0.0, 0.0, 0.0])).view(1, 3, 1, 1)
    std = torch.tensor(config["data"].get("normalize", {}).get("std", [1.0, 1.0, 1.0])).view(1, 3, 1, 1)
    return (images.cpu() * std + mean).clamp(0.0, 1.0)


def save_batch_visuals(batch, probs, output_dir, config, threshold, start_index, max_items):
    output_dir.mkdir(parents=True, exist_ok=True)
    images = denormalize_images(batch["image"], config)
    probs = probs.detach().cpu()
    saved = 0

    for index in range(min(images.shape[0], max_items)):
        image = (images[index].permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        prob_map = probs[index, 0].numpy()
        binary_map = (prob_map >= threshold).astype(np.float32)
        image_stem = Path(batch["image_path"][index]).stem
        item_id = start_index + index
        prefix = f"{item_id:05d}_{image_stem}"
        save_mask(prob_map, output_dir / f"{prefix}_prob.png")
        save_mask(binary_map, output_dir / f"{prefix}_binary.png")
        save_prediction_visualization(image, prob_map, output_dir / f"{prefix}_overlay.png")
        saved += 1

    return saved


@torch.no_grad()
def evaluate_one_dataset(model, dataset_config, config, device, output_dir):
    transform = build_transforms(config, mode="test")
    dataset = build_dataset_from_split_config(dataset_config, transform=transform, mode="test")
    data_config = config["data"]
    loader = DataLoader(
        dataset,
        batch_size=dataset_config.get("batch_size", data_config.get("batch_size", 1)),
        shuffle=False,
        num_workers=config.get("device", {}).get("num_workers", 4),
        pin_memory=config.get("device", {}).get("pin_memory", True) and device.type == "cuda",
        drop_last=False,
    )

    metrics_config = config.get("metrics", {})
    threshold = metrics_config.get("threshold", 0.5)
    tracker = RunningBinaryMetrics(
        threshold=threshold,
        max_auc_pixels=metrics_config.get("max_auc_pixels", 200000),
        auc_seed=config.get("seed", 2026),
    )

    dataset_name = dataset_config.get("name", "test")
    save_vis = metrics_config.get("save_visualizations", True)
    max_vis = metrics_config.get("max_visualizations", 16)
    saved_vis = 0

    progress = tqdm(loader, desc=f"test {dataset_name}", dynamic_ncols=True)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.sigmoid(outputs["mask_logits"])
        tracker.update(probs, masks)

        if save_vis and saved_vis < max_vis:
            saved_vis += save_batch_visuals(
                batch=batch,
                probs=probs,
                output_dir=output_dir / "visualizations" / dataset_name,
                config=config,
                threshold=threshold,
                start_index=saved_vis,
                max_items=max_vis - saved_vis,
            )

    metrics = tracker.compute()
    metrics["dataset"] = dataset_name
    metrics["samples"] = len(dataset)
    return metrics


def save_metrics(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    fieldnames = ["dataset", "samples", "f1", "iou", "auc", "precision", "recall", "mcc", "fpr"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return csv_path


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 2026))

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    logger = setup_logger(
        "test",
        log_files=[output_dir / "test.log"],
        console=True,
    )
    device = get_device(config)
    checkpoint_path = args.checkpoint or config.get("checkpoint", {}).get("path")
    if not checkpoint_path:
        raise ValueError("A checkpoint path is required. Set checkpoint.path in YAML or pass --checkpoint.")

    logger.info("using device=%s checkpoint=%s", device, checkpoint_path)
    model = load_model(config, checkpoint_path, device)

    rows = []
    for dataset_config in get_test_sets(config):
        metrics = evaluate_one_dataset(model, dataset_config, config, device, output_dir)
        rows.append(metrics)
        logger.info("dataset=%s metrics=%s", metrics["dataset"], metrics)

    csv_path = save_metrics(rows, output_dir)
    logger.info("saved metrics to %s", csv_path)


if __name__ == "__main__":
    main()
