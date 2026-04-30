import argparse
import csv
from datetime import timedelta
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
try:
    from torch.amp import GradScaler, autocast

    _AMP_USES_DEVICE_TYPE = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

    _AMP_USES_DEVICE_TYPE = False
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from datasets.tamper_dataset import build_dataset_from_split_config
from datasets.transforms import build_transforms
from losses import build_loss
from metrics import RunningBinaryMetrics
from models import build_model
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_config, save_config
from utils.logger import setup_logger
from utils.seed import set_seed
from utils.stratified_split import ensure_stratified_split
from utils.visualization import save_mask, save_prediction_visualization
from utils.warnings import suppress_pil_exif_warnings


suppress_pil_exif_warnings()


def parse_args():
    parser = argparse.ArgumentParser(description="Train tamper localization model.")
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path for resume.")
    return parser.parse_args()


def setup_distributed(timeout_minutes=120):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if not distributed:
        return False, 0, 0, 1

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Distributed training was launched with torchrun, but CUDA is not available. "
            "In your current error this usually means the installed PyTorch CUDA build is newer "
            "than the NVIDIA driver. Run `python -m tools.check_environment`, then install a "
            "driver-compatible PyTorch build such as cu121, or upgrade the NVIDIA driver."
        )

    try:
        torch.cuda.set_device(local_rank)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to initialize CUDA for distributed training. Check whether PyTorch's CUDA "
            "build matches the server NVIDIA driver. Run `python -m tools.check_environment` "
            "for diagnostics."
        ) from exc

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=timeout_minutes),
    )
    return True, local_rank, rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    import random

    import numpy as np

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_grad_scaler(enabled):
    if _AMP_USES_DEVICE_TYPE:
        try:
            return GradScaler("cuda", enabled=enabled)
        except TypeError:
            return GradScaler(enabled=enabled)
    return GradScaler(enabled=enabled)


def autocast_context(enabled):
    if _AMP_USES_DEVICE_TYPE:
        try:
            return autocast("cuda", enabled=enabled)
        except TypeError:
            return autocast(enabled=enabled)
    return autocast(enabled=enabled)


def ensure_data_split_manifests(config, logger, rank, distributed):
    if not config.get("data", {}).get("split", {}).get("enabled", False):
        return
    if is_main_process(rank):
        ensure_stratified_split(config, logger=logger)
    if distributed:
        dist.barrier()


def build_dataloaders(config, distributed):
    train_transform = build_transforms(config, mode="train")
    val_transform = build_transforms(config, mode="test")
    train_dataset = build_dataset_from_split_config(
        config["data"]["train"],
        transform=train_transform,
        mode="train",
    )
    val_dataset = build_dataset_from_split_config(
        config["data"]["val"],
        transform=val_transform,
        mode="test",
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    seed = config.get("seed", 2026)
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"].get("batch_size", 4),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config["device"].get("num_workers", 4),
        pin_memory=config["device"].get("pin_memory", True),
        drop_last=config["data"].get("drop_last", False),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"].get("val_batch_size", 1),
        shuffle=False,
        sampler=val_sampler,
        num_workers=config["device"].get("num_workers", 4),
        pin_memory=config["device"].get("pin_memory", True),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return train_loader, val_loader, train_sampler


def set_dataset_epoch(loader, epoch):
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)


def get_dataset_robust_strength(loader):
    if hasattr(loader.dataset, "get_robust_strength_factor"):
        return loader.dataset.get_robust_strength_factor()
    return 0.0


def get_dataset_active_robust_ops(loader):
    if hasattr(loader.dataset, "get_active_robust_ops"):
        return loader.dataset.get_active_robust_ops()
    return []


def build_optimizer(config, model):
    optimizer_config = config.get("optimizer", {})
    name = optimizer_config.get("name", "adamw").lower()
    lr = optimizer_config.get("lr", 1e-4)
    weight_decay = optimizer_config.get("weight_decay", 1e-4)

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=optimizer_config.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(config, optimizer):
    scheduler_config = config.get("scheduler", {})
    name = scheduler_config.get("name", "cosine").lower()
    epochs = scheduler_config.get("epochs", 100)
    warmup_epochs = scheduler_config.get("warmup_epochs", 0)

    if name == "none":
        return None

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        if name == "cosine":
            return 0.5 * (1.0 + math.cos(progress * math.pi))
        if name == "linear":
            return max(0.0, 1.0 - progress)
        raise ValueError(f"Unsupported scheduler: {name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def move_batch_to_device(batch, device):
    return {
        "image": batch["image"].to(device, non_blocking=True),
        "mask": batch["mask"].to(device, non_blocking=True),
    }


def reduce_scalar(value, device):
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    tensor = torch.tensor([float(value)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def reduce_metric_tracker(tracker):
    if not (dist.is_available() and dist.is_initialized()):
        return tracker

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, tracker.state_dict())
    merged = RunningBinaryMetrics(threshold=tracker.threshold, max_auc_pixels=None)
    for state in gathered:
        merged.merge_state_dict(state)
    return merged


def update_progress_tracker(tracker, logits, masks, config):
    progress_size = int(config.get("metrics", {}).get("progress_size", 64))
    probs = torch.sigmoid(logits.detach()).float()
    target_masks = masks.detach().float()
    if progress_size > 0:
        output_size = (progress_size, progress_size)
        probs = F.interpolate(probs, size=output_size, mode="bilinear", align_corners=False)
        target_masks = F.interpolate(target_masks, size=output_size, mode="nearest")
    tracker.update(probs, target_masks)


def progress_metric_postfix(metrics):
    if not metrics:
        return {}
    return {
        "F1": f"{metrics.get('f1', 0.0):.3f}",
        "IoU": f"{metrics.get('iou', 0.0):.3f}",
        "AUC": f"{metrics.get('auc', 0.0):.3f}",
        "AP": f"{metrics.get('ap', 0.0):.3f}",
        "MCC": f"{metrics.get('mcc', 0.0):.3f}",
        "FPR": f"{metrics.get('fpr', 0.0):.3f}",
    }


def train_one_epoch(
    model,
    criterion,
    optimizer,
    scaler,
    loader,
    device,
    epoch,
    config,
    logger,
    rank,
):
    model.train()
    running_loss = 0.0
    num_batches = 0
    amp_enabled = config.get("training", {}).get("amp", True) and device.type == "cuda"
    log_interval = config.get("logging", {}).get("log_interval", 20)
    max_grad_norm = config.get("training", {}).get("max_grad_norm", 0.0)
    total_epochs = config.get("scheduler", {}).get("epochs", epoch)
    epoch_start_time = time.time()
    threshold = config.get("metrics", {}).get("threshold", 0.5)
    progress_max_auc_pixels = config.get("metrics", {}).get("progress_max_auc_pixels", 50000)
    progress_metric_interval = max(1, int(config.get("logging", {}).get("progress_metric_interval", log_interval)))
    progress_tracker = RunningBinaryMetrics(
        threshold=threshold,
        max_auc_pixels=progress_max_auc_pixels,
    )
    progress_metrics = {}

    progress = tqdm(
        loader,
        disable=not is_main_process(rank),
        desc=f"train {epoch:03d}/{total_epochs:03d}",
        dynamic_ncols=True,
        leave=True,
        mininterval=1.0,
        position=0,
        file=sys.stdout,
    )
    for step, batch in enumerate(progress, start=1):
        inputs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        targets = {"mask": masks}

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(enabled=amp_enabled):
            outputs = model(inputs)
            loss_dict = criterion(outputs, targets, return_dict=True)
            loss = loss_dict["loss"]

        if amp_enabled:
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        loss_value = float(loss.detach().item())
        running_loss += loss_value
        num_batches += 1
        avg_loss = running_loss / max(1, num_batches)

        if is_main_process(rank):
            if step == 1 or step % progress_metric_interval == 0 or step == len(loader):
                update_progress_tracker(progress_tracker, outputs["mask_logits"], masks, config)
                progress_metrics = progress_tracker.compute()

            elapsed = max(time.time() - epoch_start_time, 1e-6)
            seconds_per_step = elapsed / step
            steps_left_this_epoch = len(loader) - step
            epochs_left_after_this = max(total_epochs - epoch, 0)
            estimated_steps_left = steps_left_this_epoch + epochs_left_after_this * len(loader)
            eta = format_duration(seconds_per_step * estimated_steps_left)
            postfix = {
                "loss": f"{loss_value:.4f}",
                "avg": f"{avg_loss:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "eta": eta,
            }
            postfix.update(progress_metric_postfix(progress_metrics))
            progress.set_postfix(postfix, refresh=False)

        if is_main_process(rank) and step % log_interval == 0:
            logger.info(
                "phase=train epoch=%d/%d step=%d/%d loss=%.6f avg_loss=%.6f lr=%.8f "
                "f1=%.6f iou=%.6f auc=%.6f ap=%.6f mcc=%.6f fpr=%.6f eta=%s",
                epoch,
                total_epochs,
                step,
                len(loader),
                loss_value,
                avg_loss,
                optimizer.param_groups[0]["lr"],
                progress_metrics.get("f1", 0.0),
                progress_metrics.get("iou", 0.0),
                progress_metrics.get("auc", 0.0),
                progress_metrics.get("ap", 0.0),
                progress_metrics.get("mcc", 0.0),
                progress_metrics.get("fpr", 0.0),
                eta,
            )

    local_loss = running_loss / max(1, num_batches)
    return {"train_loss": reduce_scalar(local_loss, device)}


@torch.no_grad()
def validate(model, criterion, loader, device, epoch, config, output_dir, rank):
    model.eval()
    threshold = config.get("metrics", {}).get("threshold", 0.5)
    max_auc_pixels = config.get("metrics", {}).get("max_auc_pixels", 200000)
    tracker = RunningBinaryMetrics(threshold=threshold, max_auc_pixels=max_auc_pixels)
    running_loss = 0.0
    num_batches = 0

    save_vis = config.get("logging", {}).get("save_visualizations", True)
    vis_interval = config.get("logging", {}).get("visualization_interval", 1)
    max_vis = config.get("logging", {}).get("max_visualizations", 8)
    should_save_vis = save_vis and is_main_process(rank) and epoch % vis_interval == 0
    saved_vis = 0
    total_epochs = config.get("scheduler", {}).get("epochs", epoch)
    epoch_start_time = time.time()
    progress_metric_interval = max(1, int(config.get("logging", {}).get("progress_metric_interval", 20)))
    progress_metrics = {}

    progress = tqdm(
        loader,
        disable=not is_main_process(rank),
        desc=f"val   {epoch:03d}/{total_epochs:03d}",
        dynamic_ncols=True,
        leave=True,
        mininterval=1.0,
        position=0,
        file=sys.stdout,
    )
    for step, batch in enumerate(progress, start=1):
        inputs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        targets = {"mask": masks}

        outputs = model(inputs)
        loss_dict = criterion(outputs, targets, return_dict=True)
        loss = loss_dict["loss"]
        loss_value = float(loss.detach().item())
        running_loss += loss_value
        num_batches += 1
        avg_loss = running_loss / max(1, num_batches)

        probs = torch.sigmoid(outputs["mask_logits"])
        tracker.update(probs, masks)

        if is_main_process(rank):
            if step == 1 or step % progress_metric_interval == 0 or step == len(loader):
                progress_metrics = tracker.compute()
            elapsed = max(time.time() - epoch_start_time, 1e-6)
            seconds_per_step = elapsed / step
            eta = format_duration(seconds_per_step * (len(loader) - step))
            postfix = {
                "loss": f"{loss_value:.4f}",
                "avg": f"{avg_loss:.4f}",
                "eta": eta,
            }
            postfix.update(progress_metric_postfix(progress_metrics))
            progress.set_postfix(postfix, refresh=False)

        if should_save_vis and saved_vis < max_vis:
            saved_vis += save_validation_visuals(
                batch,
                probs,
                config,
                output_dir / "visualizations" / f"epoch_{epoch:04d}",
                start_index=saved_vis,
                max_items=max_vis - saved_vis,
            )

    tracker = reduce_metric_tracker(tracker)
    metrics = tracker.compute()
    local_loss = running_loss / max(1, num_batches)
    metrics["val_loss"] = reduce_scalar(local_loss, device)
    return {f"val_{key}": value for key, value in metrics.items()}


def save_validation_visuals(batch, probs, config, output_dir, start_index=0, max_items=4):
    output_dir.mkdir(parents=True, exist_ok=True)
    images = batch["image"].detach().cpu()
    probs = probs.detach().cpu()
    mean = torch.tensor(config["data"].get("normalize", {}).get("mean", [0.0, 0.0, 0.0])).view(3, 1, 1)
    std = torch.tensor(config["data"].get("normalize", {}).get("std", [1.0, 1.0, 1.0])).view(3, 1, 1)

    saved = 0
    for index in range(min(images.shape[0], max_items)):
        image = (images[index] * std + mean).clamp(0.0, 1.0)
        image_array = (image.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        prob_map = probs[index, 0].numpy()
        item_id = start_index + index
        save_mask(prob_map, output_dir / f"{item_id:03d}_prob.png")
        save_prediction_visualization(image_array, prob_map, output_dir / f"{item_id:03d}_overlay.png")
        saved += 1
    return saved


def save_training_curves(history, output_dir):
    if not history:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    fieldnames = sorted({key for row in history for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    plot_keys = [key for key in fieldnames if key not in {"epoch", "lr"}]
    for key in plot_keys:
        xs = [row["epoch"] for row in history if key in row]
        ys = [row[key] for row in history if key in row]
        if not xs:
            continue
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("epoch")
        plt.ylabel(key)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f"{key}.png")
        plt.close()


def load_resume_if_needed(resume_path, model, optimizer, scheduler, scaler, device, logger, monitor):
    if not resume_path:
        return 1, None

    checkpoint = load_checkpoint(resume_path, map_location=device)
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = int(checkpoint["epoch"]) + 1
    best_metric = checkpoint.get("best_metric")
    best_monitor = checkpoint.get("best_monitor")
    if best_monitor is None and isinstance(checkpoint.get("config"), dict):
        best_monitor = checkpoint["config"].get("checkpoint", {}).get("monitor")
    if best_monitor is not None and best_monitor != monitor:
        logger.warning(
            "checkpoint best monitor is %s but current monitor is %s; resetting best metric",
            best_monitor,
            monitor,
        )
        best_metric = None
    logger.info("resumed from %s at epoch %d", resume_path, start_epoch)
    return start_epoch, best_metric


def save_training_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_metric,
    config,
    metrics=None,
):
    model_to_save = model.module if hasattr(model, "module") else model
    save_checkpoint(
        {
            "epoch": epoch,
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_metric": best_metric,
            "best_monitor": config.get("checkpoint", {}).get("monitor"),
            "metrics": metrics or {},
            "config": config,
        },
        path,
    )


DEFAULT_SELECTION_WEIGHTS = {
    "val_f1": 0.45,
    "val_iou": 0.25,
    "val_ap": 0.20,
    "val_mcc": 0.10,
}
DEFAULT_SELECTION_MAX_FPR = 0.05


def save_best_checkpoint_record(path, checkpoint_name, epoch, monitor, metric, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        checkpoint_name,
        f"epoch: {epoch}",
        f"{monitor}: {metric:.8f}",
    ]
    if metrics:
        for key in ["val_f1", "val_iou", "val_auc", "val_ap", "val_mcc", "val_fpr", "val_loss"]:
            if key in metrics:
                lines.append(f"{key}: {float(metrics[key]):.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_selection_metric(epoch_metrics, config):
    checkpoint_config = config.get("checkpoint", {})
    selection_config = checkpoint_config.get("selection", {})
    monitor = checkpoint_config.get("monitor", "val_f1")
    if not selection_config and monitor != "val_score":
        return

    score_name = selection_config.get("name", "val_score")
    weights = selection_config.get("weights", DEFAULT_SELECTION_WEIGHTS)
    max_fpr = float(selection_config.get("max_fpr", DEFAULT_SELECTION_MAX_FPR))
    fpr_key = selection_config.get("fpr_key", "val_fpr")
    score = 0.0
    missing = []

    if fpr_key not in epoch_metrics:
        missing.append(fpr_key)
    elif float(epoch_metrics[fpr_key]) > max_fpr:
        epoch_metrics[score_name] = float("-inf")
        epoch_metrics[f"{score_name}_eligible"] = 0.0
        return

    for key, weight in weights.items():
        if key not in epoch_metrics:
            missing.append(key)
            continue
        score += float(weight) * float(epoch_metrics[key])

    if missing:
        raise KeyError(f"Selection metric requires missing metrics: {missing}")
    epoch_metrics[score_name] = score
    epoch_metrics[f"{score_name}_eligible"] = 1.0


def metric_is_better(current, best, mode, min_delta=0.0):
    current = float(current)
    if not math.isfinite(current):
        return False
    if best is None:
        return True
    best = float(best)
    min_delta = max(0.0, float(min_delta))
    if mode == "max":
        return current > best + min_delta
    if mode == "min":
        return current < best - min_delta
    raise ValueError(f"Unsupported checkpoint mode: {mode}")


def should_stop_early(epoch, best_epoch, config):
    early_config = config.get("early_stopping", {})
    if not early_config.get("enabled", False):
        return False
    start_epoch_config = int(early_config.get("start_epoch", 1))
    patience = int(early_config.get("patience", 8))
    if epoch < start_epoch_config or best_epoch is None:
        return False
    return (epoch - best_epoch) >= patience


def write_tensorboard(writer, metrics, epoch):
    if writer is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(key, value, epoch)


def main():
    args = parse_args()
    config = load_config(args.config)
    timeout_minutes = config.get("device", {}).get("distributed_timeout_minutes", 120)
    distributed, local_rank, rank, _ = setup_distributed(timeout_minutes=timeout_minutes)
    set_seed(config.get("seed", 2026) + rank)

    output_dir = Path(config["experiment"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    curve_dir = output_dir / "curves"
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        curve_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, output_dir / "config.yaml")

    log_files = []
    if is_main_process(rank):
        log_files.append(log_dir / "train.log")
        server_log_file = config.get("logging", {}).get("server_log_file")
        if server_log_file:
            log_files.append(server_log_file)

    logger = setup_logger(
        "train",
        log_files=log_files,
        console=config.get("logging", {}).get("console", False) and is_main_process(rank),
    )
    if not is_main_process(rank):
        logger.disabled = True

    ensure_data_split_manifests(config, logger, rank, distributed)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    logger.info("using device=%s distributed=%s", device, distributed)

    train_loader, val_loader, train_sampler = build_dataloaders(config, distributed)
    logger.info("train samples=%d val samples=%d", len(train_loader.dataset), len(val_loader.dataset))
    logger.info(
        "train skipped_size_mismatch=%d val skipped_size_mismatch=%d",
        len(getattr(train_loader.dataset, "size_mismatch_records", [])),
        len(getattr(val_loader.dataset, "size_mismatch_records", [])),
    )

    model = build_model(config).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    criterion = build_loss(config)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    amp_enabled = config.get("training", {}).get("amp", True) and device.type == "cuda"
    scaler = build_grad_scaler(enabled=amp_enabled)
    epochs = config.get("scheduler", {}).get("epochs", 100)
    monitor = config.get("checkpoint", {}).get("monitor", "val_f1")
    mode = config.get("checkpoint", {}).get("mode", "max")
    min_delta = config.get("checkpoint", {}).get("min_delta", 0.0)

    resume_path = args.resume or config.get("checkpoint", {}).get("resume")
    start_epoch, best_metric = load_resume_if_needed(
        resume_path,
        model,
        optimizer,
        scheduler,
        scaler,
        device,
        logger,
        monitor,
    )

    writer = None
    if is_main_process(rank):
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=log_dir / "tensorboard")
        except ImportError:
            logger.warning("tensorboard is not installed; scalar summaries are disabled")

    history = []
    best_epoch = None

    try:
        for epoch in range(start_epoch, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            set_dataset_epoch(train_loader, epoch)
            set_dataset_epoch(val_loader, epoch)

            if is_main_process(rank):
                logger.info(
                    "epoch=%d robust_augmentation_strength=%.4f active_robust_ops=%s",
                    epoch,
                    get_dataset_robust_strength(train_loader),
                    ",".join(get_dataset_active_robust_ops(train_loader)) or "none",
                )

            train_metrics = train_one_epoch(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                loader=train_loader,
                device=device,
                epoch=epoch,
                config=config,
                logger=logger,
                rank=rank,
            )

            if scheduler is not None:
                scheduler.step()

            val_metrics = validate(
                model=model,
                criterion=criterion,
                loader=val_loader,
                device=device,
                epoch=epoch,
                config=config,
                output_dir=output_dir,
                rank=rank,
            )

            lr = optimizer.param_groups[0]["lr"]
            epoch_metrics = {"epoch": epoch, "lr": lr, **train_metrics, **val_metrics}
            add_selection_metric(epoch_metrics, config)
            history.append(epoch_metrics)
            stop_training = False

            if is_main_process(rank):
                logger.info("epoch=%d metrics=%s", epoch, epoch_metrics)
                write_tensorboard(writer, epoch_metrics, epoch)
                save_training_curves(history, curve_dir)

                current_metric = epoch_metrics.get(monitor)
                if current_metric is None:
                    raise KeyError(f"Monitor metric '{monitor}' not found in metrics: {epoch_metrics.keys()}")
                if (
                    monitor == config.get("checkpoint", {}).get("selection", {}).get("name", "val_score")
                    and epoch_metrics.get(f"{monitor}_eligible") == 0.0
                ):
                    logger.info(
                        "epoch=%d is not eligible for best checkpoint because val_fpr=%.6f exceeds max_fpr=%.6f",
                        epoch,
                        float(epoch_metrics.get("val_fpr", 0.0)),
                        float(config.get("checkpoint", {}).get("selection", {}).get("max_fpr", DEFAULT_SELECTION_MAX_FPR)),
                    )

                is_best = metric_is_better(current_metric, best_metric, mode, min_delta=min_delta)
                if is_best:
                    best_metric = current_metric
                    best_epoch = epoch

                checkpoint_path = checkpoint_dir / f"checkpoint-epoch{epoch}.pth"
                logger.info("saving epoch checkpoint: %s", checkpoint_path)
                save_training_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_metric,
                    config,
                    metrics=epoch_metrics,
                )
                logger.info("saved epoch checkpoint: %s", checkpoint_path)
                if is_best:
                    best_record_path = checkpoint_dir / "best_checkpoint.txt"
                    save_best_checkpoint_record(
                        best_record_path,
                        checkpoint_path.name,
                        epoch,
                        monitor,
                        current_metric,
                        metrics=epoch_metrics,
                    )
                    logger.info("updated best checkpoint record: %s", best_record_path)

                if should_stop_early(epoch, best_epoch, config):
                    logger.info(
                        "early stopping triggered at epoch=%d best_epoch=%d monitor=%s best_metric=%.8f",
                        epoch,
                        best_epoch,
                        monitor,
                        float(best_metric),
                    )
                    stop_training = True

            if distributed:
                stop_tensor = torch.tensor([1 if stop_training else 0], device=device)
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())

            if stop_training:
                break

            if distributed and config.get("training", {}).get("sync_after_epoch", False):
                dist.barrier()
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
