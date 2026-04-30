import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from utils.warnings import suppress_pil_exif_warnings


suppress_pil_exif_warnings()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class SplitSummary:
    train_count: int
    val_count: int
    train_negative: int
    train_positive: int
    val_negative: int
    val_positive: int
    skipped_size_mismatch: int


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def collect_files(directory, recursive=False):
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_size(path):
    with Image.open(path) as handle:
        return handle.size


def mask_has_foreground(mask_path, threshold):
    with Image.open(mask_path) as handle:
        mask = handle.convert("L")
        mask_array = np.asarray(mask)
    return bool((mask_array > threshold).any())


def build_mask_index(mask_dir, recursive=False):
    mask_index = {}
    for mask_path in collect_files(mask_dir, recursive=recursive):
        key = mask_path.stem.lower()
        if key in mask_index:
            raise ValueError(f"Duplicate mask stem '{key}' found: {mask_index[key]} and {mask_path}")
        mask_index[key] = mask_path
    return mask_index


def find_mask_for_image(image_path, mask_index, mask_suffixes):
    image_stem = image_path.stem.lower()
    for suffix in mask_suffixes:
        candidate_key = f"{image_stem}{suffix}".lower()
        if candidate_key in mask_index:
            return mask_index[candidate_key]
    return None


def build_labeled_samples(source_config, threshold=127, skip_size_mismatch=True):
    image_dir = resolve_project_path(source_config["image_dir"])
    mask_dir = resolve_project_path(source_config["mask_dir"])
    recursive = bool(source_config.get("recursive", False))
    strict_pairs = bool(source_config.get("strict_pairs", True))
    mask_suffixes = source_config.get("mask_suffixes") or ["", "_mask", "_gt", "_label"]

    mask_index = build_mask_index(mask_dir, recursive=recursive)
    missing_masks = []
    skipped_size_mismatch = []
    samples = []

    for image_path in collect_files(image_dir, recursive=recursive):
        mask_path = find_mask_for_image(image_path, mask_index, mask_suffixes)
        if mask_path is None:
            missing_masks.append(image_path)
            continue

        is_positive = mask_has_foreground(mask_path, threshold=threshold)
        if is_positive and skip_size_mismatch and read_size(image_path) != read_size(mask_path):
            skipped_size_mismatch.append((image_path, mask_path))
            continue

        samples.append(
            {
                "image_path": image_path,
                "mask_path": mask_path,
                "class_label": 1 if is_positive else 0,
            }
        )

    if missing_masks and strict_pairs:
        preview = "\n".join(str(path) for path in missing_masks[:10])
        raise FileNotFoundError(
            f"Missing masks for {len(missing_masks)} images. First missing paths:\n{preview}"
        )
    if not samples:
        raise RuntimeError("No samples found for stratified split.")

    return samples, len(skipped_size_mismatch)


def split_by_label(samples, val_ratio, seed):
    negatives = [sample for sample in samples if sample["class_label"] == 0]
    positives = [sample for sample in samples if sample["class_label"] == 1]
    rng = random.Random(seed)
    rng.shuffle(negatives)
    rng.shuffle(positives)

    def split_group(group):
        if not group:
            return [], []
        val_count = int(round(len(group) * val_ratio))
        if len(group) > 1:
            val_count = min(max(1, val_count), len(group) - 1)
        return group[val_count:], group[:val_count]

    train_negatives, val_negatives = split_group(negatives)
    train_positives, val_positives = split_group(positives)
    train_samples = train_negatives + train_positives
    val_samples = val_negatives + val_positives
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def write_manifest(samples, path, separator=" # "):
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(
                separator.join(
                    [
                        str(sample["image_path"]),
                        str(sample["mask_path"]),
                        str(int(sample["class_label"])),
                    ]
                )
                + "\n"
            )
    return path


def create_stratified_split(split_config):
    source_config = split_config["source"]
    val_ratio = float(split_config.get("val_ratio", 0.1))
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got: {val_ratio}")

    threshold = int(split_config.get("threshold", 127))
    seed = int(split_config.get("seed", 2026))
    skip_size_mismatch = bool(split_config.get("skip_size_mismatch", True))
    separator = split_config.get("manifest_separator", " # ")

    samples, skipped_size_mismatch = build_labeled_samples(
        source_config,
        threshold=threshold,
        skip_size_mismatch=skip_size_mismatch,
    )
    train_samples, val_samples = split_by_label(samples, val_ratio=val_ratio, seed=seed)

    train_path = write_manifest(train_samples, split_config["train_manifest"], separator=separator)
    val_path = write_manifest(val_samples, split_config["val_manifest"], separator=separator)

    return train_path, val_path, SplitSummary(
        train_count=len(train_samples),
        val_count=len(val_samples),
        train_negative=sum(sample["class_label"] == 0 for sample in train_samples),
        train_positive=sum(sample["class_label"] == 1 for sample in train_samples),
        val_negative=sum(sample["class_label"] == 0 for sample in val_samples),
        val_positive=sum(sample["class_label"] == 1 for sample in val_samples),
        skipped_size_mismatch=skipped_size_mismatch,
    )


def ensure_stratified_split(config, logger=None):
    split_config = config.get("data", {}).get("split", {})
    if not split_config.get("enabled", False):
        return None

    train_path = resolve_project_path(split_config["train_manifest"])
    val_path = resolve_project_path(split_config["val_manifest"])
    overwrite = bool(split_config.get("overwrite", False))
    if train_path.exists() and val_path.exists() and not overwrite:
        if logger is not None:
            logger.info("using existing stratified manifests: train=%s val=%s", train_path, val_path)
        return None

    if logger is not None:
        source = split_config.get("source", {})
        logger.info(
            "creating stratified train/val manifests from image_dir=%s mask_dir=%s val_ratio=%.4f",
            source.get("image_dir"),
            source.get("mask_dir"),
            float(split_config.get("val_ratio", 0.1)),
        )
    train_path, val_path, summary = create_stratified_split(split_config)
    if logger is not None:
        logger.info(
            "created stratified manifests: train=%s val=%s train=%d neg=%d pos=%d "
            "val=%d neg=%d pos=%d skipped_size_mismatch=%d",
            train_path,
            val_path,
            summary.train_count,
            summary.train_negative,
            summary.train_positive,
            summary.val_count,
            summary.val_negative,
            summary.val_positive,
            summary.skipped_size_mismatch,
        )
    return summary
