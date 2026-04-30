import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config
from utils.warnings import suppress_pil_exif_warnings


suppress_pil_exif_warnings()


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_casia_manifest.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NULL_MASK_VALUES = {"", "null", "none", "nan"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check how many all-zero masks and non-zero masks exist in a configured split."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to YAML config. Defaults to the current training config.",
    )
    parser.add_argument("--split", default="train", help="Split name, usually train/val/test.")
    parser.add_argument("--dataset", default=None, help="Dataset name when checking data.test_sets.")
    parser.add_argument("--threshold", type=int, default=127, help="Mask foreground threshold in [0, 255].")
    parser.add_argument("--show-examples", type=int, default=0, help="Print a few zero/non-zero examples.")
    return parser.parse_args()


def resolve_path(path, root=None, manifest_path=None):
    path = Path(path)
    if path.is_absolute():
        return path
    if root:
        return Path(root) / path
    if manifest_path is not None:
        return manifest_path.parent / path
    return path


def normalize_manifest_files(split_config):
    files = []
    manifest_file = split_config.get("manifest_file")
    manifest_files = split_config.get("manifest_files")
    if manifest_file:
        files.append(manifest_file)
    if manifest_files:
        if isinstance(manifest_files, (str, Path)):
            files.append(manifest_files)
        else:
            files.extend(manifest_files)
    return files


def parse_class_label(parts, mask_path):
    if len(parts) >= 3 and parts[2] != "":
        return int(parts[2])
    return 0 if mask_path is None else 1


def parse_manifest_line(line, manifest_path, split_config, line_number):
    separator = split_config.get("manifest_separator", " # ")
    if separator and separator in line:
        parts = [part.strip() for part in line.split(separator, maxsplit=2)]
    else:
        parts = [part.strip() for part in line.split(",", maxsplit=2)]

    if len(parts) < 2:
        raise ValueError(f"Invalid manifest line {line_number} in {manifest_path}: {line}")

    root = split_config.get("root")
    image_path = resolve_path(parts[0], root=root, manifest_path=manifest_path)
    mask_text = parts[1].strip()
    mask_path = None if mask_text.lower() in NULL_MASK_VALUES else resolve_path(
        mask_text,
        root=root,
        manifest_path=manifest_path,
    )
    class_label = parse_class_label(parts, mask_path)
    if mask_path is None:
        class_label = 0

    return {
        "image_path": image_path,
        "mask_path": mask_path,
        "class_label": class_label,
        "source": str(manifest_path),
        "line_number": line_number,
    }


def build_manifest_samples(split_config):
    samples = []
    manifest_files = normalize_manifest_files(split_config)
    include_authentic = split_config.get("include_authentic", True)
    skip_first = int(split_config.get("skip_first", 0))

    for manifest_file in manifest_files:
        manifest_path = resolve_path(manifest_file, root=split_config.get("root"))
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()

        for line_number, raw_line in enumerate(lines[skip_first:], start=skip_first + 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            sample = parse_manifest_line(line, manifest_path, split_config, line_number)
            is_authentic = sample["class_label"] == 0 or sample["mask_path"] is None
            if is_authentic and not include_authentic:
                continue
            samples.append(sample)

    return samples


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


def build_directory_samples(split_config):
    image_dir = resolve_path(split_config["image_dir"], root=split_config.get("root"))
    mask_dir = resolve_path(split_config["mask_dir"], root=split_config.get("root"))
    recursive = split_config.get("recursive", False)
    mask_suffixes = split_config.get("mask_suffixes") or ["", "_mask", "_gt", "_label"]
    strict_pairs = split_config.get("strict_pairs", True)

    mask_index = {}
    for mask_path in collect_files(mask_dir, recursive=recursive):
        key = mask_path.stem.lower()
        if key in mask_index:
            raise ValueError(f"Duplicate mask stem '{key}' found: {mask_index[key]} and {mask_path}")
        mask_index[key] = mask_path

    samples = []
    missing_masks = []
    for image_path in collect_files(image_dir, recursive=recursive):
        image_stem = image_path.stem.lower()
        mask_path = None
        for suffix in mask_suffixes:
            candidate_key = f"{image_stem}{suffix}".lower()
            if candidate_key in mask_index:
                mask_path = mask_index[candidate_key]
                break
        if mask_path is None:
            missing_masks.append(image_path)
            continue
        samples.append({"image_path": image_path, "mask_path": mask_path, "class_label": 1})

    if missing_masks and strict_pairs:
        preview = "\n".join(str(path) for path in missing_masks[:10])
        raise FileNotFoundError(f"Missing masks for {len(missing_masks)} images. First missing paths:\n{preview}")

    return samples


def build_samples(split_config):
    if normalize_manifest_files(split_config):
        return build_manifest_samples(split_config)
    return build_directory_samples(split_config)


def read_image_size(path):
    with Image.open(path) as handle:
        return handle.size


def filter_size_mismatch_samples(samples):
    filtered_samples = []
    skipped = []
    for sample in samples:
        mask_path = sample["mask_path"]
        if sample["class_label"] == 0 or mask_path is None:
            filtered_samples.append(sample)
            continue

        image_size = read_image_size(sample["image_path"])
        mask_size = read_image_size(mask_path)
        if image_size == mask_size:
            filtered_samples.append(sample)
        else:
            skipped.append(
                {
                    "image_path": sample["image_path"],
                    "mask_path": mask_path,
                    "image_size": image_size,
                    "mask_size": mask_size,
                }
            )
    return filtered_samples, skipped


def get_split_config(config, split, dataset_name=None):
    data_config = config["data"]
    if split in data_config:
        return data_config[split]

    if split == "test" and "test_sets" in data_config:
        test_sets = data_config["test_sets"]
        if dataset_name is None:
            return test_sets[0]
        for item in test_sets:
            if item["name"] == dataset_name:
                return item
        raise ValueError(f"Dataset '{dataset_name}' not found in data.test_sets.")

    raise KeyError(f"Split '{split}' is not defined in config['data'].")


def mask_is_all_zero(sample, threshold):
    mask_path = sample["mask_path"]
    if sample["class_label"] == 0 or mask_path is None:
        return True

    if not Path(mask_path).exists():
        raise FileNotFoundError(f"Mask file does not exist: {mask_path}")

    with Image.open(mask_path) as handle:
        mask = handle.convert("L")
        mask_array = np.asarray(mask)
    return not bool((mask_array > threshold).any())


def main():
    args = parse_args()
    config = load_config(args.config)
    split_config = get_split_config(config, args.split, args.dataset)
    samples = build_samples(split_config)
    skipped_size_mismatch = []
    if split_config.get("skip_size_mismatch", False):
        samples, skipped_size_mismatch = filter_size_mismatch_samples(samples)

    zero_count = 0
    nonzero_count = 0
    zero_examples = []
    nonzero_examples = []

    iterator = samples
    if tqdm is not None:
        iterator = tqdm(samples, desc=f"check {split_config.get('name', args.split)}", dynamic_ncols=True)

    for sample in iterator:
        is_zero = mask_is_all_zero(sample, threshold=args.threshold)
        if is_zero:
            zero_count += 1
            if len(zero_examples) < args.show_examples:
                zero_examples.append(sample)
        else:
            nonzero_count += 1
            if len(nonzero_examples) < args.show_examples:
                nonzero_examples.append(sample)

    print(f"检查配置: {args.config}")
    print(f"检查 split: {args.split}")
    print(f"检查数据集: {split_config.get('name', args.split)}")
    if split_config.get("image_dir"):
        print(f"image_dir: {resolve_path(split_config['image_dir'], root=split_config.get('root'))}")
    if split_config.get("mask_dir"):
        print(f"mask_dir: {resolve_path(split_config['mask_dir'], root=split_config.get('root'))}")
    if normalize_manifest_files(split_config):
        print(f"manifest_files: {', '.join(str(path) for path in normalize_manifest_files(split_config))}")
    if skipped_size_mismatch:
        print(f"已按训练配置跳过尺寸不一致样本: {len(skipped_size_mismatch)}")
    print(f"总 mask 数量: {len(samples)}")
    print(f"真实图负样本数量 mask全0: {zero_count}")
    print(f"篡改图正样本数量 mask非0: {nonzero_count}")

    if args.show_examples > 0:
        print("mask全0样例:")
        for sample in zero_examples:
            print(f"  image={sample['image_path']} mask={sample['mask_path']}")
        print("mask非0样例:")
        for sample in nonzero_examples:
            print(f"  image={sample['image_path']} mask={sample['mask_path']}")


if __name__ == "__main__":
    main()
