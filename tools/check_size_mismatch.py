import argparse
import csv
import sys
from pathlib import Path

from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.check_mask_balance import build_samples, get_split_config
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check original image/mask size consistency for a configured dataset split."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--split", default="train", help="Split name, usually train/val/test.")
    parser.add_argument("--dataset", default=None, help="Dataset name when checking data.test_sets.")
    parser.add_argument("--show-examples", type=int, default=20, help="Number of mismatch examples to print.")
    parser.add_argument("--csv", default=None, help="Optional CSV path to save all mismatch records.")
    return parser.parse_args()


def image_size(path):
    with Image.open(path) as handle:
        return handle.size


def check_sample_size(sample):
    image_path = Path(sample["image_path"])
    mask_path = sample.get("mask_path")

    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    img_size = image_size(image_path)

    if sample.get("class_label", 1) == 0 or mask_path is None:
        return {
            "image_path": str(image_path),
            "mask_path": "null",
            "image_size": img_size,
            "mask_size": img_size,
            "matched": True,
            "has_real_mask": False,
        }

    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file does not exist: {mask_path}")

    mask_size = image_size(mask_path)
    return {
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "image_size": img_size,
        "mask_size": mask_size,
        "matched": img_size == mask_size,
        "has_real_mask": True,
    }


def format_size(size):
    width, height = size
    return f"{width}x{height}"


def save_mismatches_csv(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "mask_path", "image_size", "mask_size"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_path": record["image_path"],
                    "mask_path": record["mask_path"],
                    "image_size": format_size(record["image_size"]),
                    "mask_size": format_size(record["mask_size"]),
                }
            )


def main():
    args = parse_args()
    config = load_config(args.config)
    split_config = get_split_config(config, args.split, args.dataset)
    samples = build_samples(split_config)

    matched_count = 0
    mismatch_count = 0
    null_mask_count = 0
    mismatch_records = []

    iterator = samples
    if tqdm is not None:
        iterator = tqdm(samples, desc=f"check size {split_config.get('name', args.split)}", dynamic_ncols=True)

    for sample in iterator:
        record = check_sample_size(sample)
        if not record["has_real_mask"]:
            null_mask_count += 1
        if record["matched"]:
            matched_count += 1
        else:
            mismatch_count += 1
            mismatch_records.append(record)

    total = len(samples)
    mismatch_ratio = mismatch_count / total if total > 0 else 0.0
    real_mask_total = total - null_mask_count
    real_mask_mismatch_ratio = mismatch_count / real_mask_total if real_mask_total > 0 else 0.0

    print(f"总样本数量: {total}")
    print(f"真实 mask 文件数量: {real_mask_total}")
    print(f"null/真实图无 mask 数量: {null_mask_count}")
    print(f"尺寸一致数量: {matched_count}")
    print(f"尺寸不一致数量: {mismatch_count}")
    print(f"尺寸不一致比例: {mismatch_ratio:.6f} ({mismatch_ratio * 100:.4f}%)")
    print(
        "真实 mask 中尺寸不一致比例: "
        f"{real_mask_mismatch_ratio:.6f} ({real_mask_mismatch_ratio * 100:.4f}%)"
    )

    if mismatch_records and args.show_examples > 0:
        print(f"尺寸不一致样例 Top {min(args.show_examples, len(mismatch_records))}:")
        for record in mismatch_records[: args.show_examples]:
            print(
                "  "
                f"image_size={format_size(record['image_size'])} "
                f"mask_size={format_size(record['mask_size'])} "
                f"image={record['image_path']} "
                f"mask={record['mask_path']}"
            )

    if args.csv:
        save_mismatches_csv(mismatch_records, args.csv)
        print(f"尺寸不一致明细已保存: {args.csv}")


if __name__ == "__main__":
    main()
