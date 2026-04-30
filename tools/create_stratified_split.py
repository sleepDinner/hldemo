import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config
from utils.stratified_split import create_stratified_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create stratified train/val manifests from the configured training source."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    split_config = config.get("data", {}).get("split")
    if not split_config:
        raise KeyError("Config must define data.split.")

    train_path, val_path, summary = create_stratified_split(split_config)
    print(f"train_manifest: {train_path}")
    print(f"val_manifest: {val_path}")
    print(
        "train: total={total} negative={negative} positive={positive}".format(
            total=summary.train_count,
            negative=summary.train_negative,
            positive=summary.train_positive,
        )
    )
    print(
        "val: total={total} negative={negative} positive={positive}".format(
            total=summary.val_count,
            negative=summary.val_negative,
            positive=summary.val_positive,
        )
    )
    print(f"skipped_size_mismatch: {summary.skipped_size_mismatch}")


if __name__ == "__main__":
    main()
