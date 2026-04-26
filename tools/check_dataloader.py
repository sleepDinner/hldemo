import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader

from datasets.tamper_dataset import build_dataset_from_split_config
from datasets.transforms import build_transforms
from utils.config import load_config
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Check whether TamperDataset can load one batch.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split to check.")
    parser.add_argument("--dataset", default=None, help="Dataset name for test_sets configs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for the check.")
    return parser.parse_args()


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


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 2026))

    mode = "train" if args.split == "train" else "test"
    split_config = get_split_config(config, args.split, args.dataset)
    transform = build_transforms(config, mode=mode)
    dataset = build_dataset_from_split_config(split_config, transform=transform, mode=mode)

    batch_size = args.batch_size or config["data"].get("batch_size", 1)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    batch = next(iter(loader))

    print(f"Dataset: {split_config.get('name', args.split)}")
    print(f"Samples: {len(dataset)}")
    print(f"image shape: {tuple(batch['image'].shape)}, dtype: {batch['image'].dtype}")
    print(f"mask shape: {tuple(batch['mask'].shape)}, dtype: {batch['mask'].dtype}")
    print(f"mask min/max: {batch['mask'].min().item():.1f}/{batch['mask'].max().item():.1f}")
    print(f"first image: {batch['image_path'][0]}")


if __name__ == "__main__":
    main()
