import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation experiments.")
    parser.add_argument("--config", required=True, help="Path to a YAML ablation config.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 2026))
    ablations = config.get("ablations", [])
    print(f"Ablation scaffold is ready. Planned variants: {len(ablations)}")
    for ablation in ablations:
        print(f"- {ablation['name']}")


if __name__ == "__main__":
    main()

