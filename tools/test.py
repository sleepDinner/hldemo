import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model
from utils.config import load_config
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Test tamper localization model.")
    parser.add_argument("--config", required=True, help="Path to a YAML test config.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 2026))
    model = build_model(config)
    print(f"Testing scaffold is ready for model: {model.__class__.__name__}")
    print(f"Config loaded from: {args.config}")
    print("Cross-dataset evaluation loop will be implemented after dataset verification.")


if __name__ == "__main__":
    main()

