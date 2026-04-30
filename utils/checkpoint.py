from pathlib import Path

import torch


def save_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def resolve_checkpoint_path(path):
    path = Path(path)
    if path.suffix.lower() != ".txt":
        return path

    with path.open("r", encoding="utf-8") as handle:
        checkpoint_name = next((line.strip() for line in handle if line.strip()), "")

    if not checkpoint_name:
        raise ValueError(f"Checkpoint pointer file is empty: {path}")

    checkpoint_path = Path(checkpoint_name)
    if checkpoint_path.is_absolute():
        return checkpoint_path
    return path.parent / checkpoint_path


def load_checkpoint(path, map_location="cpu"):
    path = resolve_checkpoint_path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
