from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class TamperDataset(Dataset):
    """Paired image-mask dataset for tamper localization.

    Two input formats are supported:
    1. Directory mode: image_dir + mask_dir.
    2. Manifest mode: manifest_file or manifest_files.

    Manifest lines can be:
    image_path # mask_path # class_id
    image_path,mask_path,class_id

    If class_id is 0 or mask_path is null, a zero mask is generated when
    include_authentic is true; otherwise that sample is skipped.
    """

    def __init__(
        self,
        image_dir=None,
        mask_dir=None,
        manifest_file=None,
        manifest_files=None,
        root=None,
        manifest_separator=" # ",
        include_authentic=True,
        skip_first=0,
        transform=None,
        mode="train",
        recursive=False,
        mask_suffixes=None,
        strict_pairs=True,
    ):
        self.root = Path(root) if root else None
        self.image_dir = self._resolve_path(image_dir) if image_dir else None
        self.mask_dir = self._resolve_path(mask_dir) if mask_dir else None
        self.manifest_files = self._normalize_manifest_files(manifest_file, manifest_files)
        self.manifest_separator = manifest_separator
        self.include_authentic = include_authentic
        self.skip_first = skip_first
        self.transform = transform
        self.mode = mode
        self.recursive = recursive
        self.mask_suffixes = mask_suffixes or ["", "_mask", "_gt", "_label"]
        self.strict_pairs = strict_pairs
        self.samples = self._build_samples()

    @staticmethod
    def _normalize_manifest_files(manifest_file, manifest_files):
        files = []
        if manifest_file:
            files.append(manifest_file)
        if manifest_files:
            if isinstance(manifest_files, (str, Path)):
                files.append(manifest_files)
            else:
                files.extend(manifest_files)
        return [Path(path) for path in files]

    def _build_samples(self):
        if self.manifest_files:
            return self._build_manifest_samples()
        return self._build_directory_samples()

    def _resolve_path(self, path, manifest_path=None):
        path = Path(path)
        if path.is_absolute():
            return path
        if self.root is not None:
            return self.root / path
        if manifest_path is not None:
            return manifest_path.parent / path
        return path

    def _collect_files(self, directory):
        if not directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory}")
        iterator = directory.rglob("*") if self.recursive else directory.iterdir()
        return sorted(
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _build_mask_index(self):
        mask_index = {}
        for mask_path in self._collect_files(self.mask_dir):
            key = mask_path.stem.lower()
            if key in mask_index:
                raise ValueError(
                    f"Duplicate mask stem '{key}' found: {mask_index[key]} and {mask_path}"
                )
            mask_index[key] = mask_path
        return mask_index

    def _find_mask_for_image(self, image_path, mask_index):
        image_stem = image_path.stem.lower()
        for suffix in self.mask_suffixes:
            candidate_key = f"{image_stem}{suffix}".lower()
            if candidate_key in mask_index:
                return mask_index[candidate_key]
        return None

    def _build_directory_samples(self):
        if self.image_dir is None or self.mask_dir is None:
            raise ValueError(
                "TamperDataset requires either image_dir + mask_dir or manifest_file/manifest_files."
            )

        image_paths = self._collect_files(self.image_dir)
        mask_index = self._build_mask_index()
        samples = []
        missing_masks = []

        for image_path in image_paths:
            mask_path = self._find_mask_for_image(image_path, mask_index)
            if mask_path is None:
                missing_masks.append(image_path)
                continue
            samples.append(
                {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "class_label": 1,
                    "source": "directory",
                }
            )

        if missing_masks and self.strict_pairs:
            preview = "\n".join(str(path) for path in missing_masks[:10])
            raise FileNotFoundError(
                f"Missing masks for {len(missing_masks)} images. First missing paths:\n{preview}"
            )

        if not samples:
            raise RuntimeError(
                f"No image-mask pairs found. image_dir={self.image_dir}, mask_dir={self.mask_dir}"
            )

        return samples

    def _build_manifest_samples(self):
        samples = []
        for manifest_file in self.manifest_files:
            manifest_path = self._resolve_path(manifest_file)
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")

            with manifest_path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()

            for line_number, raw_line in enumerate(lines[self.skip_first :], start=self.skip_first + 1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                sample = self._parse_manifest_line(line, manifest_path, line_number)
                is_authentic = sample["class_label"] == 0 or sample["mask_path"] is None
                if is_authentic and not self.include_authentic:
                    continue
                samples.append(sample)

        if not samples:
            raise RuntimeError(f"No samples found in manifest files: {self.manifest_files}")

        return samples

    def _parse_manifest_line(self, line, manifest_path, line_number):
        if self.manifest_separator and self.manifest_separator in line:
            parts = [part.strip() for part in line.split(self.manifest_separator, maxsplit=2)]
        else:
            parts = [part.strip() for part in line.split(",", maxsplit=2)]

        if len(parts) < 2:
            raise ValueError(f"Invalid manifest line {line_number} in {manifest_path}: {line}")

        image_path = self._resolve_path(parts[0], manifest_path)
        mask_text = parts[1]
        mask_path = None if mask_text.lower() == "null" else self._resolve_path(mask_text, manifest_path)
        class_label = self._parse_class_label(parts, mask_path)
        if mask_path is None:
            class_label = 0

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image path from manifest does not exist at line {line_number}: {image_path}"
            )
        if class_label != 0 and mask_path is not None and not mask_path.exists():
            raise FileNotFoundError(
                f"Mask path from manifest does not exist at line {line_number}: {mask_path}"
            )

        return {
            "image_path": image_path,
            "mask_path": mask_path,
            "class_label": class_label,
            "source": str(manifest_path),
        }

    @staticmethod
    def _parse_class_label(parts, mask_path):
        if len(parts) >= 3 and parts[2] != "":
            return int(parts[2])
        return 0 if mask_path is None else 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]
        class_label = sample["class_label"]

        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
        if class_label == 0 or mask_path is None:
            mask = Image.fromarray(np.zeros((image.height, image.width), dtype=np.uint8))
        else:
            with Image.open(mask_path) as mask_handle:
                mask = mask_handle.convert("L")

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image, mask = self._to_tensor(image, mask)

        return {
            "image": image,
            "mask": mask,
            "image_path": str(image_path),
            "mask_path": "null" if mask_path is None else str(mask_path),
            "class_label": class_label,
        }

    @staticmethod
    def _to_tensor(image, mask):
        image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mask_array = (np.asarray(mask) > 127).astype(np.float32)
        image_tensor = torch.from_numpy(image_array)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor


def build_dataset_from_config(config, split, transform=None):
    data_config = config["data"]
    split_config = data_config[split]
    return build_dataset_from_split_config(split_config, transform, mode="train" if split == "train" else "test")


def build_dataset_from_split_config(split_config, transform=None, mode="test"):
    return TamperDataset(
        image_dir=split_config.get("image_dir"),
        mask_dir=split_config.get("mask_dir"),
        manifest_file=split_config.get("manifest_file"),
        manifest_files=split_config.get("manifest_files"),
        root=split_config.get("root"),
        manifest_separator=split_config.get("manifest_separator", " # "),
        include_authentic=split_config.get("include_authentic", True),
        skip_first=split_config.get("skip_first", 0),
        transform=transform,
        mode=mode,
        recursive=split_config.get("recursive", False),
        mask_suffixes=split_config.get("mask_suffixes"),
        strict_pairs=split_config.get("strict_pairs", True),
    )
