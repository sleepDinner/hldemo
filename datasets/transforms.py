import random

import numpy as np
import torch
from PIL import Image, ImageOps


FLIP_LEFT_RIGHT = (
    Image.Transpose.FLIP_LEFT_RIGHT if hasattr(Image, "Transpose") else Image.FLIP_LEFT_RIGHT
)
FLIP_TOP_BOTTOM = (
    Image.Transpose.FLIP_TOP_BOTTOM if hasattr(Image, "Transpose") else Image.FLIP_TOP_BOTTOM
)


class TamperPairTransform:
    """Synchronized image-mask transform for tamper localization."""

    def __init__(
        self,
        input_size,
        mode="train",
        crop_size=None,
        random_crop=True,
        hflip_prob=0.5,
        vflip_prob=0.0,
        mean=None,
        std=None,
    ):
        self.input_size = self._as_hw(input_size)
        self.mode = mode
        self.crop_size = self._as_hw(crop_size or input_size)
        self.random_crop = random_crop
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.mean = torch.tensor(mean or [0.0, 0.0, 0.0], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std or [1.0, 1.0, 1.0], dtype=torch.float32).view(3, 1, 1)

    @staticmethod
    def _as_hw(size):
        if isinstance(size, int):
            return size, size
        if len(size) != 2:
            raise ValueError(f"Expected size as [H, W], got: {size}")
        return int(size[0]), int(size[1])

    def __call__(self, image, mask):
        if image.size != mask.size:
            raise ValueError(f"Image and mask size mismatch: image={image.size}, mask={mask.size}")

        if self.mode == "train":
            image, mask = self._train_transform(image, mask)
        else:
            image, mask = self._resize_pair(image, mask, self.input_size)

        image_tensor = self._image_to_tensor(image)
        mask_tensor = self._mask_to_tensor(mask)
        return image_tensor, mask_tensor

    def _train_transform(self, image, mask):
        if self.random_crop:
            image, mask = self._pad_if_needed(image, mask, self.crop_size)
            image, mask = self._random_crop_pair(image, mask, self.crop_size)

        if random.random() < self.hflip_prob:
            image = image.transpose(FLIP_LEFT_RIGHT)
            mask = mask.transpose(FLIP_LEFT_RIGHT)

        if random.random() < self.vflip_prob:
            image = image.transpose(FLIP_TOP_BOTTOM)
            mask = mask.transpose(FLIP_TOP_BOTTOM)

        image, mask = self._resize_pair(image, mask, self.input_size)
        return image, mask

    @staticmethod
    def _pad_if_needed(image, mask, crop_size):
        crop_h, crop_w = crop_size
        width, height = image.size
        pad_w = max(crop_w - width, 0)
        pad_h = max(crop_h - height, 0)

        if pad_w == 0 and pad_h == 0:
            return image, mask

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        border = (left, top, right, bottom)
        image = ImageOps.expand(image, border=border, fill=0)
        mask = ImageOps.expand(mask, border=border, fill=0)
        return image, mask

    @staticmethod
    def _random_crop_pair(image, mask, crop_size):
        crop_h, crop_w = crop_size
        width, height = image.size

        if width == crop_w and height == crop_h:
            return image, mask

        left = random.randint(0, width - crop_w)
        top = random.randint(0, height - crop_h)
        box = (left, top, left + crop_w, top + crop_h)
        return image.crop(box), mask.crop(box)

    @staticmethod
    def _resize_pair(image, mask, output_size):
        height, width = output_size
        image = image.resize((width, height), Image.BILINEAR)
        mask = mask.resize((width, height), Image.NEAREST)
        return image, mask

    def _image_to_tensor(self, image):
        image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        image_tensor = torch.from_numpy(image_array)
        return (image_tensor - self.mean) / self.std

    @staticmethod
    def _mask_to_tensor(mask):
        mask_array = (np.asarray(mask) > 127).astype(np.float32)
        return torch.from_numpy(mask_array).unsqueeze(0)


def build_transforms(config, mode):
    data_config = config["data"]
    normalize = data_config.get("normalize", {})
    augment_config = data_config.get("augmentation", {}).get(mode, {})
    return TamperPairTransform(
        input_size=data_config["input_size"],
        mode=mode,
        crop_size=augment_config.get("crop_size", data_config["input_size"]),
        random_crop=augment_config.get("random_crop", mode == "train"),
        hflip_prob=augment_config.get("hflip_prob", 0.5 if mode == "train" else 0.0),
        vflip_prob=augment_config.get("vflip_prob", 0.0),
        mean=normalize.get("mean"),
        std=normalize.get("std"),
    )
