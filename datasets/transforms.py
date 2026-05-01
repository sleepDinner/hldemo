import io
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


FLIP_LEFT_RIGHT = (
    Image.Transpose.FLIP_LEFT_RIGHT if hasattr(Image, "Transpose") else Image.FLIP_LEFT_RIGHT
)
FLIP_TOP_BOTTOM = (
    Image.Transpose.FLIP_TOP_BOTTOM if hasattr(Image, "Transpose") else Image.FLIP_TOP_BOTTOM
)


class RobustImageAugmentation:
    """Image-only degradations for post-processing robustness.

    These augmentations simulate common image transmission operations such as
    JPEG recompression, resizing, blur, and sensor/platform noise. They are
    applied only to the RGB image; masks remain unchanged as clean supervision.
    """

    def __init__(
        self,
        enabled=False,
        jpeg_prob=0.0,
        jpeg_quality=(70, 100),
        blur_prob=0.0,
        blur_radius=(0.3, 1.0),
        noise_prob=0.0,
        noise_std=(0.0, 0.015),
        resize_prob=0.0,
        resize_scale=(0.75, 1.25),
        max_ops_per_image=0,
        ops=None,
        schedule=None,
    ):
        self.enabled = enabled
        self.jpeg_prob = float(jpeg_prob)
        self.jpeg_quality = self._as_range(jpeg_quality, int)
        self.blur_prob = float(blur_prob)
        self.blur_radius = self._as_range(blur_radius, float)
        self.noise_prob = float(noise_prob)
        self.noise_std = self._as_range(noise_std, float)
        self.resize_prob = float(resize_prob)
        self.resize_scale = self._as_range(resize_scale, float)
        self.max_ops_per_image = max(0, int(max_ops_per_image or 0))
        self.schedule = schedule or {}
        self.current_epoch = 1
        self.schedule_enabled = bool(self.schedule.get("enabled", False))
        self.schedule_start_factor = float(self.schedule.get("start_factor", 1.0))
        self.schedule_end_factor = float(self.schedule.get("end_factor", 1.0))
        self.schedule_warmup_epochs = max(1, int(self.schedule.get("warmup_epochs", 1)))
        self.ops = self._build_ops(ops)

    @staticmethod
    def _as_range(value, value_type):
        if isinstance(value, (int, float)):
            return value_type(value), value_type(value)
        if len(value) != 2:
            raise ValueError(f"Expected range as [min, max], got: {value}")
        low = value_type(value[0])
        high = value_type(value[1])
        if low > high:
            low, high = high, low
        return low, high

    def _build_ops(self, ops):
        if ops:
            return self._build_configured_ops(ops)

        legacy_ops = []
        if self.jpeg_prob > 0:
            legacy_ops.append(
                {
                    "name": "jpeg",
                    "start_epoch": 1,
                    "prob": self.jpeg_prob,
                    "quality": self.jpeg_quality,
                }
            )
        if self.resize_prob > 0:
            legacy_ops.append(
                {
                    "name": "resize",
                    "start_epoch": 1,
                    "prob": self.resize_prob,
                    "scale": self.resize_scale,
                }
            )
        if self.blur_prob > 0:
            legacy_ops.append(
                {
                    "name": "blur",
                    "start_epoch": 1,
                    "prob": self.blur_prob,
                    "radius": self.blur_radius,
                }
            )
        if self.noise_prob > 0:
            legacy_ops.append(
                {
                    "name": "noise",
                    "start_epoch": 1,
                    "prob": self.noise_prob,
                    "std": self.noise_std,
                }
            )
        return legacy_ops

    @staticmethod
    def _build_configured_ops(ops):
        if isinstance(ops, list):
            configured_ops = []
            for item in ops:
                op = dict(item)
                if "name" not in op:
                    raise ValueError(f"Each robust augmentation op must define a name: {item}")
                configured_ops.append(op)
            return configured_ops

        configured_ops = []
        for name, params in ops.items():
            op = dict(params or {})
            op["name"] = name
            configured_ops.append(op)
        return configured_ops

    def __call__(self, image):
        if not self.enabled:
            return image

        selected_ops = self._sample_ops()
        for op in selected_ops:
            image = self._apply_op(image, op)
        return image

    def set_epoch(self, epoch):
        self.current_epoch = max(1, int(epoch))

    def get_strength_factor(self):
        if not self.enabled:
            return 0.0
        if not self.schedule_enabled:
            return 1.0

        if self.schedule_warmup_epochs <= 1:
            progress = 1.0
        else:
            progress = (self.current_epoch - 1) / (self.schedule_warmup_epochs - 1)
            progress = min(max(progress, 0.0), 1.0)
        factor = self.schedule_start_factor + progress * (
            self.schedule_end_factor - self.schedule_start_factor
        )
        return min(max(factor, 0.0), 1.0)

    def _scheduled_prob(self, probability):
        return min(max(float(probability) * self.get_strength_factor(), 0.0), 1.0)

    def get_active_operation_names(self):
        if not self.enabled:
            return []
        return [
            op["name"]
            for op in self.ops
            if self.current_epoch >= int(op.get("start_epoch", 1))
        ]

    def _sample_ops(self):
        sampled_ops = []
        for op in self.ops:
            if self.current_epoch < int(op.get("start_epoch", 1)):
                continue
            if random.random() < self._scheduled_prob(op.get("prob", 0.0)):
                sampled_ops.append(op)

        if self.max_ops_per_image > 0 and len(sampled_ops) > self.max_ops_per_image:
            sampled_ops = random.sample(sampled_ops, self.max_ops_per_image)
        return sampled_ops

    def _apply_op(self, image, op):
        name = op["name"].lower()
        if name == "jpeg":
            return self._jpeg_compress(image, op)
        if name == "resize":
            return self._resize_degradation(image, op)
        if name == "blur":
            return self._blur(image, op)
        if name == "noise":
            return self._add_noise(image, op)
        if name == "gamma":
            return self._adjust_gamma(image, op)
        if name == "contrast":
            return self._adjust_contrast(image, op)
        raise ValueError(f"Unsupported robust augmentation op: {name}")

    def _range_from_op(self, op, key, default, value_type):
        return self._as_range(op.get(key, default), value_type)

    def _scheduled_jpeg_quality(self, op):
        factor = self.get_strength_factor()
        min_quality, max_quality = self._range_from_op(op, "quality", self.jpeg_quality, int)
        scheduled_min = int(round(max_quality - (max_quality - min_quality) * factor))
        scheduled_min = min(max(scheduled_min, min_quality), max_quality)
        return scheduled_min, max_quality

    def _scheduled_positive_range(self, value_range):
        factor = self.get_strength_factor()
        low, high = value_range
        return max(0.0, low * factor), max(0.0, high * factor)

    def _scheduled_centered_range(self, value_range, center=1.0):
        factor = self.get_strength_factor()
        low, high = value_range
        scheduled_low = center - (center - low) * factor
        scheduled_high = center + (high - center) * factor
        if scheduled_low > scheduled_high:
            scheduled_low, scheduled_high = scheduled_high, scheduled_low
        return scheduled_low, scheduled_high

    def _scheduled_resize_scale(self, op):
        scale_range = self._range_from_op(op, "scale", self.resize_scale, float)
        return self._scheduled_centered_range(scale_range, center=1.0)

    def _jpeg_compress(self, image, op):
        min_quality, max_quality = self._scheduled_jpeg_quality(op)
        quality = random.randint(min_quality, max_quality)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")

    def _blur(self, image, op):
        radius_range = self._range_from_op(op, "radius", self.blur_radius, float)
        min_radius, max_radius = self._scheduled_positive_range(radius_range)
        radius = random.uniform(min_radius, max_radius)
        if radius <= 0:
            return image
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def _add_noise(self, image, op):
        std_range = self._range_from_op(op, "std", self.noise_std, float)
        min_std, max_std = self._scheduled_positive_range(std_range)
        std = random.uniform(min_std, max_std)
        if std <= 0:
            return image
        image_array = np.asarray(image, dtype=np.float32)
        noise = np.random.normal(loc=0.0, scale=std * 255.0, size=image_array.shape)
        noisy = np.clip(image_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, mode="RGB")

    def _resize_degradation(self, image, op):
        min_scale, max_scale = self._scheduled_resize_scale(op)
        scale = random.uniform(min_scale, max_scale)
        if abs(scale - 1.0) < 1e-3:
            return image

        width, height = image.size
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        degraded = image.resize((resized_width, resized_height), Image.BILINEAR)
        return degraded.resize((width, height), Image.BILINEAR)

    def _adjust_gamma(self, image, op):
        gamma_range = self._range_from_op(op, "gamma", (0.9, 1.1), float)
        min_gamma, max_gamma = self._scheduled_centered_range(gamma_range, center=1.0)
        gamma = random.uniform(min_gamma, max_gamma)
        if abs(gamma - 1.0) < 1e-3:
            return image

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        adjusted = np.power(np.clip(image_array, 0.0, 1.0), gamma)
        adjusted = np.clip(adjusted * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(adjusted, mode="RGB")

    def _adjust_contrast(self, image, op):
        factor_range = self._range_from_op(op, "factor", (0.9, 1.1), float)
        min_factor, max_factor = self._scheduled_centered_range(factor_range, center=1.0)
        factor = random.uniform(min_factor, max_factor)
        if abs(factor - 1.0) < 1e-3:
            return image
        return ImageEnhance.Contrast(image).enhance(factor)


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
        size_mismatch_policy="resize_mask",
        robust_augmentation=None,
        foreground_crop_prob=0.0,
        foreground_crop_min_ratio=0.0,
        foreground_crop_max_retries=10,
        foreground_crop_threshold=127,
    ):
        self.input_size = self._as_hw(input_size)
        self.mode = mode
        self.crop_size = self._as_hw(crop_size or input_size)
        self.random_crop = random_crop
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.mean = torch.tensor(mean or [0.0, 0.0, 0.0], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std or [1.0, 1.0, 1.0], dtype=torch.float32).view(3, 1, 1)
        self.size_mismatch_policy = size_mismatch_policy
        self.robust_augmentation = robust_augmentation or RobustImageAugmentation(enabled=False)
        self.foreground_crop_prob = float(foreground_crop_prob)
        self.foreground_crop_min_ratio = float(foreground_crop_min_ratio)
        self.foreground_crop_max_retries = max(1, int(foreground_crop_max_retries))
        self.foreground_crop_threshold = int(foreground_crop_threshold)

    def set_epoch(self, epoch):
        if hasattr(self.robust_augmentation, "set_epoch"):
            self.robust_augmentation.set_epoch(epoch)

    def get_robust_strength_factor(self):
        if hasattr(self.robust_augmentation, "get_strength_factor"):
            return self.robust_augmentation.get_strength_factor()
        return 0.0

    def get_active_robust_ops(self):
        if hasattr(self.robust_augmentation, "get_active_operation_names"):
            return self.robust_augmentation.get_active_operation_names()
        return []

    @staticmethod
    def _as_hw(size):
        if isinstance(size, int):
            return size, size
        if len(size) != 2:
            raise ValueError(f"Expected size as [H, W], got: {size}")
        return int(size[0]), int(size[1])

    def __call__(self, image, mask):
        if image.size != mask.size:
            if self.size_mismatch_policy == "resize_mask":
                mask = mask.resize(image.size, Image.NEAREST)
            elif self.size_mismatch_policy == "resize_image":
                image = image.resize(mask.size, Image.BILINEAR)
            elif self.size_mismatch_policy == "error":
                raise ValueError(f"Image and mask size mismatch: image={image.size}, mask={mask.size}")
            else:
                raise ValueError(f"Unsupported size_mismatch_policy: {self.size_mismatch_policy}")

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
            if self._should_use_foreground_crop(mask):
                image, mask = self._foreground_crop_pair(image, mask, self.crop_size)
            else:
                image, mask = self._random_crop_pair(image, mask, self.crop_size)

        if random.random() < self.hflip_prob:
            image = image.transpose(FLIP_LEFT_RIGHT)
            mask = mask.transpose(FLIP_LEFT_RIGHT)

        if random.random() < self.vflip_prob:
            image = image.transpose(FLIP_TOP_BOTTOM)
            mask = mask.transpose(FLIP_TOP_BOTTOM)

        image, mask = self._resize_pair(image, mask, self.input_size)
        image = self.robust_augmentation(image)
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

    def _should_use_foreground_crop(self, mask):
        if self.foreground_crop_prob <= 0.0 or random.random() >= self.foreground_crop_prob:
            return False
        return bool((np.asarray(mask) > self.foreground_crop_threshold).any())

    def _foreground_crop_pair(self, image, mask, crop_size):
        crop_h, crop_w = crop_size
        width, height = image.size
        if width == crop_w and height == crop_h:
            return image, mask

        mask_array = np.asarray(mask)
        foreground = np.argwhere(mask_array > self.foreground_crop_threshold)
        if foreground.size == 0:
            return self._random_crop_pair(image, mask, crop_size)

        min_pixels = max(1, int(round(crop_h * crop_w * self.foreground_crop_min_ratio)))
        for _ in range(self.foreground_crop_max_retries):
            y, x = foreground[random.randrange(len(foreground))]
            left_min = max(0, int(x) - crop_w + 1)
            left_max = min(int(x), width - crop_w)
            top_min = max(0, int(y) - crop_h + 1)
            top_max = min(int(y), height - crop_h)
            if left_min > left_max or top_min > top_max:
                continue

            left = random.randint(left_min, left_max)
            top = random.randint(top_min, top_max)
            box = (left, top, left + crop_w, top + crop_h)
            cropped_mask = mask.crop(box)
            if self.foreground_crop_min_ratio <= 0.0:
                return image.crop(box), cropped_mask
            if int((np.asarray(cropped_mask) > self.foreground_crop_threshold).sum()) >= min_pixels:
                return image.crop(box), cropped_mask

        return self._random_crop_pair(image, mask, crop_size)

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
    robust_config = augment_config.get("robust", {}) if mode == "train" else {}
    return TamperPairTransform(
        input_size=data_config["input_size"],
        mode=mode,
        crop_size=augment_config.get("crop_size", data_config["input_size"]),
        random_crop=augment_config.get("random_crop", mode == "train"),
        hflip_prob=augment_config.get("hflip_prob", 0.5 if mode == "train" else 0.0),
        vflip_prob=augment_config.get("vflip_prob", 0.0),
        mean=normalize.get("mean"),
        std=normalize.get("std"),
        size_mismatch_policy=data_config.get("size_mismatch_policy", "resize_mask"),
        robust_augmentation=RobustImageAugmentation(**robust_config),
        foreground_crop_prob=augment_config.get("foreground_crop_prob", 0.0),
        foreground_crop_min_ratio=augment_config.get("foreground_crop_min_ratio", 0.0),
        foreground_crop_max_retries=augment_config.get("foreground_crop_max_retries", 10),
        foreground_crop_threshold=augment_config.get("foreground_crop_threshold", 127),
    )
