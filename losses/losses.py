import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        targets = targets.flatten(1)
        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


def mask_to_boundary(mask, kernel_size=5):
    """Convert binary masks [B, 1, H, W] to boundary targets."""

    padding = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=padding)
    boundary = (dilated - eroded).clamp(0.0, 1.0)
    return boundary


class TamperLocalizationLoss(nn.Module):
    def __init__(
        self,
        mask_bce_weight=1.0,
        dice_weight=0.5,
        boundary_weight=0.3,
        boundary_dice_weight=0.5,
        coarse_weight=0.2,
        coarse_dice_weight=0.5,
        image_weight=0.1,
        boundary_kernel_size=5,
        bce_label_smoothing=0.0,
        boundary_label_smoothing=None,
        image_label_smoothing=None,
        mask_bce_mode="bce",
        mask_pos_weight=None,
        mask_pos_weight_auto=False,
        mask_pos_weight_max=20.0,
        mask_focal_alpha=0.75,
        mask_focal_gamma=2.0,
        logit_l2_weight=0.0,
    ):
        super().__init__()
        self.mask_bce_weight = mask_bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.boundary_dice_weight = boundary_dice_weight
        self.coarse_weight = coarse_weight
        self.coarse_dice_weight = coarse_dice_weight
        self.image_weight = image_weight
        self.boundary_kernel_size = boundary_kernel_size
        self.bce_label_smoothing = self._normalize_smoothing(bce_label_smoothing)
        self.boundary_label_smoothing = self._normalize_smoothing(
            bce_label_smoothing if boundary_label_smoothing is None else boundary_label_smoothing
        )
        self.image_label_smoothing = self._normalize_smoothing(
            bce_label_smoothing if image_label_smoothing is None else image_label_smoothing
        )
        self.mask_bce_mode = str(mask_bce_mode).lower()
        if self.mask_bce_mode not in {"bce", "focal"}:
            raise ValueError(f"Unsupported mask_bce_mode: {mask_bce_mode}")
        self.mask_pos_weight = self._normalize_optional_positive(mask_pos_weight, "mask_pos_weight")
        self.mask_pos_weight_auto = bool(mask_pos_weight_auto)
        self.mask_pos_weight_max = self._normalize_optional_positive(mask_pos_weight_max, "mask_pos_weight_max")
        self.mask_focal_alpha = self._normalize_optional_probability(mask_focal_alpha, "mask_focal_alpha")
        self.mask_focal_gamma = float(mask_focal_gamma)
        if self.mask_focal_gamma < 0.0:
            raise ValueError(f"mask_focal_gamma must be non-negative, got: {mask_focal_gamma}")
        self.logit_l2_weight = float(logit_l2_weight)
        if self.logit_l2_weight < 0.0:
            raise ValueError(f"logit_l2_weight must be non-negative, got: {logit_l2_weight}")
        self.dice = DiceLoss()

    @staticmethod
    def _normalize_smoothing(value):
        value = float(value)
        if value < 0.0 or value >= 0.5:
            raise ValueError(f"label smoothing must be in [0, 0.5), got: {value}")
        return value

    @staticmethod
    def _smooth_binary_targets(targets, smoothing):
        if smoothing <= 0.0:
            return targets
        return targets * (1.0 - smoothing) + (1.0 - targets) * smoothing

    @staticmethod
    def _normalize_optional_positive(value, name):
        if value is None:
            return None
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive when set, got: {value}")
        return value

    @staticmethod
    def _normalize_optional_probability(value, name):
        if value is None:
            return None
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1] when set, got: {value}")
        return value

    @staticmethod
    def _build_image_targets(mask_targets):
        """Build image-level labels from pixel masks.

        Resized masks can contain soft values, so threshold the pixel mask first
        and then mark an image as tampered if it contains any foreground pixel.
        This is less sensitive to interpolation noise than `amax > 0`.
        """

        foreground_ratio = (mask_targets > 0.5).float().flatten(1).mean(dim=1, keepdim=True)
        return (foreground_ratio > 1e-6).float()

    @staticmethod
    def _format_pos_weight(logits, value):
        if not torch.is_tensor(value):
            value = logits.new_tensor(value)
        value = value.to(device=logits.device, dtype=logits.dtype)
        return value.reshape((1,) * max(1, logits.dim() - 1))

    def _mask_pos_weight_tensor(self, logits, hard_targets):
        if self.mask_pos_weight is not None:
            return self._format_pos_weight(logits, self.mask_pos_weight)
        if not self.mask_pos_weight_auto:
            return None

        hard_targets = (hard_targets > 0.5).float()
        positives = hard_targets.sum()
        if float(positives.detach().item()) <= 0.0:
            return None

        negatives = hard_targets.numel() - positives
        pos_weight = negatives / positives.clamp_min(1.0)
        if self.mask_pos_weight_max is not None:
            pos_weight = pos_weight.clamp(max=self.mask_pos_weight_max)
        return self._format_pos_weight(logits, pos_weight.detach())

    def _mask_bce_loss(self, logits, targets, hard_targets):
        pos_weight = self._mask_pos_weight_tensor(logits, hard_targets)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        if self.mask_bce_mode == "bce":
            return bce.mean()

        probs = torch.sigmoid(logits)
        prob_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - prob_t).clamp_min(0.0).pow(self.mask_focal_gamma)
        if self.mask_focal_alpha is not None:
            alpha_t = self.mask_focal_alpha * targets + (1.0 - self.mask_focal_alpha) * (1.0 - targets)
            focal_weight = focal_weight * alpha_t
        return (bce * focal_weight).mean()

    def forward(self, outputs, targets, return_dict=False):
        mask_targets = targets["mask"].float()
        mask_logits = outputs["mask_logits"]
        mask_bce_targets = self._smooth_binary_targets(mask_targets, self.bce_label_smoothing)

        mask_bce = self._mask_bce_loss(mask_logits, mask_bce_targets, mask_targets)
        mask_dice = self.dice(mask_logits, mask_targets)
        total = self.mask_bce_weight * mask_bce + self.dice_weight * mask_dice

        losses = {
            "loss": total,
            "loss_mask_bce": mask_bce.detach(),
            "loss_mask_dice": mask_dice.detach(),
        }

        if self.logit_l2_weight > 0:
            logit_l2 = mask_logits.pow(2).mean()
            total = total + self.logit_l2_weight * logit_l2
            losses["loss_logit_l2"] = logit_l2.detach()

        coarse_logits = outputs.get("coarse_mask_logits")
        if coarse_logits is not None and self.coarse_weight > 0:
            coarse_bce = F.binary_cross_entropy_with_logits(coarse_logits, mask_bce_targets)
            coarse_dice = self.dice(coarse_logits, mask_targets)
            coarse_loss = coarse_bce + self.coarse_dice_weight * coarse_dice
            total = total + self.coarse_weight * coarse_loss
            losses["loss_coarse"] = coarse_loss.detach()
            losses["loss_coarse_bce"] = coarse_bce.detach()
            losses["loss_coarse_dice"] = coarse_dice.detach()

        boundary_logits = outputs.get("boundary_logits")
        if boundary_logits is not None and self.boundary_weight > 0:
            boundary_targets = targets.get("boundary")
            if boundary_targets is None:
                boundary_targets = mask_to_boundary(mask_targets, kernel_size=self.boundary_kernel_size)
            boundary_bce_targets = self._smooth_binary_targets(
                boundary_targets,
                self.boundary_label_smoothing,
            )
            boundary_bce = F.binary_cross_entropy_with_logits(boundary_logits, boundary_bce_targets)
            boundary_dice = self.dice(boundary_logits, boundary_targets)
            boundary_loss = boundary_bce + self.boundary_dice_weight * boundary_dice
            total = total + self.boundary_weight * boundary_loss
            losses["loss_boundary"] = boundary_loss.detach()
            losses["loss_boundary_bce"] = boundary_bce.detach()
            losses["loss_boundary_dice"] = boundary_dice.detach()

        image_logits = outputs.get("image_logits")
        if image_logits is not None and self.image_weight > 0:
            image_targets = self._build_image_targets(mask_targets)
            image_bce_targets = self._smooth_binary_targets(
                image_targets,
                self.image_label_smoothing,
            )
            image_loss = F.binary_cross_entropy_with_logits(image_logits, image_bce_targets)
            total = total + self.image_weight * image_loss
            losses["loss_image"] = image_loss.detach()

        losses["loss"] = total
        if return_dict:
            return losses
        return total


def build_loss(config):
    loss_config = config.get("loss", {})
    return TamperLocalizationLoss(
        mask_bce_weight=loss_config.get("mask_bce_weight", 1.0),
        dice_weight=loss_config.get("dice_weight", 0.5),
        boundary_weight=loss_config.get("boundary_weight", 0.3),
        boundary_dice_weight=loss_config.get("boundary_dice_weight", 0.5),
        coarse_weight=loss_config.get("coarse_weight", 0.2),
        coarse_dice_weight=loss_config.get("coarse_dice_weight", 0.5),
        image_weight=loss_config.get("image_weight", 0.1),
        boundary_kernel_size=loss_config.get("boundary_kernel_size", 5),
        bce_label_smoothing=loss_config.get("bce_label_smoothing", 0.0),
        boundary_label_smoothing=loss_config.get("boundary_label_smoothing"),
        image_label_smoothing=loss_config.get("image_label_smoothing"),
        mask_bce_mode=loss_config.get("mask_bce_mode", "bce"),
        mask_pos_weight=loss_config.get("mask_pos_weight"),
        mask_pos_weight_auto=loss_config.get("mask_pos_weight_auto", False),
        mask_pos_weight_max=loss_config.get("mask_pos_weight_max", 20.0),
        mask_focal_alpha=loss_config.get("mask_focal_alpha", 0.75),
        mask_focal_gamma=loss_config.get("mask_focal_gamma", 2.0),
        logit_l2_weight=loss_config.get("logit_l2_weight", 0.0),
    )
