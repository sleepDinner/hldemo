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
        coarse_weight=0.2,
        image_weight=0.1,
        boundary_kernel_size=5,
    ):
        super().__init__()
        self.mask_bce_weight = mask_bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.coarse_weight = coarse_weight
        self.image_weight = image_weight
        self.boundary_kernel_size = boundary_kernel_size
        self.dice = DiceLoss()

    def forward(self, outputs, targets, return_dict=False):
        mask_targets = targets["mask"].float()
        mask_logits = outputs["mask_logits"]

        mask_bce = F.binary_cross_entropy_with_logits(mask_logits, mask_targets)
        mask_dice = self.dice(mask_logits, mask_targets)
        total = self.mask_bce_weight * mask_bce + self.dice_weight * mask_dice

        losses = {
            "loss": total,
            "loss_mask_bce": mask_bce.detach(),
            "loss_mask_dice": mask_dice.detach(),
        }

        coarse_logits = outputs.get("coarse_mask_logits")
        if coarse_logits is not None and self.coarse_weight > 0:
            coarse_loss = F.binary_cross_entropy_with_logits(coarse_logits, mask_targets)
            total = total + self.coarse_weight * coarse_loss
            losses["loss_coarse"] = coarse_loss.detach()

        boundary_logits = outputs.get("boundary_logits")
        if boundary_logits is not None and self.boundary_weight > 0:
            boundary_targets = targets.get("boundary")
            if boundary_targets is None:
                boundary_targets = mask_to_boundary(mask_targets, kernel_size=self.boundary_kernel_size)
            boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets)
            total = total + self.boundary_weight * boundary_loss
            losses["loss_boundary"] = boundary_loss.detach()

        image_logits = outputs.get("image_logits")
        if image_logits is not None and self.image_weight > 0:
            image_targets = (mask_targets.flatten(1).amax(dim=1, keepdim=True) > 0).float()
            image_loss = F.binary_cross_entropy_with_logits(image_logits, image_targets)
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
        coarse_weight=loss_config.get("coarse_weight", 0.2),
        image_weight=loss_config.get("image_weight", 0.1),
        boundary_kernel_size=loss_config.get("boundary_kernel_size", 5),
    )
