from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_from_probs(p: torch.Tensor, y: torch.Tensor, eps: float = 1e-6):
    inter = (p * y).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
    return ((2.0 * inter + eps) / (denom + eps)).mean()


def dice_loss_from_logits(logits: torch.Tensor, y: torch.Tensor):
    p = torch.sigmoid(logits)
    return 1.0 - dice_from_probs(p, y)


def mask_to_boundary(mask: torch.Tensor):
    dil = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    ero = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    boundary = (dil - ero).clamp(0.0, 1.0)
    return (boundary > 0.0).float()


def mask_to_core(mask: torch.Tensor, kernel_size: int = 5, iterations: int = 2):
    core = mask
    pad = kernel_size // 2
    for _ in range(iterations):
        core = -F.max_pool2d(-core, kernel_size=kernel_size, stride=1, padding=pad)
    return (core > 0.5).float()


def dynamic_pos_weight(target: torch.Tensor, max_ratio: float = 12.0):
    pos = target.sum()
    neg = target.numel() - pos
    ratio = neg / (pos + 1.0)
    ratio = ratio.clamp(min=1.0, max=max_ratio)
    return ratio.detach()


def weighted_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    pixel_weight: Optional[torch.Tensor] = None,
):
    loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=pos_weight,
    )
    if pixel_weight is not None:
        loss = loss * pixel_weight
    return loss.mean()


class SegmentationCriterion(nn.Module):
    def __init__(
        self,
        aux_region_weight: float = 0.18,
        aux_boundary_weight: float = 0.06,
        aux_confidence_weight: float = 0.06,
        boundary_emphasis: float = 2.0,
    ):
        super().__init__()
        self.aux_region_weight = aux_region_weight
        self.aux_boundary_weight = aux_boundary_weight
        self.aux_confidence_weight = aux_confidence_weight
        self.boundary_emphasis = boundary_emphasis

    def forward(self, outputs: Dict[str, torch.Tensor], masks: torch.Tensor):
        masks = masks.float()
        logits = outputs["logits"]

        boundary_target = mask_to_boundary(masks)
        core_target = mask_to_core(masks)

        main_pos_weight = dynamic_pos_weight(masks, max_ratio=8.0)
        main_pixel_weight = 1.0 + self.boundary_emphasis * boundary_target
        main_loss = weighted_bce_with_logits(
            logits,
            masks,
            pos_weight=main_pos_weight,
            pixel_weight=main_pixel_weight,
        ) + dice_loss_from_logits(logits, masks)

        aux_region_loss = 0.0
        if "aux_region_logits" in outputs and len(outputs["aux_region_logits"]) > 0:
            pos_weight = dynamic_pos_weight(masks, max_ratio=8.0)
            losses = []
            for aux in outputs["aux_region_logits"]:
                losses.append(
                    weighted_bce_with_logits(aux, masks, pos_weight=pos_weight)
                    + dice_loss_from_logits(aux, masks)
                )
            aux_region_loss = torch.stack(losses).mean()

        aux_boundary_loss = 0.0
        if "aux_boundary_logits" in outputs and len(outputs["aux_boundary_logits"]) > 0:
            pos_weight = dynamic_pos_weight(boundary_target, max_ratio=20.0)
            losses = []
            for aux in outputs["aux_boundary_logits"]:
                losses.append(
                    weighted_bce_with_logits(aux, boundary_target, pos_weight=pos_weight)
                    + 0.5 * dice_loss_from_logits(aux, boundary_target)
                )
            aux_boundary_loss = torch.stack(losses).mean()

        aux_confidence_loss = 0.0
        if "aux_confidence_logits" in outputs and len(outputs["aux_confidence_logits"]) > 0:
            pos_weight = dynamic_pos_weight(core_target, max_ratio=12.0)
            losses = []
            for aux in outputs["aux_confidence_logits"]:
                losses.append(
                    weighted_bce_with_logits(aux, core_target, pos_weight=pos_weight)
                    + dice_loss_from_logits(aux, core_target)
                )
            aux_confidence_loss = torch.stack(losses).mean()

        total = main_loss
        if isinstance(aux_region_loss, torch.Tensor):
            total = total + self.aux_region_weight * aux_region_loss
        if isinstance(aux_boundary_loss, torch.Tensor):
            total = total + self.aux_boundary_weight * aux_boundary_loss
        if isinstance(aux_confidence_loss, torch.Tensor):
            total = total + self.aux_confidence_weight * aux_confidence_loss

        return total, {
            "main": float(main_loss.detach().item()),
            "aux_region": float(aux_region_loss.detach().item()) if isinstance(aux_region_loss, torch.Tensor) else 0.0,
            "aux_boundary": float(aux_boundary_loss.detach().item()) if isinstance(aux_boundary_loss, torch.Tensor) else 0.0,
            "aux_confidence": float(aux_confidence_loss.detach().item()) if isinstance(aux_confidence_loss, torch.Tensor) else 0.0,
        }
