from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits.float()).flatten(1)
    t = target.float().flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def structure_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Boundary-weighted BCE + weighted IoU (PraNet/F3Net style structure loss)."""
    logits, target = logits.float(), target.float()
    w = 1 + 5 * (F.avg_pool2d(target, 31, 1, 15) - target).abs()
    wbce = (w * F.binary_cross_entropy_with_logits(logits, target, reduction="none")).sum((2, 3)) / w.sum((2, 3))
    p = torch.sigmoid(logits)
    inter, union = (p * target * w).sum((2, 3)), ((p + target) * w).sum((2, 3))
    return (wbce + (1 - (inter + 1) / (union - inter + 1))).mean()


def mask_to_boundary(mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Morphological gradient (dilate - erode) of a binary mask; computed on-device so it matches augmentation."""
    pad = k // 2
    dil = F.max_pool2d(mask, k, 1, pad)
    ero = -F.max_pool2d(-mask, k, 1, pad)
    return (dil - ero).clamp(0, 1)


class CODLoss(torch.nn.Module):
    """Weighted sum of BCE-with-logits, Dice, optional structure and boundary losses; weights come from config."""

    def __init__(self, bce: float = 1.0, dice: float = 1.0, structure: float = 0.0, boundary: float = 0.0):
        super().__init__()
        self.w = dict(bce=bce, dice=dice, structure=structure, boundary=boundary)

    def forward(self, out: dict, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        logits = out["logits"].float()
        parts = {}
        if self.w["bce"]:
            parts["bce"] = F.binary_cross_entropy_with_logits(logits, target)
        if self.w["dice"]:
            parts["dice"] = dice_loss(logits, target)
        if self.w["structure"]:
            parts["structure"] = structure_loss(logits, target)
        if self.w["boundary"]:
            if "boundary" not in out:
                raise ValueError("loss.boundary > 0 requires model.boundary_head: true")
            bt = mask_to_boundary(target)
            parts["boundary"] = F.binary_cross_entropy_with_logits(out["boundary"].float(), bt) + dice_loss(out["boundary"], bt)
        total = sum(self.w[k] * v for k, v in parts.items())
        return total, {k: float(v.detach()) for k, v in parts.items()}
