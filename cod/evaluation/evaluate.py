from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from cod.data.dataset import CODDataset, eval_collate
from cod.evaluation.metrics import CODMetrics
from cod.evaluation.degrade import degrade_depth


@torch.inference_mode()
def evaluate(model, dataset: CODDataset, device, *, use_depth: bool, threshold: float = 0.5,
             batch_size: int = 8, num_workers: int = 0, degradation: str | None = None,
             severity: float = 1.0, seed: int = 0) -> dict:
    """Evaluate on `dataset`. Predictions are resized to each GT's original resolution.

    use_depth=False runs the pure RGB path even for fusion models (depth withheld).
    `degradation` perturbs the cached estimated depth (see cod.evaluation.degrade).
    """
    model.eval()
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                     collate_fn=eval_collate)
    meter = CODMetrics(threshold)
    gen = torch.Generator().manual_seed(seed)
    for b in dl:
        img = b["image"].to(device)
        depth = valid = None
        if use_depth:
            depth, valid = b["depth"], b["depth_valid"]
            if degradation:
                depth = degrade_depth(depth, degradation, severity, gen)
            depth, valid = depth.to(device), valid.to(device)
        prob = torch.sigmoid(model(img, depth, valid)["logits"].float())
        for p, gt in zip(prob, b["gt"]):
            p = F.interpolate(p[None], size=gt.shape, mode="bilinear", align_corners=False)[0, 0]
            meter.update(p.cpu().numpy(), gt.numpy())
    return meter.compute()
