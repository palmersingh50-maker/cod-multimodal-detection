from __future__ import annotations

import torch
import torch.nn as nn

from cod.models.backbone import DinoV2Backbone
from cod.models.decoder import MultiScaleDecoder
from cod.models.depth_encoder import DEPTH_ENCODERS, to_grid
from cod.models.fusion import FUSION_TYPES, build_fuser


class CODModel(nn.Module):
    """DINOv2 RGB branch -> (optional) RGB/depth token fusion at selected levels -> multiscale decoder.

    forward(image, depth=None, depth_valid=None):
      * depth is None  -> pure RGB path (identical computation to the RGB-only baseline architecture).
      * depth given    -> estimated depth is encoded and fused (if cfg fusion != "none").
    Returns dict(logits, [boundary], [reliability]) with logits at input resolution.
    """

    def __init__(self, cfg: dict, pretrained: bool = True):
        super().__init__()
        m = cfg["model"]
        self.fusion_type = m.get("fusion", "none")
        if self.fusion_type not in FUSION_TYPES:
            raise ValueError(f"fusion must be one of {FUSION_TYPES}")
        D = m["embed_dim"]
        self.backbone = DinoV2Backbone(m["backbone"], m["layers"], pretrained)
        self.proj = nn.ModuleList([nn.Sequential(nn.GroupNorm(1, self.backbone.hidden), nn.Conv2d(self.backbone.hidden, D, 1))
                                   for _ in m["layers"]])
        self.levels = list(m.get("fusion_levels", range(len(m["layers"])))) if self.fusion_type != "none" else []
        if self.levels:
            self.depth_encoder = DEPTH_ENCODERS[m.get("depth_encoder", "cnn")](m.get("depth_channels", 48))
            C = self.depth_encoder.out_channels
            self.depth_proj = nn.ModuleDict({str(l): nn.Conv2d(C, D, 1) for l in self.levels})
            self.fusers = nn.ModuleDict({str(l): build_fuser(self.fusion_type, D, m.get("num_heads", 4)) for l in self.levels})
        self.decoder = MultiScaleDecoder(D, m.get("boundary_head", False))

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def forward(self, image, depth=None, depth_valid=None) -> dict:
        feats = [pr(f) for pr, f in zip(self.proj, self.backbone(image))]
        out_rel = None
        if depth is not None and self.levels:
            B, _, h, w = feats[0].shape
            if depth_valid is None:
                depth_valid = torch.ones(B, device=image.device)
            dfeat = self.depth_encoder(depth, depth_valid, (h, w))
            cues = self.depth_encoder.cues(depth, depth_valid)[:, 1:]  # [grad, valid]
            cues = to_grid(cues, (h, w)).flatten(2).transpose(1, 2)
            rels = []
            for l in self.levels:
                rgb_t = feats[l].flatten(2).transpose(1, 2)
                d_t = self.depth_proj[str(l)](dfeat).flatten(2).transpose(1, 2)
                fuser = self.fusers[str(l)]
                fused = fuser(rgb_t, d_t, cues)
                feats[l] = fused.transpose(1, 2).reshape(B, -1, h, w)
                if getattr(fuser, "last_reliability", None) is not None:
                    rels.append(fuser.last_reliability.transpose(1, 2).reshape(B, 1, h, w))
            if rels:
                out_rel = torch.stack(rels).mean(0)
        out = self.decoder(feats, image)
        if out_rel is not None:
            out["reliability"] = out_rel
        return out
