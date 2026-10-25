from __future__ import annotations

import torch
import torch.nn as nn


class DinoV2Backbone(nn.Module):
    """DINOv2 (Hugging Face weights) returning patch-token maps from several blocks.

    Output: list of (B, C, H/14, W/14) tensors, one per requested (1-based) block.
    """

    patch = 14

    def __init__(self, name: str = "facebook/dinov2-small", layers: list[int] = (3, 6, 9, 12), pretrained: bool = True):
        super().__init__()
        from transformers import AutoConfig, Dinov2Model

        if pretrained:
            self.model = Dinov2Model.from_pretrained(name)
        else:
            self.model = Dinov2Model(AutoConfig.from_pretrained(name))
        self.layers = list(layers)
        self.hidden = self.model.config.hidden_size
        self.n_blocks = self.model.config.num_hidden_layers
        if max(self.layers) > self.n_blocks:
            raise ValueError(f"layers {self.layers} exceed {self.n_blocks} blocks of {name}")

    def set_trainable(self, trainable: bool, last_n_blocks: int | None = None) -> None:
        """Freeze/unfreeze; `last_n_blocks` limits fine-tuning to the final N blocks (+ final norm)."""
        for p in self.model.parameters():
            p.requires_grad = False
        if not trainable:
            return
        if last_n_blocks is None:
            for p in self.model.parameters():
                p.requires_grad = True
            return
        for blk in self.model.encoder.layer[-last_n_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.model.layernorm.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        B, _, H, W = x.shape
        if H % self.patch or W % self.patch:
            raise ValueError(f"input size {(H, W)} must be a multiple of {self.patch}")
        h, w = H // self.patch, W // self.patch
        hs = self.model(pixel_values=x, output_hidden_states=True).hidden_states  # (emb, blk1, ..., blkN)
        return [hs[i][:, 1:, :].transpose(1, 2).reshape(B, self.hidden, h, w) for i in self.layers]
