import pytest
import torch

from cod.models import CODModel
from cod.models.fusion import FUSION_TYPES
from cod.training.losses import CODLoss, dice_loss, mask_to_boundary

FUSIONS = [f for f in FUSION_TYPES if f != "none"]


@pytest.mark.parametrize("fusion", FUSION_TYPES)
def test_forward_backward_all_fusions(cfg_factory, fusion):
    torch.manual_seed(0)
    cfg = cfg_factory(fusion, boundary_head=True)
    m = CODModel(cfg, pretrained=False)
    x, d = torch.randn(2, 3, 112, 112), torch.rand(2, 1, 112, 112)
    out = m(x, d if fusion != "none" else None, torch.ones(2))
    assert out["logits"].shape == (2, 1, 112, 112) and out["boundary"].shape == (2, 1, 112, 112)
    if fusion == "reliability":
        assert out["reliability"].shape == (2, 1, 8, 8) and 0 <= out["reliability"].min() and out["reliability"].max() <= 1
    loss, parts = CODLoss(boundary=0.5, structure=0.5)(out, (torch.rand(2, 1, 112, 112) > 0.5).float())
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for n, p in m.named_parameters() if n.startswith(("decoder", "proj"))]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_cross_attn_starts_as_rgb_identity(cfg_factory):
    torch.manual_seed(0)
    m = CODModel(cfg_factory("cross_attn"), pretrained=False).eval()
    x, d = torch.randn(1, 3, 112, 112), torch.rand(1, 1, 112, 112)
    with torch.no_grad():
        assert torch.allclose(m(x, d)["logits"], m(x)["logits"], atol=1e-6)


@pytest.mark.parametrize("fusion", FUSIONS)
def test_rgb_path_works_without_depth(cfg_factory, fusion):
    m = CODModel(cfg_factory(fusion), pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.randn(1, 3, 112, 112))["logits"].shape == (1, 1, 112, 112)


def test_invalid_depth_zeroes_reliability(cfg_factory):
    m = CODModel(cfg_factory("reliability"), pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 112, 112), torch.zeros(1, 1, 112, 112), torch.zeros(1))
    assert out["reliability"].max() == 0


def test_backbone_freezing(cfg_factory):
    m = CODModel(cfg_factory("none"), pretrained=False)
    m.backbone.set_trainable(False)
    assert not any(p.requires_grad for p in m.backbone.parameters())
    m.backbone.set_trainable(True, last_n_blocks=2)
    trainable = {n for n, p in m.backbone.named_parameters() if p.requires_grad}
    assert trainable and all("layer.10" in n or "layer.11" in n or "layernorm" in n for n in trainable)
    m.backbone.set_trainable(True)
    assert all(p.requires_grad for p in m.backbone.parameters())


def test_rejects_non_multiple_of_patch(cfg_factory):
    m = CODModel(cfg_factory("none"), pretrained=False)
    with pytest.raises(ValueError):
        m(torch.randn(1, 3, 100, 100))


def test_losses_stable_and_correct():
    t = torch.zeros(1, 1, 8, 8)
    t[..., 2:6, 2:6] = 1
    assert dice_loss(torch.where(t > 0, 30.0, -30.0), t) < 1e-3
    assert dice_loss(torch.where(t > 0, -30.0, 30.0), t) > 0.9
    extreme = {"logits": torch.full((1, 1, 8, 8), 1e4)}
    loss, _ = CODLoss()(extreme, t)
    assert torch.isfinite(loss)
    big = torch.zeros(1, 1, 32, 32)
    big[..., 8:24, 8:24] = 1
    b = mask_to_boundary(big)
    assert b[..., 14:18, 14:18].sum() == 0 and b[..., 0:4, 0:4].sum() == 0 and b[..., 7:10, 12].sum() > 0
