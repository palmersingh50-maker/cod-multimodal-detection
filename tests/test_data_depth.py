import numpy as np
import pytest
import torch

from cod.data.dataset import CODDataset, eval_collate, pair_folder, split_train_val
from cod.depth.estimator import load_depth_png, normalize_depth, save_depth_png
from cod.evaluation.degrade import MODES, degrade_depth


def test_pairing_and_mask_alignment(synthetic_dataset):
    s = pair_folder(synthetic_dataset / "img", synthetic_dataset / "gt")
    ds = CODDataset(s, "syn", 112, train=False, use_depth=True, depth_cache=synthetic_dataset / "depth")
    b = eval_collate([ds[i] for i in range(3)])
    assert b["image"].shape == (3, 3, 112, 112) and b["mask"].shape == (3, 1, 112, 112) and b["depth"].shape == (3, 1, 112, 112)
    # mask and depth were derived from the same rectangle -> must stay co-registered after resizing
    m, d = b["mask"][0, 0] > 0.5, b["depth"][0, 0]
    assert d[m].mean() > d[~m].mean() + 0.5
    assert b["gt"][0].shape != (112, 112)  # GT stays at original resolution


def test_train_augmentation_keeps_depth_mask_aligned(synthetic_dataset):
    s = pair_folder(synthetic_dataset / "img", synthetic_dataset / "gt")
    ds = CODDataset(s, "syn", 112, train=True, use_depth=True, depth_cache=synthetic_dataset / "depth")
    for i in range(6):
        it = ds[i]
        m, d = it["mask"][0] > 0.5, it["depth"][0]
        if m.any() and (~m).any():
            assert d[m].mean() > d[~m].mean() + 0.3


def test_missing_depth_cache_is_an_error(synthetic_dataset, tmp_path):
    s = pair_folder(synthetic_dataset / "img", synthetic_dataset / "gt")
    with pytest.raises(FileNotFoundError):
        CODDataset(s, "syn", 112, train=False, use_depth=True, depth_cache=tmp_path / "nope")


def test_split_is_disjoint_and_seeded(synthetic_dataset):
    s = pair_folder(synthetic_dataset / "img", synthetic_dataset / "gt")
    a, b = split_train_val(s, 0.34, 1)
    assert not set(a) & set(b) and len(a) + len(b) == len(s)
    assert split_train_val(s, 0.34, 1) == (a, b)


def test_depth_normalisation_and_validity():
    d, ok = normalize_depth(np.random.default_rng(0).normal(size=(40, 40)) * 5)
    assert ok and d.min() >= 0 and d.max() <= 1
    z, ok = normalize_depth(np.full((8, 8), 3.0))
    assert not ok and z.max() == 0
    z, ok = normalize_depth(np.full((8, 8), np.nan))
    assert not ok
    d, ok = normalize_depth(np.array([[1, 2], [np.nan, 4]] * 4, dtype=float))
    assert ok and np.isfinite(d).all()


def test_depth_png_roundtrip(tmp_path):
    d = np.random.default_rng(0).random((20, 30)).astype(np.float32)
    save_depth_png(tmp_path / "d.png", d, True)
    r, ok = load_depth_png(tmp_path / "d.png")
    assert ok and np.abs(r - d).max() < 1e-4


@pytest.mark.parametrize("mode", MODES)
def test_degradations_keep_range_and_shape(mode):
    d = torch.rand(2, 1, 32, 32)
    o = degrade_depth(d, mode, 1.0, torch.Generator().manual_seed(0))
    assert o.shape == d.shape and o.min() >= 0 and o.max() <= 1
