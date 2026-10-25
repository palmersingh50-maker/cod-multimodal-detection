import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cod.config import load_config


def tiny_cfg(fusion="none", **model):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")
    cfg["data"]["image_size"] = 112
    cfg["model"].update(fusion=fusion, embed_dim=32, depth_channels=16, **model)
    return cfg


@pytest.fixture
def cfg_factory():
    return tiny_cfg


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Folder dataset: noise images with a bright rectangle and the matching mask."""
    (tmp_path / "img").mkdir(), (tmp_path / "gt").mkdir(), (tmp_path / "depth" / "syn").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(6):
        h, w = 90 + 7 * i, 130 + 5 * i
        img = rng.integers(0, 80, (h, w, 3), dtype=np.uint8)
        m = np.zeros((h, w), np.uint8)
        m[20:50, 30:80] = 255
        img[m > 0] += 120
        Image.fromarray(img).save(tmp_path / "img" / f"s{i}.jpg")
        Image.fromarray(m).save(tmp_path / "gt" / f"s{i}.png")
        d = (m > 0).astype(np.float32) * 0.8 + 0.1
        Image.fromarray((d * 65535).astype(np.uint16)).save(tmp_path / "depth" / "syn" / f"s{i}.png")
    return tmp_path


@pytest.fixture
def tiny_checkpoint(tmp_path):
    from cod.models import CODModel

    def make(fusion):
        cfg = tiny_cfg(fusion)
        m = CODModel(cfg, pretrained=False)
        path = tmp_path / f"tiny_{fusion}.pt"
        torch.save({"model": m.state_dict(), "config": cfg}, path)
        return path

    return make
