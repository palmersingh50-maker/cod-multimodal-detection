import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from cod.api.app import create_app
from cod.api.settings import Settings
from cod.inference.predictor import Predictor, make_overlay, to_mask
from cod.models import CODModel


def jpeg_bytes(w=120, h=90, fmt="JPEG"):
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(buf, format=fmt)
    return buf.getvalue()


def test_checkpoint_roundtrip_and_single_image(tiny_checkpoint):
    p = Predictor(tiny_checkpoint("none"), "cpu")
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (75, 101, 3), dtype=np.uint8))
    r = p.predict(img)
    assert r["prob"].shape == (75, 101) and 0 <= r["prob"].min() and r["prob"].max() <= 1
    assert set(r["timings_ms"]) >= {"preprocess_ms", "model_ms", "postprocess_ms", "total_ms"} and not r["used_depth"]
    assert to_mask(r["prob"]).dtype == np.uint8 and make_overlay(img, r["prob"]).size == img.size
    with pytest.raises(ValueError):
        p.predict(img, use_depth=True)


def test_checkpoint_weights_identical_after_reload(tiny_checkpoint, cfg_factory):
    path = tiny_checkpoint("gated")
    ck = torch.load(path, weights_only=False)
    p = Predictor(path, "cpu")
    for k, v in p.model.state_dict().items():
        assert torch.equal(v, ck["model"][k])


@pytest.fixture
def client(tiny_checkpoint):
    app = create_app(Settings(rgb_checkpoint=str(tiny_checkpoint("none")), device="cpu", max_upload_mb=1))
    with TestClient(app) as c:
        yield c


def test_health_and_status(client):
    assert client.get("/health").json() == {"status": "ok"}
    s = client.get("/api/v1/status").json()
    assert s["ready"] and s["modes"] == ["rgb"]


def test_predict_ok(client):
    r = client.post("/api/v1/predict", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")}, data={"mode": "rgb"})
    assert r.status_code == 200
    j = r.json()
    assert (j["width"], j["height"]) == (120, 90)
    import base64
    mask = Image.open(io.BytesIO(base64.b64decode(j["results"]["rgb"]["mask_png"])))
    assert mask.size == (120, 90) and set(np.unique(np.array(mask))) <= {0, 255}
    assert "X-Request-ID" in r.headers or "x-request-id" in r.headers


@pytest.mark.parametrize("payload,code,status", [
    (b"", "empty_file", 400),
    (b"not an image at all", "invalid_image", 415),
    (b"\x89PNG\r\n\x1a\n" + b"junk" * 10, "invalid_image", 415),
])
def test_predict_rejects_bad_files(client, payload, code, status):
    r = client.post("/api/v1/predict", files={"file": ("x.png", payload, "image/png")})
    assert r.status_code == status and r.json()["error"]["code"] == code
    assert "Traceback" not in r.text and "/Users" not in r.text


def test_predict_rejects_oversize_and_bad_params(client):
    big = b"\xff\xd8" + b"0" * (2 * 2**20)
    assert client.post("/api/v1/predict", files={"file": ("b.jpg", big, "image/jpeg")}).status_code == 413
    ok = jpeg_bytes()
    assert client.post("/api/v1/predict", files={"file": ("a.jpg", ok, "image/jpeg")}, data={"mode": "multimodal"}).status_code == 400
    assert client.post("/api/v1/predict", files={"file": ("a.jpg", ok, "image/jpeg")}, data={"threshold": "2"}).status_code == 422
    assert client.post("/api/v1/predict", files={"file": ("a.jpg", jpeg_bytes(10, 10), "image/jpeg")}).status_code == 422
    assert client.post("/api/v1/predict").status_code == 422


def test_gif_not_allowed(client):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format="GIF")
    assert client.post("/api/v1/predict", files={"file": ("a.gif", buf.getvalue(), "image/gif")}).status_code == 415


def test_no_model_returns_503():
    with TestClient(create_app(Settings(rgb_checkpoint=None, mm_checkpoint=None, device="cpu"))) as c:
        assert c.get("/api/v1/status").json()["ready"] is False
        r = c.post("/api/v1/predict", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")})
        assert r.status_code == 503
