"""FastAPI backend. Models load once per worker (lifespan); inference runs in a worker thread, serialised by a lock."""
from __future__ import annotations

import base64
import io
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from cod.api.settings import Settings
from cod.depth.estimator import DepthEstimator
from cod.inference.predictor import Predictor, make_overlay, png_bytes, to_mask
from cod.utils import get_device

log = logging.getLogger("cod.api")
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
MODES = ("rgb", "multimodal", "both")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


def decode_image(data: bytes, s: Settings) -> Image.Image:
    """Validate by content (not extension / client MIME): size, decoder format, dimensions; strip EXIF orientation."""
    if len(data) == 0:
        raise ApiError(400, "empty_file", "The uploaded file is empty.")
    if len(data) > s.max_upload_mb * 2**20:
        raise ApiError(413, "file_too_large", f"File exceeds the {s.max_upload_mb:g} MB limit.")
    Image.MAX_IMAGE_PIXELS = s.max_side * s.max_side
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt, size = probe.format, probe.size
            probe.verify()
        if fmt not in ALLOWED_FORMATS:
            raise ApiError(415, "unsupported_type", f"Unsupported image type; allowed: {', '.join(sorted(ALLOWED_FORMATS))}.")
        if max(size) > s.max_side or min(size) < s.min_side:
            raise ApiError(422, "bad_dimensions", f"Image sides must be between {s.min_side} and {s.max_side} px.")
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except ApiError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, ValueError):
        raise ApiError(415, "invalid_image", "The file could not be decoded as a supported image.") from None


class Engine:
    """Loads the predictors named in settings; at least one checkpoint is required (no placeholder predictions)."""

    def __init__(self, s: Settings):
        self.s = s
        self.lock = threading.Lock()
        device = get_device(s.device)
        self.device = device
        self.rgb: Predictor | None = None
        self.mm: Predictor | None = None
        if not (s.rgb_checkpoint or s.mm_checkpoint):
            return
        self.depth = None
        if s.mm_checkpoint:
            self.depth = DepthEstimator(s.depth_model, device)
            self.mm = Predictor(s.mm_checkpoint, device, self.depth)
            if not self.mm.uses_depth:
                raise RuntimeError("COD_MM_CHECKPOINT must be a depth-fusion checkpoint")
        if s.rgb_checkpoint:
            self.rgb = Predictor(s.rgb_checkpoint, device)
            if self.rgb.uses_depth:
                raise RuntimeError("COD_RGB_CHECKPOINT must be an RGB-only checkpoint")

    @property
    def ready(self) -> bool:
        return self.rgb is not None or self.mm is not None

    def available_modes(self) -> list[str]:
        m = []
        if self.rgb or self.mm:
            m.append("rgb")  # a fusion checkpoint can run with depth withheld
        if self.mm:
            m.append("multimodal")
        if self.rgb and self.mm:
            m.append("both")
        return m

    def run(self, img: Image.Image, mode: str, threshold: float, opacity: float) -> dict:
        with self.lock:
            runs = {}
            if mode in ("rgb", "both"):
                p = self.rgb or self.mm
                runs["rgb"] = p.predict(img, use_depth=False)
            if mode in ("multimodal", "both"):
                runs["multimodal"] = self.mm.predict(img, use_depth=True)
        out = {}
        for k, r in runs.items():
            out[k] = {
                "mask_png": _b64(png_bytes(to_mask(r["prob"], threshold))),
                "overlay_png": _b64(png_bytes(make_overlay(img, r["prob"], threshold, opacity))),
                "probability_png": _b64(png_bytes((r["prob"] * 255).astype("uint8"))),
                "foreground_fraction": float((r["prob"] >= threshold).mean()),
                "used_depth": r["used_depth"],
                "timings_ms": {a: round(b, 1) for a, b in r["timings_ms"].items()},
            }
        return out


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = None
        try:
            app.state.engine = await run_in_threadpool(Engine, s)
            log.info("engine ready=%s device=%s modes=%s", app.state.engine.ready, app.state.engine.device,
                     app.state.engine.available_modes())
        except Exception:  # keep the process up so /health explains the problem
            log.exception("failed to load models")
            app.state.engine = None
        yield

    app = FastAPI(title="Camouflaged Object Detection API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        rid, t0 = uuid.uuid4().hex[:12], time.perf_counter()
        request.state.rid = rid
        resp = await call_next(request)
        resp.headers["X-Request-ID"] = rid
        log.info("%s %s -> %d %.0fms rid=%s", request.method, request.url.path, resp.status_code,
                 (time.perf_counter() - t0) * 1000, rid)
        return resp

    def err(status: int, code: str, message: str, request: Request | None = None):
        return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})

    @app.exception_handler(ApiError)
    async def _api_err(request, e: ApiError):
        return err(e.status, e.code, e.message)

    @app.exception_handler(RequestValidationError)
    async def _val_err(request, e):
        return err(422, "invalid_request", "Invalid request parameters.")

    @app.exception_handler(Exception)
    async def _unhandled(request, e):
        log.exception("unhandled error rid=%s", getattr(request.state, "rid", "-"))
        return err(500, "internal_error", "Internal server error.")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/v1/status")
    async def status():
        e: Engine | None = app.state.engine
        return {"ready": bool(e and e.ready), "device": str(e.device) if e else None,
                "modes": e.available_modes() if e else [], "max_upload_mb": s.max_upload_mb}

    @app.post("/api/v1/predict")
    async def predict(file: UploadFile = File(...), mode: str = Form("rgb"), threshold: float = Form(0.5),
                      opacity: float = Form(0.5)):
        e: Engine | None = app.state.engine
        if e is None or not e.ready:
            raise ApiError(503, "model_unavailable", "No model is loaded on the server.")
        if mode not in MODES or mode not in e.available_modes():
            raise ApiError(400, "bad_mode", f"mode must be one of {e.available_modes()}.")
        if not (0.0 < threshold < 1.0) or not (0.0 <= opacity <= 1.0):
            raise ApiError(422, "invalid_request", "threshold must be in (0,1) and opacity in [0,1].")
        data = await file.read(int(s.max_upload_mb * 2**20) + 1)
        img = decode_image(data, s)
        t0 = time.perf_counter()
        results = await run_in_threadpool(e.run, img, mode, threshold, opacity)
        return {"width": img.width, "height": img.height, "threshold": threshold, "results": results,
                "total_ms": round((time.perf_counter() - t0) * 1000, 1)}

    if s.frontend_dir and Path(s.frontend_dir).is_dir():
        app.mount("/", StaticFiles(directory=s.frontend_dir, html=True), name="frontend")
    return app


app = create_app
