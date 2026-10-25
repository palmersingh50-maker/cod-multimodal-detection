from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


@dataclass
class Settings:
    """All configuration comes from environment variables (see .env.example); nothing is hardcoded to a path."""
    rgb_checkpoint: str | None = field(default_factory=lambda: _env("COD_RGB_CHECKPOINT"))
    mm_checkpoint: str | None = field(default_factory=lambda: _env("COD_MM_CHECKPOINT"))
    depth_model: str = field(default_factory=lambda: _env("COD_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf"))
    device: str = field(default_factory=lambda: _env("COD_DEVICE", "auto"))
    max_upload_mb: float = field(default_factory=lambda: float(_env("COD_MAX_UPLOAD_MB", "10")))
    max_side: int = field(default_factory=lambda: int(_env("COD_MAX_SIDE", "4096")))
    min_side: int = field(default_factory=lambda: int(_env("COD_MIN_SIDE", "32")))
    cors_origins: list[str] = field(default_factory=lambda: (_env("COD_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173") or "").split(","))
    frontend_dir: str | None = field(default_factory=lambda: _env("COD_FRONTEND_DIR"))
