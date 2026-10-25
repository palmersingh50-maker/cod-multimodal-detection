#!/usr/bin/env bash
# One command: builds the UI if needed and serves UI + API at http://127.0.0.1:8000
cd "$(dirname "$0")"
[ -d frontend/dist ] || (cd frontend && npm install && npm run build)
export COD_RGB_CHECKPOINT="${COD_RGB_CHECKPOINT:-checkpoints/rgb.pt}"
[ -f checkpoints/multimodal.pt ] && export COD_MM_CHECKPOINT="${COD_MM_CHECKPOINT:-checkpoints/multimodal.pt}"
export COD_FRONTEND_DIR=frontend/dist
exec .venv/bin/uvicorn cod.api.app:create_app --factory --host 127.0.0.1 --port 8000
