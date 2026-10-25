# Backend + built frontend in one image. Default build is CPU; for NVIDIA GPUs swap the base image for a CUDA
# runtime and install the matching torch wheel (see README "Docker / CUDA").
FROM node:22-slim AS ui
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models/hf
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY cod ./cod
COPY configs ./configs
COPY --from=ui /ui/dist ./frontend_dist
RUN useradd -m app && mkdir -p /models && chown -R app /models /app
USER app
ENV COD_FRONTEND_DIR=/app/frontend_dist COD_DEVICE=auto
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["uvicorn", "cod.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
