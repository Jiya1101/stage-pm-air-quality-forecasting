# STAGE-PM inference service for Google Cloud Run.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only torch keeps the image ~1 GB smaller than the default CUDA wheel.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src ./src
COPY deploy ./deploy
ENV PYTHONPATH=/app/src \
    STAGEPM_CHECKPOINT=/app/deploy/model/best.pt \
    STAGEPM_REPLAY=/app/deploy/replay.npz

# Cloud Run injects $PORT (default 8080) and requires listening on 0.0.0.0.
CMD ["sh", "-c", "uvicorn aqf.serve.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
