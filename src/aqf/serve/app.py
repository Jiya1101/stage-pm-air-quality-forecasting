"""FastAPI service: JSON forecast API + a single-page dashboard.

Run locally:
    STAGEPM_CHECKPOINT=deploy/model/best.pt STAGEPM_REPLAY=deploy/replay.npz \
        uvicorn aqf.serve.app:app --port 8080

Cloud Run sets $PORT and expects the server on 0.0.0.0 -- see the Dockerfile.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from aqf.serve.predictor import Predictor

STATIC = Path(__file__).parent / "static"
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = os.environ.get("STAGEPM_CHECKPOINT", "deploy/model/best.pt")
    replay = os.environ.get("STAGEPM_REPLAY", "deploy/replay.npz")
    _state["predictor"] = Predictor(ckpt, replay)  # loaded once at startup, not per request
    yield
    _state.clear()


app = FastAPI(title="STAGE-PM Delhi PM2.5 forecast", version="1.0", lifespan=lifespan)


def _p() -> Predictor:
    return _state["predictor"]


@app.get("/health")
def health():
    return {"status": "ok", "model": _p().run_name}


@app.get("/api/meta")
def meta():
    return _p().meta()


@app.get("/api/forecast")
def forecast(t: str | None = Query(None, description="ISO timestamp inside the replay range; default = latest")):
    try:
        return _p().forecast(t)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"bad timestamp: {e}")


@app.get("/api/series/{station_id}")
def series(station_id: str, hours: int = Query(72, ge=6, le=720), end: str | None = None):
    try:
        return _p().series(station_id, hours, end)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"unknown station {station_id}")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
