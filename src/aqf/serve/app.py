"""FastAPI service: JSON forecast API + a single-page dashboard + optional Google Cloud integrations.

Run locally:
    STAGEPM_CHECKPOINT=deploy/model/best.pt STAGEPM_REPLAY=deploy/replay.npz \
        uvicorn aqf.serve.app:app --port 8080

Cloud Run sets $PORT and expects the server on 0.0.0.0 -- see the Dockerfile. The cloud features (Cloud Storage model
loading, Firestore history and feedback, API-key-guarded hourly job) are described in aqf/serve/cloud.py and switched on
by environment variables; without them the service behaves as a plain local forecast API.

Endpoints
    GET  /                       dashboard
    GET  /health                 liveness check (Cloud Run / uptime check)
    GET  /api/meta               model, replay range, stations
    GET  /api/forecast?t=        forecast for a replay hour
    GET  /api/series/{station}   observed PM2.5 history
    GET  /api/cloud              which cloud features are active (service, revision, model source, store)
    GET  /api/history?limit=     stored forecasts (Firestore, or in memory locally)
    POST /api/feedback           "useful / not useful" rating, rate-limited
    POST /jobs/run-forecast      advance the replay clock one hour, issue and store a forecast (needs X-API-Key)
"""
from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from aqf.serve import cloud
from aqf.serve.predictor import Predictor

log = logging.getLogger("stagepm")
logging.basicConfig(level=logging.INFO)  # Cloud Run sends stdout/stderr to Cloud Logging

STATIC = Path(__file__).parent / "static"
LOCAL_DIR = os.path.join(tempfile.gettempdir(), "stagepm")
_state: dict = {}


def _resolve(uri_env: str, bundled: str, local_name: str, label: str, source: dict) -> str:
    """Path to use for a file: downloaded from Cloud Storage when its URI is set and reachable, else the bundled copy."""
    uri = os.environ.get(uri_env, "").strip()
    if not uri:
        return bundled
    try:
        path = cloud.fetch_gcs(uri, os.path.join(LOCAL_DIR, local_name))
        source[label] = uri
        log.info("loaded %s from %s", label, uri)
        return path
    except Exception as e:  # missing package, no permission, wrong URI ...
        log.warning("could not load %s from %s (%r); using the bundled file", label, uri, e)
        return bundled


@asynccontextmanager
async def lifespan(app: FastAPI):
    source = {"model": "bundled", "replay": "bundled"}
    ckpt = _resolve("STAGEPM_MODEL_URI", os.environ.get("STAGEPM_CHECKPOINT", "deploy/model/best.pt"), "best.pt", "model", source)
    replay = _resolve("STAGEPM_REPLAY_URI", os.environ.get("STAGEPM_REPLAY", "deploy/replay.npz"), "replay.npz", "replay", source)
    _state["predictor"] = Predictor(ckpt, replay)  # loaded once at startup, not per request
    _state["source"] = source
    _state["store"] = cloud.make_store()
    _state["api_key"] = os.environ.get("STAGEPM_API_KEY", "").strip()
    _state["bucket"] = os.environ.get("STAGEPM_BUCKET", "").strip()
    _state["feedback_limiter"] = cloud.RateLimiter(max_events=20, window_s=60.0)
    log.info("ready: model=%s replay=%s store=%s", source["model"], source["replay"], _state["store"].kind)
    yield
    _state.clear()


app = FastAPI(title="STAGE-PM Delhi PM2.5 forecast", version="1.1", lifespan=lifespan)


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


# ----------------------------------------------------------------------------- cloud features
@app.get("/api/cloud")
def cloud_status():
    """Which cloud features are active. K_SERVICE / K_REVISION are set by Cloud Run itself."""
    return {
        "platform": "Google Cloud Run" if os.environ.get("K_SERVICE") else "local",
        "service": os.environ.get("K_SERVICE"),
        "revision": os.environ.get("K_REVISION"),
        "model_source": _state["source"]["model"],
        "replay_source": _state["source"]["replay"],
        "store": _state["store"].kind,
        "snapshot_bucket": _state["bucket"] or None,
        "job_endpoint_enabled": bool(_state["api_key"]),
    }


@app.get("/api/history")
def history(limit: int = Query(24, ge=1, le=100)):
    store = _state["store"]
    return {"store": store.kind, "items": [cloud.headline(d) for d in store.recent_forecasts(limit)]}


class Feedback(BaseModel):
    issued_at: str = Field(..., max_length=40)
    rating: Literal["useful", "not_useful"]
    station_id: str | None = Field(None, max_length=10)
    comment: str = Field("", max_length=200)


@app.post("/api/feedback")
def feedback(fb: Feedback, request: Request):
    ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (request.client.host if request.client else "unknown")
    if not _state["feedback_limiter"].allow(ip):
        raise HTTPException(status_code=429, detail="too many feedback messages, try again in a minute")
    store = _state["store"]
    store.add_feedback({**fb.model_dump(), "created_at": cloud._now_iso()})
    log.info("feedback rating=%s issued_at=%s", fb.rating, fb.issued_at)
    return {"stored": True, "store": store.kind}


@app.post("/jobs/run-forecast")
def run_forecast(x_api_key: str | None = Header(default=None)):
    """Hourly job (Cloud Scheduler): move the replay clock one hour on, forecast from there, store the result."""
    expected = _state["api_key"]
    if not expected:
        raise HTTPException(status_code=503, detail="job endpoint disabled: STAGEPM_API_KEY is not set")
    if not cloud.key_ok(x_api_key, expected):
        raise HTTPException(status_code=401, detail="missing or invalid API key")
    p, store = _p(), _state["store"]
    cur = store.get_cursor()
    nxt = 0 if cur is None else (int(cur) + 1) % p.n_steps
    doc = cloud.summarise(p.forecast_at(p.first_idx + nxt))
    store.add_forecast(doc)
    store.set_cursor(nxt)
    snapshot = None
    if _state["bucket"]:
        try:
            snapshot = cloud.upload_json(_state["bucket"], "forecasts/latest.json", doc)
        except Exception as e:
            log.warning("could not write the snapshot to Cloud Storage (%r)", e)
    log.info("job run-forecast issued_at=%s cursor=%d store=%s snapshot=%s", doc["issued_at"], nxt, store.kind, snapshot)
    return {"stored": True, "store": store.kind, "issued_at": doc["issued_at"], "cursor": nxt, "snapshot": snapshot,
            "mean_now": doc["mean_now"], "max_p_exceed_24h": doc["max_p_exceed_24h"]}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
