# Deploying STAGE-PM to Google Cloud Run

The service is a FastAPI app (`src/aqf/serve/app.py`) that loads one trained checkpoint and a small replay
dataset at startup, serves a JSON API and a single-page dashboard, and runs on CPU.

| Endpoint | Returns |
|---|---|
| `GET /` | dashboard (time slider, per-station forecasts, trend chart, source mix, regime) |
| `GET /api/forecast?t=<ISO time>` | +1/+6/+24 h forecast per station: mean, 95% interval, exceedance probability, AQI band, source contribution, regime, stability index; the actual reading when one exists |
| `GET /api/series/{station_id}?hours=72&end=<ISO time>` | observed PM2.5 history for the trend chart |
| `GET /api/meta` | model name, replay range, stations |
| `GET /health` | liveness check |

## 1. Package the model and replay data

After the A->H suite has run and you have picked the checkpoint to ship:

```bash
python scripts/export_replay.py --checkpoint runs/H_full_model/best.pt \
    --raw data/real_2022_2026/raw.npz --start 2025-10-01 --end 2025-12-15
```

This writes `deploy/model/best.pt` and `deploy/replay.npz` (the replay hours plus 48 h of lookback). Both are small
enough to commit, and the Docker build copies them into the image.

## 2. Test locally (no Docker needed)

```bash
STAGEPM_CHECKPOINT=deploy/model/best.pt STAGEPM_REPLAY=deploy/replay.npz \
    PYTHONPATH=src uvicorn aqf.serve.app:app --port 8080
# open http://localhost:8080
```

## 3. Deploy to Cloud Run

One-time setup: install the Google Cloud CLI, and create a GCP project with billing enabled (Cloud Build needs it;
this workload stays within the free tier).

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

gcloud run deploy stage-pm --source . --region asia-south1 \
    --allow-unauthenticated --memory 2Gi --cpu 1 --max-instances 2
```

`--source .` uploads the repo (minus `.gcloudignore`), builds the `Dockerfile` in Cloud Build and deploys it.
The command prints the public `https://stage-pm-...run.app` URL.

## Notes and limits

* It is a **replay** demo: ERA5 weather lags real time by about 5 days, so a live forecast would need another
  weather feed. Replay over held-out hours also lets the page show forecast vs actual.
* Source-contribution and regime outputs have no ground truth on real data; the page says so and only names a
  regime when the model is clearly confident.
* First request after idle takes a few seconds (cold start loads torch and the checkpoint).
