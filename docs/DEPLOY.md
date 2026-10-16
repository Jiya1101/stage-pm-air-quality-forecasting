# Deploying STAGE-PM to Google Cloud

The forecast service (`src/aqf/serve/app.py`) runs on Cloud Run and uses eight Google Cloud services, each with a real job.
Everything is designed to stay inside the free allowances; see "Cost control" before you start.

```
Cloud Scheduler (hourly) --X-API-Key--> Cloud Run service "stage-pm" --writes--> Firestore (forecast history, feedback)
                                              |  loads model + data at start-up        |
                                              +----------- Cloud Storage <-------------+--> snapshot forecasts/latest.json
GitHub push --> Cloud Build --> Artifact Registry --> Cloud Run          API key kept in Secret Manager
Cloud Monitoring uptime check + Cloud Logging watch the service          Dedicated service account, minimal roles (IAM)
```

| # | Service | Job in this project |
|---|---|---|
| 1 | Cloud Run | Forecast API and dashboard (scale to zero, 2 GiB, max 2 instances) |
| 2 | Cloud Build | Builds and deploys on every push to `main` (`cloudbuild.yaml`) |
| 3 | Artifact Registry | Stores the container images |
| 4 | Cloud Storage | Versioned model and data: Cloud Run loads `best.pt` and `replay.npz` from the bucket, so a new model needs no rebuild; also holds `forecasts/latest.json` |
| 5 | Firestore | Stored hourly forecasts (with the values later observed) and user feedback |
| 6 | Cloud Scheduler | Calls `POST /jobs/run-forecast` every hour; each call moves the replay clock one hour on, so the history grows like a live feed |
| 7 | Secret Manager | The API key that guards the job endpoint |
| 8 | Cloud Monitoring and Logging | Uptime check on `/health`, request logs, a log-based metric |

IAM ties it together: a dedicated service account `stage-pm-run` with only the roles it needs.

Every cloud feature is optional in the code. If a setting is missing (or a service is unreachable) the service falls back to the
bundled model and an in-memory store, so you can do the steps one at a time and stop at any point. Use **one US region
(`us-central1`)**: some free allowances (Cloud Storage) only apply in certain US regions.

## 0. One-time setup (PowerShell)

Install the CLI and sign in with a **personal Gmail** account (college accounts often block public access):

```powershell
winget install -e --id Google.CloudSDK
gcloud auth login
```

Create the project, link billing (required by Cloud Build and Cloud Run even inside the free allowance; set a budget alert
at about $1 in Billing, Budgets and alerts, which only emails you), and set variables used below:

```powershell
$PROJECT = "stage-pm-demo-4821"      # must be globally unique: change the digits if it already exists
$REGION  = "us-central1"
$BUCKET  = "$PROJECT-models"
$SA      = "stage-pm-run@$PROJECT.iam.gserviceaccount.com"

gcloud projects create $PROJECT --name="STAGE-PM"
gcloud config set project $PROJECT
gcloud billing accounts list
gcloud billing projects link $PROJECT --billing-account=XXXXXX-XXXXXX-XXXXXX
```

## 1. Basic deployment (do this first; it is also your fallback)

```powershell
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
cd "C:\jiya\Cloud Computing\air-quality-forecasting"
gcloud run deploy stage-pm --source . --region $REGION --allow-unauthenticated --memory 2Gi --cpu 1 --min-instances 0 --max-instances 2
$URL = gcloud run services describe stage-pm --region $REGION --format "value(status.url)"
curl.exe "$URL/health"
```

Say **Y** if asked to create a repository. The first build takes 5-10 minutes (PyTorch). Open `$URL` in a browser.

## 2. Add the other services (each step is independent)

### 2.1 Enable the APIs

```powershell
gcloud services enable storage.googleapis.com firestore.googleapis.com secretmanager.googleapis.com cloudscheduler.googleapis.com monitoring.googleapis.com logging.googleapis.com
```

### 2.2 Cloud Storage: versioned model and data

```powershell
gcloud storage buckets create "gs://$BUCKET" --location=$REGION --uniform-bucket-level-access
gcloud storage buckets update "gs://$BUCKET" --versioning
gcloud storage cp deploy/model/best.pt "gs://$BUCKET/model/best.pt"
gcloud storage cp deploy/replay.npz "gs://$BUCKET/data/replay.npz"
```

### 2.3 Firestore

```powershell
gcloud firestore databases create --location=$REGION --type=firestore-native
```

### 2.4 IAM: a dedicated service account with minimal roles

```powershell
gcloud iam service-accounts create stage-pm-run --display-name="STAGE-PM Cloud Run"
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:$SA" --role="roles/datastore.user"
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member="serviceAccount:$SA" --role="roles/storage.objectUser"
```

### 2.5 Secret Manager: the API key

```powershell
$KEY = [guid]::NewGuid().ToString("N")
$KEY | gcloud secrets create stage-pm-api-key --data-file=- --replication-policy=automatic
gcloud secrets add-iam-policy-binding stage-pm-api-key --member="serviceAccount:$SA" --role="roles/secretmanager.secretAccessor"
```

Keep `$KEY` in this window (you need it in 2.7). It is never written into the repository or the image.

### 2.6 Redeploy with the cloud settings

```powershell
gcloud run deploy stage-pm --source . --region $REGION --allow-unauthenticated --memory 2Gi --cpu 1 --min-instances 0 --max-instances 2 --service-account $SA --set-secrets "STAGEPM_API_KEY=stage-pm-api-key:latest" --set-env-vars "STAGEPM_MODEL_URI=gs://$BUCKET/model/best.pt,STAGEPM_REPLAY_URI=gs://$BUCKET/data/replay.npz,STAGEPM_FIRESTORE=1,STAGEPM_BUCKET=$BUCKET"
curl.exe "$URL/api/cloud"
```

`/api/cloud` should now report the model coming from Cloud Storage, `"store": "firestore"` and the job endpoint enabled. The dashboard's
"Cloud pipeline" panel shows the same.

### 2.7 Cloud Scheduler: the hourly job

```powershell
curl.exe -X POST -H "X-API-Key: $KEY" "$URL/jobs/run-forecast"
gcloud scheduler jobs create http stage-pm-hourly --location $REGION --schedule "0 * * * *" --time-zone "Asia/Kolkata" --uri "$URL/jobs/run-forecast" --http-method POST --headers "X-API-Key=$KEY"
gcloud scheduler jobs run stage-pm-hourly --location $REGION
curl.exe "$URL/api/history?limit=3"
```

The first command tests the endpoint by hand; the last shows the stored forecasts. Check the **Firestore** console too (collections
`forecasts`, `feedback`, `state`) and the **Cloud Storage** bucket (`forecasts/latest.json`).

### 2.8 Cloud Monitoring and Logging

In the console: Monitoring, Uptime checks, Create. Protocol HTTPS, host = your run.app host (without `https://`), path `/health`, every 5 minutes.
Logging, Log-based metrics, Create: filter `resource.type="cloud_run_revision" textPayload:"job run-forecast"`. Read logs from the CLI:

```powershell
gcloud run services logs read stage-pm --region $REGION --limit 20
```

Alert policies may not be free: create an uptime check and the metric, and only add an alert policy if the pricing page says it is free.

### 2.9 CI/CD with Cloud Build

```powershell
gcloud artifacts repositories create stage-pm --repository-format=docker --location=$REGION
```

Connect GitHub once in the console (Cloud Build, Triggers, Connect repository, GitHub, select `Jiya1101/stage-pm-air-quality-forecasting`). Then:

```powershell
gcloud builds triggers create github --name=stage-pm-main --repo-owner=Jiya1101 --repo-name=stage-pm-air-quality-forecasting --branch-pattern="^main$" --build-config=cloudbuild.yaml
$PN = gcloud projects describe $PROJECT --format "value(projectNumber)"
$BUILDER = "$PN-compute@developer.gserviceaccount.com"
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:$BUILDER" --role="roles/run.admin"
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:$BUILDER" --role="roles/iam.serviceAccountUser"
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:$BUILDER" --role="roles/artifactregistry.writer"
```

If the build later says it ran as a different account, grant the same three roles to the account named in the error. Commit and push
something small to `main` and watch the build in the Cloud Build history.

## 3. Evidence to collect for the report (dated screenshots)

Cloud Run service page; the run.app URL in a browser with the dashboard's Cloud pipeline panel; Firestore collections; the Cloud Storage bucket
(versioning on); Secret Manager secret list (not the value); Cloud Scheduler job and its last run; Monitoring uptime check; Cloud Build trigger and
build history; Artifact Registry repository; IAM page showing `stage-pm-run`. Never show the API key.

## 4. Cost control and clean-up

* `--min-instances 0` means nothing runs (and nothing is billed) when idle; `--max-instances 2` caps the worst case. Feedback writes are
  rate-limited to 20 a minute per client so public use cannot flood Firestore.
* Free allowances change: before relying on them, read the pricing page of each service. The budget alert only emails you; it does not stop spending.
* When you are done:

```powershell
gcloud scheduler jobs delete stage-pm-hourly --location $REGION
gcloud run services delete stage-pm --region $REGION
gcloud projects delete $PROJECT
```

Deleting the project removes everything, including stored data and images.

## 5. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| "Billing must be enabled" | Link a billing account (section 0) |
| Container does not start | `gcloud run services logs read stage-pm --region $REGION --limit 50`; send the last lines |
| `/api/cloud` shows `bundled` / `memory` after step 2.6 | The log line "could not load ... using the bundled file" or "Firestore requested but unavailable" names the cause: usually a missing role (2.4), a wrong bucket name, or the Firestore database not created (2.3) |
| `401` from `/jobs/run-forecast` | The key in the header differs from the secret; re-create the Scheduler job with the current `$KEY` |
| `503` from `/jobs/run-forecast` | `STAGEPM_API_KEY` is not set on the service; redo 2.5 and 2.6 |
| Scheduler asks for an App Engine app | Accept; it is only needed to host the scheduler location |
| Build trigger fails with permission errors | Grant the three roles in 2.9 to the account named in the error |
| Public access blocked | You are on a college or organisation account; use a personal Gmail |

## Endpoints

| Endpoint | Returns |
|---|---|
| `GET /` | Dashboard (time slider, station table, trend chart, source mix, regime, Cloud pipeline panel) |
| `GET /api/forecast?t=<ISO time>` | +1/+6/+24 h forecast per station with 95% range, exceedance probability, AQI band, source and regime outputs, and the actual reading when one exists |
| `GET /api/series/{station_id}` | Observed PM2.5 history |
| `GET /api/meta` | Model, replay range, stations |
| `GET /api/cloud` | Which cloud features are active (platform, revision, model source, store, bucket) |
| `GET /api/history?limit=` | Stored forecasts |
| `POST /api/feedback` | Useful / not useful rating (validated, rate-limited) |
| `POST /jobs/run-forecast` | Hourly job; needs the `X-API-Key` header |
| `GET /health` | Liveness check |

The dashboard replays stored data (ERA5 weather lags real time by about five days); the hourly job moves the replay clock on one hour per call.

## Run locally (no cloud account)

```bash
STAGEPM_API_KEY=local-test PYTHONPATH=src python -m uvicorn aqf.serve.app:app --port 8080
```

`python tests/test_cloud.py` checks the cloud logic without any cloud account.
