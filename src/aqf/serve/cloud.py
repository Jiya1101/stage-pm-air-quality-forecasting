"""Optional Google Cloud integrations for the forecast service.

Every feature is switched on by an environment variable and has a local fallback, so the same code runs on a laptop
(no cloud account, no extra packages) and on Cloud Run:

    STAGEPM_MODEL_URI / STAGEPM_REPLAY_URI   gs://bucket/object   Cloud Storage: load the model / replay data from a bucket
    STAGEPM_BUCKET                           bucket name          Cloud Storage: also write forecasts/latest.json
    STAGEPM_FIRESTORE=1                      (+ optional STAGEPM_FIRESTORE_PROJECT)  Firestore: forecast history + feedback
    STAGEPM_API_KEY                          secret value         guards POST /jobs/run-forecast (Secret Manager on Cloud Run)

The google-cloud packages are imported lazily, only when their feature is on.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

log = logging.getLogger("stagepm.cloud")


# ----------------------------------------------------------------------------- Cloud Storage
def fetch_gcs(uri: str, local_path: str) -> str:
    """Download gs://bucket/path/to/object to local_path with the Cloud Run service account's credentials."""
    if not uri.startswith("gs://"):
        raise ValueError(f"not a gs:// URI: {uri}")
    bucket, _, blob = uri[5:].partition("/")
    if not bucket or not blob:
        raise ValueError(f"gs:// URI needs a bucket and an object name: {uri}")
    from google.cloud import storage

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    storage.Client().bucket(bucket).blob(blob).download_to_filename(local_path)
    return local_path


def upload_json(bucket: str, name: str, payload: dict) -> str:
    """Write a small JSON file to Cloud Storage and return its gs:// URI."""
    from google.cloud import storage

    storage.Client().bucket(bucket).blob(name).upload_from_string(json.dumps(payload), content_type="application/json")
    return f"gs://{bucket}/{name}"


# ----------------------------------------------------------------------------- API key (Secret Manager value)
def key_ok(supplied: str | None, expected: str | None) -> bool:
    """Constant-time comparison; an unset expected key never matches."""
    if not expected or not expected.strip() or supplied is None:
        return False
    return hmac.compare_digest(supplied.strip().encode(), expected.strip().encode())


# ----------------------------------------------------------------------------- abuse protection
class RateLimiter:
    """At most `max_events` per `window_s` seconds per key (per instance, in memory). Keeps free-tier writes bounded."""

    def __init__(self, max_events: int = 20, window_s: float = 60.0):
        self.max_events, self.window_s = max_events, window_s
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window_s:
                q.popleft()
            if len(q) >= self.max_events:
                return False
            q.append(now)
            return True


# ----------------------------------------------------------------------------- forecast documents
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_time(s: str) -> str:
    """'2025-11-12 06:00:00' -> '2025-11-12T06:00:00Z' (also a valid Firestore document id)."""
    return s.strip().replace(" ", "T").rstrip("Z") + "Z"


def summarise(forecast: dict) -> dict:
    """Compact document stored for one issued forecast: headline numbers plus every station's forecast and actuals."""
    st = forecast["stations"]
    last = lambda s: s["forecast"][-1]
    return {
        "issued_at": iso_time(forecast["issued_at"]),
        "created_at": _now_iso(),
        "stability_index": forecast["stability_index"],
        "mean_now": round(sum(s["current_pm25"] for s in st) / len(st), 1),
        "n_over_200_now": sum(1 for s in st if s["current_pm25"] > 200),
        "mean_pred_24h": round(sum(last(s)["pm25"] for s in st) / len(st), 1),
        "max_p_exceed_24h": max(last(s)["exceed_prob"] for s in st),
        "stations": [
            {"id": s["id"], "name": s["name"], "now": s["current_pm25"],
             "f": [{"h": f["horizon_h"], "pm25": f["pm25"], "lo95": f["lo95"], "hi95": f["hi95"],
                    "p_exceed": f["exceed_prob"], "actual": f["actual"]} for f in s["forecast"]]}
            for s in st
        ],
    }


def headline(doc: dict) -> dict:
    """The doc without the per-station detail, for list views."""
    return {k: v for k, v in doc.items() if k != "stations"}


# ----------------------------------------------------------------------------- stores
class MemoryStore:
    """Local stand-in for Firestore: same interface, lives only as long as the process."""

    kind = "memory"

    def __init__(self):
        self._forecasts: dict[str, dict] = {}
        self._feedback: list[dict] = []
        self._cursor: int | None = None
        self._lock = threading.Lock()

    def add_forecast(self, doc: dict) -> None:
        with self._lock:
            self._forecasts[doc["issued_at"]] = doc

    def recent_forecasts(self, limit: int = 24) -> list[dict]:
        with self._lock:
            return sorted(self._forecasts.values(), key=lambda d: d["created_at"], reverse=True)[:limit]

    def add_feedback(self, doc: dict) -> None:
        with self._lock:
            self._feedback.append(doc)

    def count_feedback(self) -> int:
        with self._lock:
            return len(self._feedback)

    def get_cursor(self) -> int | None:
        return self._cursor

    def set_cursor(self, cursor: int) -> None:
        self._cursor = cursor


class FirestoreStore:
    """Cloud Firestore (native mode). Collections: forecasts, feedback; one state/replay document holds the replay cursor."""

    kind = "firestore"

    def __init__(self, project: str | None = None):
        from google.cloud import firestore

        self._fs = firestore
        self.db = firestore.Client(project=project)

    def add_forecast(self, doc: dict) -> None:
        self.db.collection("forecasts").document(doc["issued_at"]).set(doc)

    def recent_forecasts(self, limit: int = 24) -> list[dict]:
        q = self.db.collection("forecasts").order_by("created_at", direction=self._fs.Query.DESCENDING).limit(limit)
        return [d.to_dict() for d in q.stream()]

    def add_feedback(self, doc: dict) -> None:
        self.db.collection("feedback").add(doc)

    def count_feedback(self) -> int:
        return sum(1 for _ in self.db.collection("feedback").limit(1000).stream())

    def get_cursor(self) -> int | None:
        snap = self.db.collection("state").document("replay").get()
        return snap.to_dict().get("cursor") if snap.exists else None

    def set_cursor(self, cursor: int) -> None:
        self.db.collection("state").document("replay").set({"cursor": int(cursor), "updated_at": _now_iso()})


def make_store():
    """Firestore when STAGEPM_FIRESTORE is set and reachable; otherwise the in-memory stand-in (logged, never fatal)."""
    if os.environ.get("STAGEPM_FIRESTORE", "").strip().lower() in ("1", "true", "yes"):
        try:
            store = FirestoreStore(project=os.environ.get("STAGEPM_FIRESTORE_PROJECT") or None)
            log.info("using Cloud Firestore")
            return store
        except Exception as e:  # missing package, no credentials, API not enabled ...
            log.warning("Firestore requested but unavailable (%r); using the in-memory store", e)
    return MemoryStore()
