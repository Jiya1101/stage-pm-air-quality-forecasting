"""Tests for the cloud features that need no cloud account: key check, rate limiter, store, job endpoint, feedback, history.

Run:  python tests/test_cloud.py
Uses the deploy/ model and replay data and FastAPI's TestClient; Cloud Storage and Firestore themselves are not
exercised here (they need a Google account) -- the in-memory store implements the same interface.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("STAGEPM_CHECKPOINT", "deploy/model/best.pt")
os.environ.setdefault("STAGEPM_REPLAY", "deploy/replay.npz")
os.environ["STAGEPM_API_KEY"] = "test-key-123"
for k in ("STAGEPM_MODEL_URI", "STAGEPM_REPLAY_URI", "STAGEPM_BUCKET", "STAGEPM_FIRESTORE"):
    os.environ.pop(k, None)

from fastapi.testclient import TestClient

from aqf.serve import cloud
from aqf.serve.app import app


def test_key_and_limiter():
    assert cloud.key_ok("abc", "abc") and cloud.key_ok(" abc\r\n", "abc\n")  # trailing whitespace from secrets is tolerated
    assert not cloud.key_ok("abd", "abc") and not cloud.key_ok(None, "abc") and not cloud.key_ok("x", "") and not cloud.key_ok("", None)
    rl = cloud.RateLimiter(max_events=3, window_s=60)
    assert [rl.allow("a") for _ in range(4)] == [True, True, True, False] and rl.allow("b")
    print("OK: API key comparison and rate limiter")


def test_iso_and_gcs_uri_validation():
    assert cloud.iso_time("2025-11-12 06:00:00") == "2025-11-12T06:00:00Z" and cloud.iso_time("2025-11-12T06:00:00Z") == "2025-11-12T06:00:00Z"
    for bad in ("model.pt", "gs://bucket", "gs:///x"):
        try:
            cloud.fetch_gcs(bad, "x")
            raise AssertionError("accepted " + bad)
        except ValueError:
            pass
    print("OK: timestamp normalisation and gs:// URI validation")


def test_service():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        info = c.get("/api/cloud").json()
        assert info["platform"] == "local" and info["model_source"] == "bundled" and info["store"] == "memory" and info["job_endpoint_enabled"]

        assert c.post("/jobs/run-forecast").status_code == 401                                      # no key
        assert c.post("/jobs/run-forecast", headers={"X-API-Key": "wrong"}).status_code == 401      # wrong key
        r1 = c.post("/jobs/run-forecast", headers={"X-API-Key": "test-key-123"}).json()
        r2 = c.post("/jobs/run-forecast", headers={"X-API-Key": "test-key-123"}).json()
        assert r1["stored"] and r1["cursor"] == 0 and r2["cursor"] == 1 and r1["issued_at"] < r2["issued_at"]  # replay clock moves one hour on

        h = c.get("/api/history?limit=5").json()
        assert h["store"] == "memory" and len(h["items"]) == 2 and "stations" not in h["items"][0]
        assert {"issued_at", "mean_now", "n_over_200_now", "max_p_exceed_24h"} <= set(h["items"][0])

        ok = c.post("/api/feedback", json={"issued_at": r2["issued_at"], "rating": "useful", "station_id": "DL001", "comment": "ok"})
        assert ok.status_code == 200 and ok.json()["stored"]
        assert c.post("/api/feedback", json={"issued_at": "x", "rating": "great"}).status_code == 422           # not an allowed rating
        assert c.post("/api/feedback", json={"issued_at": "x", "rating": "useful", "comment": "y" * 500}).status_code == 422  # too long
        codes = [c.post("/api/feedback", json={"issued_at": "x", "rating": "useful"}).status_code for _ in range(25)]
        assert 429 in codes                                                                               # rate limit kicks in
        print("OK: job endpoint (auth, replay clock), history, feedback validation and rate limit")

    os.environ["STAGEPM_API_KEY"] = ""
    with TestClient(app) as c:                                                                             # key unset -> endpoint disabled
        assert c.post("/jobs/run-forecast", headers={"X-API-Key": "anything"}).status_code == 503
        print("OK: job endpoint is disabled when no API key is configured")


def test_google_clients_against_stand_ins():
    """Exercise FirestoreStore, fetch_gcs/upload_json and the start-up fallbacks with stand-ins for the Google libraries."""
    import types
    from datetime import datetime

    docs, blobs = {}, {}

    class Snap:
        def __init__(self, d): self._d = d; self.exists = d is not None
        def to_dict(self): return dict(self._d)
    class Doc:
        def __init__(self, col, id): self.col, self.id = col, id
        def set(self, data): docs.setdefault(self.col, {})[self.id] = dict(data)
        def get(self): return Snap(docs.get(self.col, {}).get(self.id))
    class Query:
        DESCENDING = "DESC"
    class Col:
        def __init__(self, name): self.name, self._lim, self._order = name, None, None
        def document(self, id): return Doc(self.name, id)
        def add(self, data): docs.setdefault(self.name, {})[f"auto{len(docs.get(self.name, {}))}"] = dict(data)
        def order_by(self, field, direction=None): self._order = (field, direction); return self
        def limit(self, n): self._lim = n; return self
        def stream(self):
            items = list(docs.get(self.name, {}).values())
            if self._order: items.sort(key=lambda d: d[self._order[0]], reverse=self._order[1] == "DESC")
            return [Snap(d) for d in items[: self._lim]]
    class FsClient:
        def __init__(self, project=None): self.project = project
        def collection(self, name): return Col(name)
    fs = types.ModuleType("google.cloud.firestore"); fs.Client = FsClient; fs.Query = Query

    class Blob:
        def __init__(self, b, n): self.b, self.n = b, n
        def download_to_filename(self, path): open(path, "wb").write(blobs[(self.b, self.n)])
        def upload_from_string(self, data, content_type=None): blobs[(self.b, self.n)] = data.encode() if isinstance(data, str) else data
    class Bucket:
        def __init__(self, b): self.b = b
        def blob(self, n): return Blob(self.b, n)
    class StClient:
        def bucket(self, b): return Bucket(b)
    st = types.ModuleType("google.cloud.storage"); st.Client = StClient

    gc = types.ModuleType("google.cloud"); gc.firestore, gc.storage = fs, st
    g = types.ModuleType("google"); g.cloud = gc
    saved = {k: sys.modules.get(k) for k in ("google", "google.cloud", "google.cloud.firestore", "google.cloud.storage")}
    sys.modules.update({"google": g, "google.cloud": gc, "google.cloud.firestore": fs, "google.cloud.storage": st})
    try:
        store = cloud.FirestoreStore()
        assert store.get_cursor() is None
        store.set_cursor(7); assert store.get_cursor() == 7
        for i in range(3):
            store.add_forecast({"issued_at": f"2025-11-0{i + 1}T00:00:00Z", "created_at": f"2026-10-16T00:00:0{i}Z", "mean_now": i})
        assert [d["mean_now"] for d in store.recent_forecasts(2)] == [2, 1]          # newest first, limited
        store.add_feedback({"rating": "useful"}); assert store.count_feedback() == 1

        url = cloud.upload_json("bkt", "forecasts/latest.json", {"a": 1})
        assert url == "gs://bkt/forecasts/latest.json" and blobs[("bkt", "forecasts/latest.json")] == b'{"a": 1}'
        blobs[("bkt", "model/best.pt")] = b"abc"
        out = cloud.fetch_gcs("gs://bkt/model/best.pt", os.path.join(os.environ.get("TEMP", "."), "stagepm_test", "best.pt"))
        assert open(out, "rb").read() == b"abc"

        os.environ["STAGEPM_FIRESTORE"] = "1"
        assert cloud.make_store().kind == "firestore"
    finally:
        for k, v in saved.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v

    # with the libraries absent, a requested cloud feature must fall back, never crash the service
    os.environ.update({"STAGEPM_FIRESTORE": "1", "STAGEPM_MODEL_URI": "gs://no-such-bucket/best.pt", "STAGEPM_API_KEY": "k"})
    with TestClient(app) as c:
        info = c.get("/api/cloud").json()
        assert info["store"] == "memory" and info["model_source"] == "bundled" and c.get("/health").status_code == 200
    for k in ("STAGEPM_FIRESTORE", "STAGEPM_MODEL_URI"):
        os.environ.pop(k, None)
    print("OK: Firestore / Cloud Storage code paths against stand-ins, and safe fallbacks when the cloud is unreachable")


if __name__ == "__main__":
    test_key_and_limiter(); test_iso_and_gcs_uri_validation(); test_service(); test_google_clients_against_stand_ins()
    print("\nAll cloud tests passed.")
