"""Model-serving API for StreamPulse (standard library only, no extra installs).

Loads the LATEST version of each model from outputs/registry (run `python run.py` first).

Endpoints
---------
GET  /health                      -> status and loaded model versions
POST /v1/abandonment              -> {"features": {<feature name>: value, ...}}
                                     returns abandonment probability + risk level
POST /v1/forecast                 -> {"features": {...}} returns next-30s throughput (Mbps)
GET  /v1/recommendations?viewer_id=12&model=hybrid
                                  -> top-10 title ids for a viewer
POST /v1/reload                   -> reload LATEST versions (after a new publish)

Usage
-----
    python serve.py                   # serve on http://127.0.0.1:8000
    python serve.py --selftest        # start, send example requests, print responses, stop

Example
-------
    curl -s localhost:8000/v1/abandonment -d '{"features": {"rebuffer_s_now": 8, "buffer_s": 0, ...}}'
Missing features are filled with the training mean of that feature (and reported back),
so a partial request still works but tells the caller what was imputed.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from streampulse.registry import ModelRegistry

REGISTRY = ModelRegistry(Path(__file__).parent / "outputs" / "registry")


class Models:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.load()

    def load(self) -> None:
        with self.lock:
            self.abandon, self.abandon_meta = REGISTRY.load("abandonment")
            self.forecast, self.forecast_meta = REGISTRY.load("throughput_forecast")
            self.recs, self.recs_meta = REGISTRY.load("recommender")

    def versions(self) -> dict:
        return {m["name"]: {"version": m["version"], "sim_hour": m.get("sim_hour")}
                for m in (self.abandon_meta, self.forecast_meta, self.recs_meta)}


def vectorise(model, features: dict) -> tuple[np.ndarray, list[str]]:
    """Order named features the way the model expects; impute missing ones with the mean."""
    names, means = model.feature_names, model.scaler.mean_
    unknown = sorted(set(features) - set(names))
    if unknown:
        raise ValueError(f"unknown features: {unknown}")
    imputed = [n for n in names if n not in features]
    x = [float(features[n]) if n in features else float(means[i]) for i, n in enumerate(names)]
    return np.asarray([x]), imputed


def risk_level(p: float) -> str:
    return "high" if p >= 0.5 else "medium" if p >= 0.2 else "low"


def make_handler(models: Models):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/health":
                return self._send(200, {"status": "ok", "models": models.versions()})
            if url.path == "/v1/recommendations":
                q = parse_qs(url.query)
                try:
                    viewer = int(q.get("viewer_id", [""])[0])
                except ValueError:
                    return self._send(400, {"error": "viewer_id must be an integer"})
                model = q.get("model", ["hybrid"])[0]
                if model not in ("hybrid", "content", "mf", "popularity"):
                    return self._send(400, {"error": "model must be hybrid, content, mf or popularity"})
                if not 0 <= viewer < models.recs.P.shape[0]:
                    return self._send(404, {"error": f"viewer_id {viewer} not found"})
                titles = models.recs.recommend(viewer, model).tolist()
                return self._send(200, {"viewer_id": viewer, "model": model, "titles": titles,
                                        "model_version": models.recs_meta["version"]})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/v1/reload":
                models.load()
                return self._send(200, {"reloaded": models.versions()})
            if path not in ("/v1/abandonment", "/v1/forecast"):
                return self._send(404, {"error": "not found"})
            try:
                feats = self._json().get("features", {})
                model = models.abandon if path == "/v1/abandonment" else models.forecast
                X, imputed = vectorise(model, feats)
            except (ValueError, json.JSONDecodeError, TypeError) as e:
                return self._send(400, {"error": str(e)})
            if path == "/v1/abandonment":
                p = float(model.predict_proba(X)[0])
                return self._send(200, {"abandon_probability": round(p, 4), "risk": risk_level(p),
                                        "imputed_features": imputed,
                                        "model_version": models.abandon_meta["version"]})
            mbps = math.exp(float(model.predict(X)[0]))
            return self._send(200, {"throughput_next_30s_mbps": round(mbps, 2), "imputed_features": imputed,
                                    "model_version": models.forecast_meta["version"]})
    return Handler


def selftest(port: int) -> None:
    models = Models()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(models))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def call(method: str, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(base + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return {"http_status": e.code, **json.loads(e.read())}

    healthy = {"rebuffer_s_now": 0, "stalled_recently": 0, "buffer_s": 18, "log_bitrate": math.log(8),
               "log_throughput": math.log(25), "thr_to_bitrate": 3.1, "minutes_watched": 12}
    stalling = {"rebuffer_s_now": 12, "stalled_recently": 1, "stall_events": 3, "buffer_s": 0,
                "log_bitrate": math.log(1.6), "bitrate_drop": 1, "log_throughput": math.log(1.2),
                "thr_to_bitrate": 0.75, "minutes_watched": 4}
    print("GET  /health ->", call("GET", "/health"))
    print("POST /v1/abandonment (smooth playback)    ->", call("POST", "/v1/abandonment", {"features": healthy}))
    print("POST /v1/abandonment (stalling, low buffer)->", call("POST", "/v1/abandonment", {"features": stalling}))
    lt = math.log
    forecast = {"lag1": lt(20), "lag2": lt(22), "lag3": lt(21), "ewma": lt(21), "session_mean": lt(21),
                "lag_diff": lt(20) - lt(22), "group_mean_now": lt(18), "hour_sin": math.sin(2 * math.pi * 21 / 24),
                "hour_cos": math.cos(2 * math.pi * 21 / 24), "beats_log": math.log1p(30), "device_smart_tv": 1,
                "device_set_top_box": 0, "device_mobile": 0, "device_web": 0, "device_tablet": 0}
    print("POST /v1/forecast (recent ~20-22 Mbps, 9pm, smart TV) ->", call("POST", "/v1/forecast", {"features": forecast}))
    print("GET  /v1/recommendations?viewer_id=12 ->", call("GET", "/v1/recommendations?viewer_id=12"))
    print("POST /v1/abandonment (bad feature name) ->", call("POST", "/v1/abandonment", {"features": {"colour": 1}}))
    server.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest(args.port)
    models = Models()
    print(f"Serving {models.versions()} on http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(models)).serve_forever()


if __name__ == "__main__":
    main()
