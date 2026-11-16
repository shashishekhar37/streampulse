"""Task 2: forecast each session's network throughput 30 s ahead (online regression).

Why it matters: an adaptive-bitrate player that knows the next 30 s of bandwidth can
pick a bitrate that won't stall. This is a time-series problem solved with lag
features (last 3 values, EWMA, trend) plus context (CDN/region load, hour, device).

The target is log(throughput); errors are reported in Mbps. Compared against two
classic baselines: persistence ("same as last time") and the session EWMA.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..feature_store import FeatureStore
from .models import ResidualForecaster


class ForecastTask:
    name = "forecast"

    def __init__(self) -> None:
        self.model = ResidualForecaster(FeatureStore.FORECAST_FEATURES)
        self.pending: dict[int, tuple] = {}
        self.err = {k: deque(maxlen=20_000) for k in ("model", "persistence", "ewma")}
        self.sq = {k: deque(maxlen=20_000) for k in ("model", "persistence", "ewma")}

    def process(self, items: list[tuple[dict, object]], fs: FeatureStore) -> None:
        learn_X, learn_y = [], []
        for ev, s in items:  # 1) the actual value for last tick's forecast has arrived
            prev = self.pending.pop(ev["session_id"], None)
            if prev is None:
                continue
            x, preds = prev
            actual = max(ev["throughput_mbps"], 0.05)
            for k, logp in preds.items():
                e = float(np.exp(logp)) - actual
                self.err[k].append(abs(e))
                self.sq[k].append(e * e)
            learn_X.append(x)
            learn_y.append(np.log(actual))
        if learn_X:  # 2) learn from the resolved examples
            self.model.learn(np.vstack(learn_X), np.asarray(learn_y))

        feats = [(ev, fs.forecast_features(ev, s)) for ev, s in items]  # 3) forecast next tick
        feats = [(ev, x) for ev, x in feats if x is not None]
        if not feats:
            return
        X = np.vstack([x for _, x in feats])
        pm = self.model.predict(X)
        for (ev, x), p in zip(feats, pm):
            self.pending[ev["session_id"]] = (x, {"model": p, "persistence": x[0], "ewma": x[3]})

    def on_session_end(self, ev: dict) -> None:
        self.pending.pop(ev["session_id"], None)

    def report(self) -> dict:
        out = {}
        for k in self.err:
            if self.err[k]:
                out[f"forecast_mae_{k}"] = float(np.mean(self.err[k]))
                out[f"forecast_rmse_{k}"] = float(np.sqrt(np.mean(self.sq[k])))
        return out

    def summary(self) -> dict:
        return {"examples_trained": self.model.n_trained}