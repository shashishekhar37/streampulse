"""Model and data monitoring.

* ``PageHinkley``   : concept-drift detector on a model's loss (is the model getting worse?)
* ``PSIMonitor``    : data-drift detector using the Population Stability Index
                      (has the input distribution changed vs. a reference period?)
* ``RollingWindow`` : keeps the last N (label, score) pairs for rolling metrics
"""

from __future__ import annotations

from collections import deque

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


class PageHinkley:
    def __init__(self, delta: float = 0.005, threshold: float = 5.0, alpha: float = 0.9999,
                 min_instances: int = 240) -> None:
        self.delta, self.threshold, self.alpha, self.min_instances = delta, threshold, alpha, min_instances
        self.reset()

    def reset(self) -> None:
        self.n, self.mean, self.cum, self.min_cum = 0, 0.0, 0.0, 0.0

    def update(self, value: float) -> bool:
        self.n += 1
        self.mean += (value - self.mean) / self.n
        self.cum = self.alpha * self.cum + (value - self.mean - self.delta)
        self.min_cum = min(self.min_cum, self.cum)
        if self.n >= self.min_instances and self.cum - self.min_cum > self.threshold:
            self.reset()
            return True
        return False


class PSIMonitor:
    """Population Stability Index per feature: sum((cur - ref) * ln(cur / ref)) over bins.

    Rule of thumb: < 0.1 stable, 0.1 - 0.25 moderate shift, > 0.25 significant shift.
    """

    def __init__(self, feature_names: list[str], reference_size: int = 20_000,
                 window: int = 20_000, bins: int = 10) -> None:
        self.names = feature_names
        self.reference_size, self.bins = reference_size, bins
        self._ref: list[np.ndarray] = []
        self.edges: list[np.ndarray] | None = None
        self.ref_frac: list[np.ndarray] = []
        self.current: deque = deque(maxlen=window)

    def update(self, X: np.ndarray) -> None:
        if self.edges is None:
            self._ref.extend(X)
            if len(self._ref) >= self.reference_size:
                R = np.vstack(self._ref)
                self.edges = [np.unique(np.quantile(R[:, j], np.linspace(0, 1, self.bins + 1)[1:-1]))
                              for j in range(R.shape[1])]
                self.ref_frac = [self._frac(R[:, j], j) for j in range(R.shape[1])]
                self._ref = []
        else:
            self.current.extend(X)

    def _frac(self, col: np.ndarray, j: int) -> np.ndarray:
        counts = np.bincount(np.searchsorted(self.edges[j], col, side="right"),
                             minlength=len(self.edges[j]) + 1)
        return np.clip(counts / max(len(col), 1), 1e-4, None)

    def compute(self) -> dict[str, float]:
        if self.edges is None or len(self.current) < 1000:
            return {}
        C = np.vstack(self.current)
        out = {}
        for j, name in enumerate(self.names):
            cur, ref = self._frac(C[:, j], j), self.ref_frac[j]
            out[name] = float(np.sum((cur - ref) * np.log(cur / ref)))
        return out


class RollingWindow:
    def __init__(self, size: int = 20_000) -> None:
        self.y: deque = deque(maxlen=size)
        self.scores: dict[str, deque] = {}

    def add(self, y: int, **scores: float) -> None:
        self.y.append(y)
        for k, v in scores.items():
            self.scores.setdefault(k, deque(maxlen=self.y.maxlen)).append(v)

    def classification_metrics(self) -> dict[str, float]:
        y = np.asarray(self.y)
        out: dict[str, float] = {}
        if len(set(y.tolist())) < 2:
            return out
        for k, s in self.scores.items():
            p = np.asarray(s)
            out[f"{k}_roc_auc"] = roc_auc_score(y, p)
            out[f"{k}_pr_auc"] = average_precision_score(y, p)
            out[f"{k}_brier"] = brier_score_loss(y, np.clip(p, 0, 1))
        out["positive_rate"] = float(y.mean())
        return out