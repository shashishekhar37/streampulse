"""Task 1: predict session abandonment (supervised, online, delayed labels).

Question: "Will this viewer quit within about the next 60 seconds?"

Every heartbeat is scored immediately. The true label is only known later: either the
session ends with reason "abandoned" in this window or the next (label 1), or the viewer
is still watching after that (label 0). Predictions wait in a pending buffer until their
label resolves; only then are they evaluated and learned from. This is prequential
evaluation with delayed labels, which is how production feedback loops behave.

Three scorers run side by side on the same events:
    online    : logistic regression updated continuously with partial_fit
    frozen    : same model trained once on the first 12 hours, never updated
    heuristic : rule of thumb "stalling now = likely to quit"

Monitoring: a Page-Hinkley test on the online model's mean log-loss per window detects
concept drift and raises an alert (in production: page the team / trigger a review).
Optionally (drift_boost > 1) the model also learns faster for 2 hours after an alert.
Experiments showed the boost slightly lowers PR-AUC here, because continuous learning
already adapts, so it is off by default. PSI tracks drift in the input features.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from ..feature_store import FeatureStore
from ..monitoring import PageHinkley, PSIMonitor, RollingWindow
from .models import OnlineClassifier

HORIZON_TICKS = 1  # this heartbeat's window or the next one: about 60 s
BOOST_TICKS = 240  # 2 hours of faster learning after a drift alarm


class AbandonmentTask:
    name = "abandonment"

    def __init__(self, frozen_train_hours: float = 12, drift_detection: bool = True,
                 drift_boost: float = 1.0) -> None:
        self.drift_boost = drift_boost
        names = FeatureStore.ABANDON_FEATURES
        self.online = OnlineClassifier(names, eta0=0.001)
        self.frozen = OnlineClassifier(names, eta0=0.001, seed=99)
        self.frozen_train_ticks = int(frozen_train_hours * 120)
        self._frozen_X: list[np.ndarray] = []
        self._frozen_y: list[np.ndarray] = []
        self.pending: dict[int, deque] = defaultdict(deque)
        self.window = RollingWindow(20_000)
        self.detector = PageHinkley() if drift_detection else None
        self.psi = PSIMonitor(["rebuffer_s_now", "log_throughput", "log_bitrate", "buffer_s"])
        self._psi_idx = [names.index(n) for n in self.psi.names]
        self._tick_losses: list[float] = []
        self.drift_ticks: list[int] = []
        self.boost_until = -1
        self.resolved = 0

    # --------------------------------------------------------- scoring
    def score(self, tick: int, items: list[tuple[dict, object]], fs: FeatureStore) -> None:
        if not items:
            return
        X = np.vstack([fs.abandonment_features(ev, s) for ev, s in items])
        p_on = self.online.predict_proba(X)
        p_fr = self.frozen.predict_proba(X)
        p_he = np.clip(X[:, 0] / 15 + 0.3 * X[:, 1], 0, 1)
        for (ev, _), x, a, b, c in zip(items, X, p_on, p_fr, p_he):
            self.pending[ev["session_id"]].append((tick, x, a, b, c))
        self.psi.update(X[:, self._psi_idx])

    # ------------------------------------------------- label resolution
    def on_session_end(self, tick: int, ev: dict) -> None:
        entries = self.pending.pop(ev["session_id"], None)
        if entries:
            abandoned = ev["end_reason"] == "abandoned"
            self._learn([(e, int(abandoned and tick - e[0] <= HORIZON_TICKS)) for e in entries], tick)

    def end_of_window(self, tick: int) -> None:
        """Call once per window, after session ends: time out labels, then check for drift."""
        done = []
        for q in self.pending.values():
            while q and q[0][0] + HORIZON_TICKS <= tick:
                done.append((q.popleft(), 0))  # still watching -> label 0
        self._learn(done, tick)
        self.pending = defaultdict(deque, {k: q for k, q in self.pending.items() if q})
        self._check_drift(tick)

    def _learn(self, labelled: list, tick: int) -> None:
        if not labelled:
            return
        X = np.vstack([e[1] for e, _ in labelled])
        y = np.array([lab for _, lab in labelled])
        for (_, _, a, b, c), lab in labelled:
            self.window.add(lab, online=a, frozen=b, heuristic=c)
            p = min(max(a, 1e-6), 1 - 1e-6)
            self._tick_losses.append(-np.log(p) if lab else -np.log(1 - p))
        self.online.learn(X, y)
        self.resolved += len(y)

        if not self.frozen.fitted:
            self._frozen_X.append(X)
            self._frozen_y.append(y)
            if tick >= self.frozen_train_ticks:
                self.frozen.learn(np.vstack(self._frozen_X), np.concatenate(self._frozen_y), epochs=5)
                self._frozen_X, self._frozen_y = [], []

    def _check_drift(self, tick: int) -> None:
        losses, self._tick_losses = self._tick_losses, []
        if tick == self.boost_until:
            self.online.boost_learning_rate(1.0)
        # wait for the warm-up (first 12 h) so early learning isn't mistaken for drift;
        # feed one value per window so the noise level doesn't depend on traffic volume
        if self.detector is None or not losses or not self.frozen.fitted:
            return
        if self.detector.update(float(np.mean(losses))):
            self.drift_ticks.append(tick)
            if self.drift_boost > 1:
                self.online.boost_learning_rate(self.drift_boost)
                self.boost_until = tick + BOOST_TICKS

    # ------------------------------------------------------- reporting
    def report(self) -> dict:
        m = self.window.classification_metrics()
        psi = self.psi.compute()
        return {**{f"abandon_{k}": v for k, v in m.items()},
                **{f"psi_{k}": v for k, v in psi.items()}}

    def summary(self) -> dict:
        return {"labels_resolved": self.resolved,
                "drift_detected_at_hour": [round(t / 120, 2) for t in self.drift_ticks],
                "top_coefficients": dict(sorted(self.online.coefficients().items(),
                                                key=lambda kv: -abs(kv[1]))[:8])}