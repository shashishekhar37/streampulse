"""Task 3: detect CDN/region quality incidents (unsupervised anomaly detection).

Every 30 s window, heartbeats are aggregated per (CDN, region): mean throughput,
rebuffering ratio, mean bitrate, dropped frames, session count. No labels are used for
training: incidents are rare and new kinds keep appearing, so we learn "normal" instead.

Two detectors:
    zscore          : per-group running mean/variance (EWMA) of peer-adjusted metrics.
                      Peer adjustment (subtract the median of all groups this window)
                      removes shared effects such as evening congestion (seasonality).
    isolation_forest: sklearn IsolationForest retrained every hour on the last 6 hours
                      of windows (sliding-window batch retraining inside a stream).

An alert needs 2 consecutive anomalous windows (debouncing cuts false alarms).
The injected incidents in config.INCIDENTS are used only for evaluation.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
from sklearn.ensemble import IsolationForest

from ..config import INCIDENTS, TICK_SECONDS
from ..feature_store import GROUPS

Z_THRESHOLD = 4.0
CONSECUTIVE = 2
MIN_SESSIONS = 3  # groups with fewer active streams are too noisy to judge


class _EWMA:
    """Exponentially weighted mean/variance with a floor on the standard deviation.

    The floor matters: a quiet group whose rebuffering is almost always 0 has a tiny
    variance, so a single stalled viewer would otherwise produce a huge z-score.
    """

    def __init__(self, alpha: float = 0.02, min_std: float = 0.05) -> None:
        self.alpha, self.min_std, self.mean, self.var, self.n = alpha, min_std, 0.0, 1.0, 0

    def z(self, x: float) -> float:
        std = max(np.sqrt(self.var), self.min_std)
        return (x - self.mean) / std if self.n > 20 else 0.0

    def update(self, x: float) -> None:
        self.n += 1
        if self.n == 1:
            self.mean, self.var = x, 0.05
            return
        d = x - self.mean
        self.mean += self.alpha * d
        self.var = (1 - self.alpha) * (self.var + self.alpha * d * d)


class AnomalyTask:
    name = "anomaly"

    def __init__(self, retrain_every_ticks: int = 120, train_window_ticks: int = 720) -> None:
        self.stats = defaultdict(lambda: {"thr": _EWMA(min_std=0.25), "reb": _EWMA(min_std=0.03)})
        self.history: deque = deque(maxlen=train_window_ticks * len(GROUPS))
        self.iforest: IsolationForest | None = None
        self.retrain_every = retrain_every_ticks
        self.streak = {"zscore": defaultdict(int), "iforest": defaultdict(int)}
        self.records: list[dict] = []  # one row per (tick, group)
        self.alerts: list[dict] = []

    def process(self, tick: int, groups: dict) -> None:
        keys = [k for k, g in groups.items() if g["n"] >= MIN_SESSIONS]
        if len(keys) < 5:
            return
        med_thr = float(np.median([groups[k]["mean_log_thr"] for k in keys]))
        med_reb = float(np.median([groups[k]["rebuffer_ratio"] for k in keys]))
        med_br = float(np.median([np.log(groups[k]["mean_bitrate"]) for k in keys]))

        rows = []
        for k in keys:
            g = groups[k]
            r_thr = g["mean_log_thr"] - med_thr
            r_reb = g["rebuffer_ratio"] - med_reb
            r_br = np.log(g["mean_bitrate"]) - med_br
            st = self.stats[k]
            z = max(-st["thr"].z(r_thr), st["reb"].z(r_reb))
            vec = [r_thr, r_reb, r_br, np.log(g["n"]), g["dropped_frames"]]
            rows.append((k, g, z, vec, r_thr, r_reb))

        if_scores = {}
        if self.iforest is not None:
            V = np.array([r[3] for r in rows])
            for (k, *_), s in zip(rows, self.iforest.decision_function(V)):
                if_scores[k] = float(s)

        for k, g, z, vec, r_thr, r_reb in rows:
            flags = {"zscore": z > Z_THRESHOLD, "iforest": if_scores.get(k, 1.0) < 0}
            alerting = {}
            for det, flag in flags.items():
                self.streak[det][k] = self.streak[det][k] + 1 if flag else 0
                alerting[det] = self.streak[det][k] >= CONSECUTIVE
                if self.streak[det][k] == CONSECUTIVE:
                    self.alerts.append({"tick": tick, "hour": round(tick * TICK_SECONDS / 3600, 2),
                                        "detector": det, "cdn": k[0], "region": k[1]})
            self.records.append({"tick": tick, "cdn": k[0], "region": k[1], "zscore": z,
                                 "iforest": if_scores.get(k, np.nan), "truth": g["_incident"],
                                 "alert_zscore": alerting["zscore"], "alert_iforest": alerting["iforest"]})
            if not flags["zscore"]:  # don't let an incident pollute the "normal" baseline
                self.stats[k]["thr"].update(r_thr)
                self.stats[k]["reb"].update(r_reb)
                self.history.append(vec)

        if tick and tick % self.retrain_every == 0 and len(self.history) > 2000:
            self.iforest = IsolationForest(n_estimators=150, contamination=0.001,
                                           random_state=0).fit(np.array(self.history))

    # ------------------------------------------------------- evaluation
    def evaluate(self) -> dict:
        out = {}
        if not self.records:
            return out
        truth = np.array([r["truth"] for r in self.records])
        for det in ("zscore", "iforest"):
            pred = np.array([r[f"alert_{det}"] for r in self.records])
            tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
            found, delays = 0, []
            for start_h, dur, cdn, region in INCIDENTS:
                t0, t1 = int(start_h * 120), int((start_h + dur / 60) * 120)
                hits = [a["tick"] for a in self.alerts if a["detector"] == det and a["cdn"] == cdn
                        and a["region"] == region and t0 <= a["tick"] <= t1]
                if hits:
                    found += 1
                    delays.append((min(hits) - t0) * TICK_SECONDS / 60)
            in_incident = lambda a: any(a["cdn"] == c and a["region"] == r and
                                        int(s * 120) <= a["tick"] <= int((s + d / 60) * 120) + 2
                                        for s, d, c, r in INCIDENTS)
            false_alerts = sum(1 for a in self.alerts if a["detector"] == det and not in_incident(a))
            out[det] = {"precision": tp / (tp + fp) if tp + fp else 0.0,
                        "recall": tp / (tp + fn) if tp + fn else 0.0,
                        "incidents_detected": f"{found}/{len(INCIDENTS)}",
                        "mean_detection_delay_min": float(np.mean(delays)) if delays else None,
                        "false_alerts": false_alerts}
        return out

    def report(self) -> dict:
        return {"anomaly_alerts_total": len(self.alerts)}

    def summary(self) -> dict:
        return self.evaluate()