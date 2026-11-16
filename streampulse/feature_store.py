"""Online feature store.

Holds the state every model needs, updated one event at a time:

* session state : running totals for each active session (stall time, switches, ...)
* viewer state  : long-term behaviour profile per viewer (genres, hours, completion)
* group state   : per-window aggregates for each (CDN, region) pair

Features are computed from this state at prediction time, so training and serving
use exactly the same code (no training/serving skew).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .config import CDNS, DEVICES, GENRES, REGIONS, TICK_SECONDS


@dataclass
class SessionState:
    viewer_id: int
    title_id: int
    genre: str
    device: str
    region: str
    cdn: str
    is_live: bool
    startup_ms: float
    start_hour: float
    beats: int = 0
    stall_ms_total: float = 0.0
    stall_events: int = 0
    switches: int = 0
    last_stall_beat: int = -999
    thr_hist: list = field(default_factory=list)  # last 3 log-throughputs
    thr_ewma: float | None = None
    thr_sum: float = 0.0
    bitrate: float = 0.0
    prev_bitrate: float = 0.0


@dataclass
class ViewerState:
    sessions: int = 0
    watch_s: float = 0.0
    completed: int = 0
    abandoned: int = 0
    genre_counts: np.ndarray = field(default_factory=lambda: np.zeros(len(GENRES)))
    hour_buckets: np.ndarray = field(default_factory=lambda: np.zeros(4))  # night/morning/afternoon/evening
    device_counts: np.ndarray = field(default_factory=lambda: np.zeros(len(DEVICES)))
    live: int = 0


class FeatureStore:
    ABANDON_FEATURES = (
        ["rebuffer_s_now", "stalled_recently", "stall_ratio_session", "stall_events", "buffer_s",
         "log_bitrate", "bitrate_drop", "switches_per_min", "log_throughput", "thr_to_bitrate",
         "log_startup_s", "minutes_watched", "dropped_frames", "is_live", "hour_sin", "hour_cos",
         "viewer_abandon_rate"]
        + [f"device_{d}" for d in DEVICES]
    )
    FORECAST_FEATURES = (
        ["lag1", "lag2", "lag3", "ewma", "session_mean", "lag_diff", "group_mean_now", "hour_sin", "hour_cos", "beats_log"]
        + [f"device_{d}" for d in DEVICES]
    )

    def __init__(self) -> None:
        self.sessions: dict[int, SessionState] = {}
        self.viewers: dict[int, ViewerState] = defaultdict(ViewerState)
        self.group_now: dict[tuple[str, str], dict] = {}

    # ----------------------------------------------------------- updates
    @staticmethod
    def _hour(ts: str) -> float:
        t = datetime.fromisoformat(ts)
        return t.hour + t.minute / 60

    def on_start(self, ev: dict) -> None:
        self.sessions[ev["session_id"]] = SessionState(
            ev["viewer_id"], ev["title_id"], ev.get("genre", "drama"), ev["device"], ev["region"],
            ev["cdn"], bool(ev.get("is_live", False)), ev["startup_ms"], self._hour(ev["ts"]))

    def on_heartbeat(self, ev: dict) -> SessionState | None:
        s = self.sessions.get(ev["session_id"])
        if s is None:  # heartbeat for a session we never saw start (e.g. its start went to DLQ)
            return None
        s.beats += 1
        if ev["rebuffer_ms"] > 0:
            s.stall_ms_total += ev["rebuffer_ms"]
            s.stall_events += 1
            s.last_stall_beat = s.beats
        s.switches += ev["bitrate_switch"]
        lt = math.log(max(ev["throughput_mbps"], 0.05))
        s.thr_hist = ([lt] + s.thr_hist)[:3]
        s.thr_ewma = lt if s.thr_ewma is None else 0.7 * s.thr_ewma + 0.3 * lt
        s.thr_sum += lt
        s.prev_bitrate = s.bitrate or ev["bitrate_mbps"]
        s.bitrate = ev["bitrate_mbps"]
        return s

    def on_end(self, ev: dict) -> SessionState | None:
        s = self.sessions.pop(ev["session_id"], None)
        if s is None:
            return None
        v = self.viewers[ev["viewer_id"]]
        v.sessions += 1
        v.watch_s += ev["watch_s"]
        v.completed += ev["end_reason"] == "completed"
        v.abandoned += ev["end_reason"] == "abandoned"
        v.genre_counts[GENRES.index(s.genre)] += 1
        v.hour_buckets[int(s.start_hour // 6) % 4] += 1
        v.device_counts[DEVICES.index(s.device)] += 1
        v.live += s.is_live
        return s

    def set_group_aggregates(self, heartbeats: list[dict]) -> None:
        """Aggregate this window's heartbeats per (CDN, region)."""
        acc: dict[tuple[str, str], list] = defaultdict(list)
        for hb in heartbeats:
            acc[(hb["cdn"], hb["region"])].append(hb)
        out = {}
        for key, hbs in acc.items():
            thr = np.log(np.maximum([h["throughput_mbps"] for h in hbs], 0.05))
            out[key] = {
                "n": len(hbs),
                "mean_log_thr": float(thr.mean()),
                "rebuffer_ratio": sum(h["rebuffer_ms"] for h in hbs) / (len(hbs) * TICK_SECONDS * 1000),
                "mean_bitrate": float(np.mean([h["bitrate_mbps"] for h in hbs])),
                "dropped_frames": float(np.mean([h["dropped_frames"] for h in hbs])),
                "_incident": any(h.get("_incident", False) for h in hbs),
            }
        self.group_now = out

    # ---------------------------------------------------------- features
    @staticmethod
    def _onehot(value: str, options: list[str]) -> list[float]:
        return [1.0 if value == o else 0.0 for o in options]

    def abandonment_features(self, ev: dict, s: SessionState) -> np.ndarray:
        hour = self._hour(ev["ts"])
        played_min = max(s.beats * TICK_SECONDS / 60, 0.5)
        v = self.viewers.get(s.viewer_id)
        v_rate = (v.abandoned + 0.3) / (v.sessions + 1) if v else 0.3
        x = [
            ev["rebuffer_ms"] / 1000,
            1.0 if s.beats - s.last_stall_beat <= 2 else 0.0,
            s.stall_ms_total / (s.beats * TICK_SECONDS * 1000),
            float(s.stall_events),
            ev["buffer_s"],
            math.log(ev["bitrate_mbps"]),
            1.0 if ev["bitrate_mbps"] < s.prev_bitrate else 0.0,
            s.switches / played_min,
            math.log(max(ev["throughput_mbps"], 0.05)),
            min(ev["throughput_mbps"] / ev["bitrate_mbps"], 20),
            math.log1p(s.startup_ms / 1000),
            played_min,
            math.log1p(ev["dropped_frames"]),
            float(s.is_live),
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            v_rate,
        ] + self._onehot(s.device, DEVICES)
        return np.asarray(x, dtype=float)

    def forecast_features(self, ev: dict, s: SessionState) -> np.ndarray | None:
        if len(s.thr_hist) < 1:
            return None
        h = s.thr_hist + [s.thr_hist[-1]] * (3 - len(s.thr_hist))
        g = self.group_now.get((s.cdn, s.region))
        hour = self._hour(ev["ts"])
        x = [h[0], h[1], h[2], s.thr_ewma, s.thr_sum / s.beats, h[0] - h[1],
             g["mean_log_thr"] if g else h[0],
             math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
             math.log1p(s.beats)] + self._onehot(s.device, DEVICES)
        return np.asarray(x, dtype=float)

    def viewer_profile(self, viewer_id: int) -> np.ndarray | None:
        v = self.viewers.get(viewer_id)
        if v is None or v.sessions < 3:
            return None
        n = v.sessions
        return np.concatenate([
            v.genre_counts / n,
            v.hour_buckets / n,
            [math.log1p(v.watch_s / n / 60), v.completed / n, v.abandoned / n, v.live / n],
        ])


# group key helpers
GROUPS = [(c, r) for c in CDNS for r in REGIONS]