"""Synthetic video-streaming telemetry.

``StreamingPlatformSimulator`` simulates a streaming service minute by minute and
yields the telemetry events a real player would send:

* ``session_start`` : a viewer presses play (title, device, region, CDN, startup time)
* ``heartbeat``     : every 30 s while watching (bitrate, throughput, buffer, stalls)
* ``session_end``   : the viewer stops (watch time, reason: completed / left / abandoned)

What makes it realistic enough to learn from:

* 2,000 viewers drawn from hidden personas with different tastes, devices and hours
* 400 titles with Zipf-like popularity (a few hits, a long tail)
* daily traffic pattern with an evening peak, and evening network congestion
* adaptive bitrate (ABR): the player picks the highest bitrate the network supports
* buffer model: when throughput < bitrate the buffer drains and the video stalls
* viewers abandon more when they see buffering (strength depends on persona)
* injected CDN incidents (see config.INCIDENTS) and concept drift (config.DRIFT_HOUR)
* a small fraction of malformed events, to exercise data validation

Fields starting with "_" are hidden ground truth for evaluation, never model input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator

import numpy as np

from .config import (BITRATE_LADDER, CDNS, DEVICES, DRIFT_HOUR, GENRES, INCIDENTS,
                     PERSONA_SHARE, PERSONAS, REGIONS, TICK_SECONDS)

LADDER = np.array(BITRATE_LADDER)


@dataclass
class _Session:
    session_id: int
    viewer: int
    title: int
    cdn: str
    start_tick: int
    intended_s: float
    bitrate: float
    est_thr: float
    ar: float = 0.0
    buffer_s: float = 0.0
    watched_s: float = 0.0
    beats: int = 0


class StreamingPlatformSimulator:
    def __init__(self, hours: float = 48, n_viewers: int = 1000, n_titles: int = 400,
                 traffic: float = 1.0, malformed_rate: float = 0.001, seed: int = 7,
                 start: datetime = datetime(2026, 3, 2)) -> None:
        self.hours = hours
        self.n_ticks = int(hours * 3600 / TICK_SECONDS)
        self.traffic = traffic
        self.malformed_rate = malformed_rate
        self.start = start
        self.rng = np.random.default_rng(seed)
        self._make_viewers(n_viewers)
        self._make_titles(n_titles)

    # ------------------------------------------------------------ population
    def _make_viewers(self, n: int) -> None:
        rng = self.rng
        names = list(PERSONA_SHARE)
        self.persona_names = names
        self.v_persona = rng.choice(len(names), size=n, p=[PERSONA_SHARE[k] for k in names])
        self.v_region = rng.integers(len(REGIONS), size=n)
        self.v_device = np.array([rng.choice(len(DEVICES), p=PERSONAS[names[p]]["devices"])
                                  for p in self.v_persona])
        self.v_bandwidth = np.exp(rng.normal(math.log(12), 0.7, size=n)).clip(1.5, 300)  # Mbps plan
        self.v_taste = np.vstack([
            rng.dirichlet(np.asarray(PERSONAS[names[p]]["genre_pref"]) * 2) for p in self.v_persona])
        hours = np.vstack([np.asarray(PERSONAS[names[p]]["hours"], float) for p in range(len(names))])
        self.persona_hours = hours / hours.mean(axis=1, keepdims=True)
        self.v_patience = np.array([PERSONAS[names[p]]["patience"] for p in self.v_persona])
        self.v_cdn_pref = rng.dirichlet([3, 3, 2], size=n)

    def _make_titles(self, n: int) -> None:
        rng = self.rng
        self.t_genre = rng.integers(len(GENRES), size=n)
        rank = rng.permutation(n) + 1
        self.t_pop = 1.0 / rank ** 0.9
        dur_min = np.empty(n)
        for i, g in enumerate(self.t_genre):
            name = GENRES[g]
            dur_min[i] = {"sports": rng.uniform(100, 180), "news": rng.uniform(10, 30),
                          "kids": rng.uniform(11, 25), "drama": rng.choice([45, 55, 120]),
                          "action": rng.uniform(95, 140), "documentary": rng.uniform(40, 95),
                          "comedy": rng.choice([22, 30, 95]), "reality": rng.uniform(40, 60)}[name]
        self.t_duration_s = dur_min * 60
        self.t_live = (np.array([GENRES[g] for g in self.t_genre]) == "sports") & (rng.random(n) < 0.6)

    # ------------------------------------------------------------- dynamics
    def _hour(self, tick: int) -> float:
        return tick * TICK_SECONDS / 3600

    def _congestion(self, sim_hour: float) -> float:
        h = sim_hour % 24
        peak = math.exp(-((h - 21) ** 2) / 8)  # evening peak around 21:00
        strength = 0.55 if sim_hour >= DRIFT_HOUR else 0.35
        return 1 - strength * peak

    def _cdn_factor(self, cdn: str, region: int, sim_hour: float) -> tuple[float, bool]:
        for start_h, dur_min, c, r in INCIDENTS:
            if c == cdn and REGIONS[region] == r and start_h <= sim_hour < start_h + dur_min / 60:
                return 0.1, True
        return {"cdn_a": 1.0, "cdn_b": 0.95, "cdn_c": 0.9}[cdn], False

    def _abr(self, est_thr: float) -> float:
        ok = LADDER[LADDER <= 0.9 * est_thr]
        return float(ok[-1]) if len(ok) else float(LADDER[0])

    # --------------------------------------------------------------- events
    def _ts(self, tick: int) -> str:
        jitter = self.rng.uniform(0, TICK_SECONDS - 0.001)
        return (self.start + timedelta(seconds=tick * TICK_SECONDS + jitter)).isoformat(timespec="milliseconds")

    def _maybe_corrupt(self, ev: dict) -> dict:
        if self.rng.random() >= self.malformed_rate:
            return ev
        kind = self.rng.integers(3)
        ev = dict(ev)
        if kind == 0:
            ev.pop("bitrate_mbps", None)
        elif kind == 1:
            ev["throughput_mbps"] = -1.0
        else:
            ev["cdn"] = "cdn_unknown"
        return ev

    def __iter__(self) -> Iterator[dict]:
        rng = self.rng
        active: dict[int, _Session] = {}
        busy = np.zeros(len(self.v_persona), dtype=bool)
        next_sid = 0
        title_base = self.t_pop ** 0.5

        for tick in range(self.n_ticks):
            sim_hour = self._hour(tick)
            hour_of_day = int(sim_hour % 24)
            drifted = sim_hour >= DRIFT_HOUR
            congestion = self._congestion(sim_hour)

            # ---- heartbeats and endings for active sessions
            for sid in list(active):
                s = active[sid]
                v = s.viewer
                s.beats += 1
                cdn_f, incident = self._cdn_factor(s.cdn, self.v_region[v], sim_hour)
                s.ar = 0.8 * s.ar + rng.normal(0, 0.4)
                thr = float(self.v_bandwidth[v] * congestion * cdn_f * math.exp(s.ar))
                if self.v_device[v] == DEVICES.index("mobile"):
                    thr *= 0.6
                s.est_thr = 0.6 * s.est_thr + 0.4 * thr
                new_bitrate = self._abr(s.est_thr)
                switched = new_bitrate != s.bitrate
                dropped_quality = new_bitrate < s.bitrate
                s.bitrate = new_bitrate

                downloaded = TICK_SECONDS * thr / s.bitrate  # seconds of video fetched
                s.buffer_s = min(20.0, s.buffer_s + downloaded - TICK_SECONDS)
                stall_s = 0.0
                if s.buffer_s < 0:
                    stall_s = min(TICK_SECONDS, -s.buffer_s)
                    s.buffer_s = 0.0
                s.watched_s += TICK_SECONDS - stall_s
                dropped = int(rng.poisson(2 + 40 * (s.bitrate > 0.9 * thr)))

                yield self._maybe_corrupt({
                    "type": "heartbeat", "ts": self._ts(tick), "session_id": sid, "viewer_id": v,
                    "device": DEVICES[self.v_device[v]], "region": REGIONS[self.v_region[v]],
                    "cdn": s.cdn, "bitrate_mbps": s.bitrate, "throughput_mbps": round(thr, 3),
                    "buffer_s": round(s.buffer_s, 2), "rebuffer_ms": int(stall_s * 1000),
                    "bitrate_switch": int(switched), "dropped_frames": dropped,
                    "position_s": round(s.watched_s, 1), "_incident": incident,
                })

                # abandonment hazard. Before the drift: stalls and very low picture quality drive
                # quitting. After the drift (e.g. a competitor launches in 4K): viewers stop
                # tolerating visible quality DROPS, and stalls hurt 50% more. Same features, new rules,
                # so a model trained before the drift misses a whole new cause of abandonment.
                pain = stall_s / self.v_patience[v]
                if drifted:
                    hazard = 0.001 + 0.045 * pain + 0.06 * dropped_quality
                else:
                    hazard = 0.001 + 0.03 * pain + 0.01 * (s.bitrate <= 1.6)
                reason = None
                if rng.random() < min(hazard, 0.9):
                    reason = "abandoned"
                elif s.watched_s >= s.intended_s:
                    reason = "completed" if s.watched_s >= 0.9 * self.t_duration_s[s.title] else "left"
                if reason:
                    yield {"type": "session_end", "ts": self._ts(tick), "session_id": sid, "viewer_id": v,
                           "title_id": s.title, "watch_s": round(s.watched_s, 1),
                           "duration_s": round(float(self.t_duration_s[s.title]), 1), "end_reason": reason}
                    busy[v] = False
                    del active[sid]

            # ---- new sessions
            idle = np.flatnonzero(~busy)
            p_start = 0.0022 * self.traffic * self.persona_hours[self.v_persona[idle], hour_of_day]
            starters = idle[rng.random(len(idle)) < p_start]
            for v in starters:
                w = title_base * self.v_taste[v, self.t_genre] ** 2
                title = int(rng.choice(len(w), p=w / w.sum()))
                cdn = CDNS[int(rng.choice(3, p=self.v_cdn_pref[v]))]
                cdn_f, _ = self._cdn_factor(cdn, self.v_region[v], sim_hour)
                thr0 = self.v_bandwidth[v] * congestion * cdn_f * (0.6 if self.v_device[v] == 2 else 1)
                bitrate = self._abr(thr0)
                startup_ms = int(600 + 4000 * bitrate / max(thr0, 0.1) + rng.exponential(300))
                persona = PERSONAS[self.persona_names[self.v_persona[v]]]
                frac = rng.beta(*persona["watch_frac"])
                s = _Session(next_sid, int(v), title, cdn, tick,
                             intended_s=max(60.0, frac * self.t_duration_s[title]),
                             bitrate=bitrate, est_thr=thr0, buffer_s=4.0)
                active[next_sid] = s
                busy[v] = True
                yield {"type": "session_start", "ts": self._ts(tick), "session_id": next_sid,
                       "viewer_id": int(v), "title_id": title, "genre": GENRES[self.t_genre[title]],
                       "is_live": bool(self.t_live[title]), "device": DEVICES[self.v_device[v]],
                       "region": REGIONS[self.v_region[v]], "cdn": cdn, "startup_ms": min(startup_ms, 30000),
                       "_persona": self.persona_names[self.v_persona[v]]}
                next_sid += 1