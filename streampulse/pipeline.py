"""StreamPulse pipeline: one pass over every 30-second micro-batch.

For each window, in this order:
    1. session_start events -> feature store, recommender (evaluate, then learn), personas
    2. heartbeats           -> feature store, then CDN/region aggregates for the window
    3. abandonment          -> score every heartbeat now (labels arrive later)
    4. forecasting          -> check last window's forecasts, learn, forecast next window
    5. anomaly detection    -> score each (CDN, region) aggregate
    6. session_end events   -> resolve abandonment labels, recommender feedback, viewer profiles
    7. label timeouts       -> heartbeats still watching after 60 s become label 0
    8. housekeeping         -> clustering refresh, hourly metrics, model registry publish
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from .feature_store import FeatureStore
from .ingest import DeadLetterQueue, MicroBatcher
from .registry import ModelRegistry
from .tasks.abandonment import AbandonmentTask
from .tasks.anomaly import AnomalyTask
from .tasks.forecasting import ForecastTask
from .tasks.recommender import RecommenderTask
from .tasks.segmentation import SegmentationTask

TICKS_PER_HOUR = 120


@dataclass
class RunResult:
    hourly: pd.DataFrame
    summary: dict
    anomaly_records: pd.DataFrame
    alerts: list = field(default_factory=list)


class StreamPulsePipeline:
    def __init__(self, registry_dir: str | Path | None = None, publish_every_hours: int = 6,
                 drift_detection: bool = True, drift_boost: float = 1.0, verbose: bool = True) -> None:
        self.fs = FeatureStore()
        self.dlq = DeadLetterQueue()
        self.abandon = AbandonmentTask(drift_detection=drift_detection, drift_boost=drift_boost)
        self.forecast = ForecastTask()
        self.anomaly = AnomalyTask()
        self.recs = RecommenderTask()
        self.segments = SegmentationTask()
        self.registry = ModelRegistry(registry_dir) if registry_dir else None
        self.publish_every = publish_every_hours * TICKS_PER_HOUR
        self.verbose = verbose
        self.counts = {"events": 0, "sessions": 0, "heartbeats": 0, "windows": 0}

    def run(self, events: Iterable[dict]) -> RunResult:
        rows: list[dict] = []
        t0 = time.perf_counter()
        last_tick = 0
        if self.verbose:
            print(f"{'hour':>5} {'events/s':>9} {'abandon PR-AUC on/frz/rule':>27} "
                  f"{'fcst MAE mdl/last':>18} {'rec HR@10 hyb/pop':>18} {'alerts':>6} {'DLQ':>5}")

        for batch in MicroBatcher(self.dlq).batches(events):
            tick = batch.window_index
            last_tick = tick
            self.counts["windows"] += 1
            self.counts["events"] += batch.size

            for ev in batch.starts:                                         # 1
                self.fs.on_start(ev)
                self.recs.on_session_start(ev)
                self.segments.on_session_start(ev)
            self.counts["sessions"] += len(batch.starts)

            items = []                                                      # 2
            for ev in batch.heartbeats:
                s = self.fs.on_heartbeat(ev)
                if s is not None:
                    items.append((ev, s))
            self.counts["heartbeats"] += len(items)
            self.fs.set_group_aggregates([ev for ev, _ in items])

            self.abandon.score(tick, items, self.fs)                        # 3
            self.forecast.process(items, self.fs)                           # 4
            self.anomaly.process(tick, self.fs.group_now)                   # 5

            for ev in batch.ends:                                           # 6
                self.abandon.on_session_end(tick, ev)
                self.forecast.on_session_end(ev)
                self.recs.on_session_end(ev)
                self.fs.on_end(ev)
            self.abandon.end_of_window(tick)                                # 7

            self.segments.maybe_update(tick, self.fs)                       # 8
            if tick and tick % TICKS_PER_HOUR == 0:
                rows.append(self._hourly_row(tick, t0))
            if self.registry and tick and tick % self.publish_every == 0:
                self._publish(tick)

        if self.registry:
            self._publish(last_tick)
        summary = self._summary(time.perf_counter() - t0)
        return RunResult(pd.DataFrame(rows), summary, pd.DataFrame(self.anomaly.records), self.anomaly.alerts)

    # ---------------------------------------------------------------- helpers
    def _hourly_row(self, tick: int, t0: float) -> dict:
        row = {"hour": tick // TICKS_PER_HOUR, **self.abandon.report(), **self.forecast.report(),
               **self.recs.report(), **self.anomaly.report(), "dlq_events": self.dlq.count,
               "events_per_s": self.counts["events"] / (time.perf_counter() - t0)}
        if self.verbose:
            g = lambda k: row.get(k, float("nan"))
            print(f"{row['hour']:>5} {row['events_per_s']:>9,.0f} "
                  f"{g('abandon_online_pr_auc'):>9.3f}/{g('abandon_frozen_pr_auc'):.3f}/{g('abandon_heuristic_pr_auc'):.3f} "
                  f"{g('forecast_mae_model'):>10.2f}/{g('forecast_mae_persistence'):.2f} "
                  f"{g('rec_hit@10_hybrid'):>11.3f}/{g('rec_hit@10_popularity'):.3f} "
                  f"{row['anomaly_alerts_total']:>6} {row['dlq_events']:>5}")
        return row

    def _publish(self, tick: int) -> None:
        rep = self.abandon.report()
        self.registry.publish("abandonment", self.abandon.online, {
            "sim_hour": tick / TICKS_PER_HOUR, "n_trained": self.abandon.online.n_trained,
            "feature_names": self.abandon.online.feature_names,
            "metrics": {k: rep[k] for k in rep if k.startswith("abandon_online")}})
        frep = self.forecast.report()
        self.registry.publish("throughput_forecast", self.forecast.model, {
            "sim_hour": tick / TICKS_PER_HOUR, "n_trained": self.forecast.model.n_trained,
            "feature_names": self.forecast.model.feature_names,
            "metrics": {k: frep[k] for k in frep if k.endswith("_model")}})
        rrep = self.recs.report()
        self.registry.publish("recommender", self.recs, {
            "sim_hour": tick / TICKS_PER_HOUR, "viewers_seen": len(self.recs.history),
            "metrics": {k: v for k, v in rrep.items() if k.startswith("rec_hit")}})

    def _summary(self, seconds: float) -> dict:
        return {
            "throughput": {**self.counts, "seconds": round(seconds, 1),
                           "events_per_s": round(self.counts["events"] / seconds)},
            "data_quality": {"dead_letter_events": self.dlq.count, "reasons": dict(self.dlq.reasons)},
            "abandonment": {**self.abandon.report(), **self.abandon.summary()},
            "forecast": {**self.forecast.report(), **self.forecast.summary()},
            "anomaly": self.anomaly.summary(),
            "recommender": {**self.recs.report(), **self.recs.summary()},
            "segmentation": self.segments.summary(self.fs),
        }