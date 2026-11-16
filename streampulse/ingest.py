"""Ingestion layer: broker-like queue, schema validation, event-time micro-batching.

    simulator ──► QueueSource (producer thread) ──► validate ──► MicroBatcher ──► pipeline
                                                       │
                                                       └──► dead-letter queue (bad events)

* ``QueueSource`` plays the role of Kafka/Kinesis: a producer thread publishes events
  into a bounded queue; the consumer reads them. Swap it for a real consumer later.
* ``validate`` checks every event against a schema. Bad events go to a dead-letter
  queue (DLQ) instead of crashing the pipeline or poisoning the models.
* ``MicroBatcher`` groups events into 30-second tumbling windows by **event time**
  (the timestamp inside the event, not arrival time). A window is closed and emitted
  when the first event of a later window arrives (a simple watermark).
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from .config import CDNS, DEVICES, REGIONS, TICK_SECONDS

_END = object()


class QueueSource:
    def __init__(self, events: Iterable[dict], rate: float = 0.0, maxsize: int = 50_000,
                 tap_file: str | Path | None = None) -> None:
        self.events, self.rate = events, rate
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.tap_file = Path(tap_file) if tap_file else None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._produce, daemon=True, name="producer")

    def _produce(self) -> None:
        interval = 1.0 / self.rate if self.rate > 0 else 0.0
        tap = self.tap_file.open("w") if self.tap_file else None
        next_t = time.perf_counter()
        try:
            for ev in self.events:
                if self._stop.is_set():
                    break
                if interval:
                    next_t += interval
                    if (d := next_t - time.perf_counter()) > 0:
                        time.sleep(d)
                if tap:
                    tap.write(json.dumps(ev) + "\n")
                self.q.put(ev)
        finally:
            if tap:
                tap.close()
            self.q.put(_END)

    def stop(self) -> None:
        self._stop.set()

    def __iter__(self) -> Iterator[dict]:
        self._thread.start()
        while (item := self.q.get()) is not _END:
            yield item


# --------------------------------------------------------------------- schema
_NUM = (int, float)
SCHEMAS: dict[str, dict[str, tuple]] = {
    "session_start": {"session_id": (int,), "viewer_id": (int,), "title_id": (int,),
                      "device": (str,), "region": (str,), "cdn": (str,), "startup_ms": _NUM},
    "heartbeat": {"session_id": (int,), "viewer_id": (int,), "device": (str,), "region": (str,),
                  "cdn": (str,), "bitrate_mbps": _NUM, "throughput_mbps": _NUM, "buffer_s": _NUM,
                  "rebuffer_ms": _NUM, "bitrate_switch": (int,), "dropped_frames": (int,)},
    "session_end": {"session_id": (int,), "viewer_id": (int,), "title_id": (int,),
                    "watch_s": _NUM, "duration_s": _NUM, "end_reason": (str,)},
}
_ALLOWED = {"device": set(DEVICES), "region": set(REGIONS), "cdn": set(CDNS)}
_NON_NEGATIVE = {"bitrate_mbps", "throughput_mbps", "buffer_s", "rebuffer_ms", "startup_ms", "watch_s"}


def validate(ev: dict) -> str | None:
    """Return None if the event is valid, else a short reason string."""
    schema = SCHEMAS.get(ev.get("type"))
    if schema is None:
        return "unknown_type"
    if "ts" not in ev:
        return "missing:ts"
    for name, types in schema.items():
        if name not in ev or ev[name] is None:
            return f"missing:{name}"
        if not isinstance(ev[name], types) or isinstance(ev[name], bool) and bool not in types:
            return f"type:{name}"
        if name in _ALLOWED and ev[name] not in _ALLOWED[name]:
            return f"value:{name}"
        if name in _NON_NEGATIVE and ev[name] < 0:
            return f"range:{name}"
    return None


@dataclass
class DeadLetterQueue:
    max_kept: int = 1000
    reasons: Counter = field(default_factory=Counter)
    samples: list = field(default_factory=list)

    def put(self, ev: dict, reason: str) -> None:
        self.reasons[reason] += 1
        if len(self.samples) < self.max_kept:
            self.samples.append({"reason": reason, "event": ev})

    @property
    def count(self) -> int:
        return sum(self.reasons.values())


# --------------------------------------------------------------- windowing
@dataclass
class MicroBatch:
    window_start: datetime
    window_index: int
    starts: list = field(default_factory=list)
    heartbeats: list = field(default_factory=list)
    ends: list = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.starts) + len(self.heartbeats) + len(self.ends)


class MicroBatcher:
    def __init__(self, dlq: DeadLetterQueue, window_s: int = TICK_SECONDS) -> None:
        self.dlq, self.window_s = dlq, window_s
        self.epoch: float | None = None
        self.late_events = 0

    def batches(self, events: Iterable[dict]) -> Iterator[MicroBatch]:
        current: MicroBatch | None = None
        for ev in events:
            if (reason := validate(ev)) is not None:
                self.dlq.put(ev, reason)
                continue
            t = datetime.fromisoformat(ev["ts"]).timestamp()
            if self.epoch is None:
                self.epoch = t - (t % self.window_s)
            idx = int((t - self.epoch) // self.window_s)
            if current is None:
                current = self._new(idx)
            if idx < current.window_index:  # arrived after its window closed
                self.late_events += 1
                self.dlq.put(ev, "late_event")
                continue
            if idx > current.window_index:  # watermark passed: close window
                yield current
                current = self._new(idx)
            {"session_start": current.starts, "heartbeat": current.heartbeats,
             "session_end": current.ends}[ev["type"]].append(ev)
        if current is not None and current.size:
            yield current

    def _new(self, idx: int) -> MicroBatch:
        return MicroBatch(datetime.fromtimestamp(self.epoch + idx * self.window_s), idx)