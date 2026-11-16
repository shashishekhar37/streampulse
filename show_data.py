"""Quick script to display sample raw events and feature vectors from StreamPulse."""

import json
import numpy as np
from streampulse.simulator import StreamingPlatformSimulator
from streampulse.ingest import QueueSource, MicroBatcher, DeadLetterQueue
from streampulse.feature_store import FeatureStore

sim = StreamingPlatformSimulator(hours=0.5, traffic=1.0, seed=7)
source = QueueSource(sim, rate=0.0)
dlq = DeadLetterQueue()
fs = FeatureStore()

events = list(source)
print(f"Total events generated: {len(events)}\n")

# --- Show one example of each event type ---
seen = set()
for ev in events:
    et = ev.get("type")
    if et not in seen:
        seen.add(et)
        print(f"=== Sample {et} event ===")
        # Remove internal _persona if present for cleaner display
        clean = {k: v for k, v in ev.items() if not k.startswith("_")}
        print(json.dumps(clean, indent=2, default=str))
        print()
    if len(seen) == 3:
        break

# --- Show feature vectors ---
batches = list(MicroBatcher(dlq).batches(source2 := QueueSource(StreamingPlatformSimulator(hours=0.25, traffic=1.0, seed=7), rate=0.0)))

# Process a few batches to build up features
for i, batch in enumerate(batches[:5]):
    for ev in batch.starts:
        fs.on_start(ev)
    for ev in batch.heartbeats:
        s = fs.on_heartbeat(ev)
    for ev in batch.ends:
        fs.on_end(ev)

# Show abandonment features from the first heartbeat
source3 = QueueSource(StreamingPlatformSimulator(hours=0.25, traffic=1.0, seed=7), rate=0.0)
fs2 = FeatureStore()
for batch in list(MicroBatcher(DeadLetterQueue()).batches(source3))[:3]:
    for ev in batch.starts:
        fs2.on_start(ev)
    for ev in batch.heartbeats:
        s = fs2.on_heartbeat(ev)
        if s is not None:
            ab_feats = fs2.abandonment_features(ev, s)
            fc_feats = fs2.forecast_features(ev, s)
            print("=== Abandonment Features (1 heartbeat) ===")
            print(f"Feature names: {FeatureStore.ABANDON_FEATURES}")
            print(f"Values:        {np.round(ab_feats, 4)}")
            print()
            print("=== Forecast Features (1 heartbeat) ===")
            print(f"Feature names: {FeatureStore.FORECAST_FEATURES}")
            print(f"Values:        {np.round(fc_feats, 4) if fc_feats is not None else None}")
            print()
            break
    else:
        continue
    break

# Show viewer profile for segmentation
source4 = QueueSource(StreamingPlatformSimulator(hours=2.0, traffic=1.0, seed=7), rate=0.0)
fs3 = FeatureStore()
for batch in MicroBatcher(DeadLetterQueue()).batches(source4):
    for ev in batch.starts:
        fs3.on_start(ev)
    for ev in batch.heartbeats:
        fs3.on_heartbeat(ev)
    for ev in batch.ends:
        fs3.on_end(ev)

# Find a viewer with a profile
for vid in fs3.viewers:
    vp = fs3.viewer_profile(vid)
    if vp is not None:
        print("=== Viewer Segmentation Profile (1 viewer) ===")
        print(f"Viewer ID: {vid}")
        print(f"Profile vector: {np.round(vp, 4)}")
        seg_features = ["genre_" + g for g in ["sports", "news", "kids", "drama", "action", "documentary", "comedy", "reality"]]
        seg_features += ["night", "morning", "afternoon", "evening"]
        seg_features += ["log_watch_s", "completion_rate", "abandon_rate"]
        print(f"Feature names: {seg_features}")
        print()
        break
