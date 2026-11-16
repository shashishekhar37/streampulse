"""Run StreamPulse end to end on simulated streaming-media telemetry.

Examples
--------
    python run.py                      # 48 simulated hours, as fast as possible
    python run.py --hours 12           # quicker run
    python run.py --rate 2000          # replay at 2,000 events/sec, like a live feed
    python run.py --tap outputs/telemetry.jsonl   # also save the raw event stream
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from streampulse.ingest import QueueSource
from streampulse.pipeline import StreamPulsePipeline
from streampulse.report import make_dashboard
from streampulse.simulator import StreamingPlatformSimulator

OUT = Path(__file__).parent / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=48, help="simulated hours of telemetry")
    ap.add_argument("--traffic", type=float, default=0.6, help="traffic multiplier (1.0 ~ 150 concurrent streams)")
    ap.add_argument("--rate", type=float, default=0.0, help="events/sec replay rate (0 = max speed)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-drift-detection", action="store_true")
    ap.add_argument("--drift-boost", type=float, default=1.0,
                    help="learning-rate multiplier for 2 h after a drift alarm (1 = alert only)")
    ap.add_argument("--tap", help="write every raw event to this JSONL file")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    shutil.rmtree(OUT / "registry", ignore_errors=True)  # fresh registry for each run

    sim = StreamingPlatformSimulator(hours=args.hours, traffic=args.traffic, seed=args.seed)
    source = QueueSource(sim, rate=args.rate, tap_file=args.tap)
    pipeline = StreamPulsePipeline(registry_dir=OUT / "registry",
                                   drift_detection=not args.no_drift_detection,
                                   drift_boost=args.drift_boost)
    try:
        result = pipeline.run(source)
    except KeyboardInterrupt:
        source.stop()
        print("\nstopped by user")
        return

    result.hourly.to_csv(OUT / "hourly_metrics.csv", index=False)
    result.anomaly_records.to_csv(OUT / "anomaly_scores.csv", index=False)
    (OUT / "alerts.json").write_text(json.dumps(result.alerts, indent=2))
    (OUT / "summary.json").write_text(json.dumps(result.summary, indent=2, default=float))
    (OUT / "dead_letter_sample.json").write_text(json.dumps(pipeline.dlq.samples[:50], indent=2, default=str))
    make_dashboard(result, OUT / "dashboard.png")

    s = result.summary
    ab, fc, rc, an, sg = s["abandonment"], s["forecast"], s["recommender"], s["anomaly"], s["segmentation"]
    print("\n================ StreamPulse summary ================")
    t = s["throughput"]
    print(f"Ingestion     : {t['events']:,} events, {t['sessions']:,} sessions in {t['seconds']}s "
          f"({t['events_per_s']:,} events/s); {s['data_quality']['dead_letter_events']} bad events -> DLQ")
    print(f"Abandonment   : PR-AUC online {ab.get('abandon_online_pr_auc', 0):.3f} | frozen "
          f"{ab.get('abandon_frozen_pr_auc', 0):.3f} | rule {ab.get('abandon_heuristic_pr_auc', 0):.3f} "
          f"(ROC-AUC online {ab.get('abandon_online_roc_auc', 0):.3f}); drift detected at hours {ab['drift_detected_at_hour']}")
    print(f"Forecasting   : MAE {fc['forecast_mae_model']:.2f} Mbps vs persistence {fc['forecast_mae_persistence']:.2f} "
          f"vs EWMA {fc['forecast_mae_ewma']:.2f}")
    for det, m in an.items():
        delay = m['mean_detection_delay_min']
        print(f"Anomaly {det:<8}: {m['incidents_detected']} incidents found, "
              f"avg delay {delay if delay is None else round(delay, 1)} min, {m['false_alerts']} false alerts, "
              f"window precision {m['precision']:.2f} / recall {m['recall']:.2f}")
    print("Recommender   : HitRate@10 " + " | ".join(f"{m} {rc[f'rec_hit@10_{m}']:.3f}" for m in
                                                     ("hybrid", "content", "mf", "popularity", "random"))
          + f"\n                cold-start HitRate@10 (first session of a viewer): {rc['cold_start_hit@10']}")
    if sg:
        print(f"Segmentation  : {sg['viewers_clustered']} viewers, ARI {sg['adjusted_rand_index']}, "
              f"silhouette {sg['silhouette']}")
        for seg in sg["segments"]:
            print(f"   segment {seg['segment']}: {seg['viewers']:>4} viewers | {seg['description']:<32} "
                  f"| true persona {seg['dominant_true_persona']} ({seg['purity']:.0%})")
    print(f"\nOutputs in {OUT}/: dashboard.png, hourly_metrics.csv, summary.json, alerts.json, registry/")


if __name__ == "__main__":
    main()