"""Dashboard image summarising a run (outputs/dashboard.png)."""

from __future__ import annotations

from pathlib import Path

from .config import DRIFT_HOUR, INCIDENTS


def make_dashboard(result, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = result.hourly
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    blue, orange, grey, red = "#1f6f8b", "#d98c1f", "#8a8f98", "#c2383d"

    a = ax[0, 0]
    for col, label, style in [("abandon_online_pr_auc", "online model", dict(color=blue, lw=2.2)),
                              ("abandon_frozen_pr_auc", "frozen (trained hours 0-12)", dict(color=orange, lw=2, ls="--")),
                              ("abandon_heuristic_pr_auc", "rule: stall time", dict(color=grey, lw=1.5, ls=":"))]:
        if col in h:
            a.plot(h["hour"], h[col], label=label, **style)
    a.axvline(DRIFT_HOUR, color=red, alpha=0.5, label="concept drift starts")
    a.set_title("1. Abandonment prediction: rolling PR-AUC")
    a.set_xlabel("simulated hour")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[0, 1]
    for col, label, style in [("forecast_mae_model", "online regression", dict(color=blue, lw=2.2)),
                              ("forecast_mae_persistence", "persistence (last value)", dict(color=orange, lw=2, ls="--")),
                              ("forecast_mae_ewma", "EWMA", dict(color=grey, lw=1.5, ls=":"))]:
        if col in h:
            a.plot(h["hour"], h[col], label=label, **style)
    a.set_title("2. Throughput forecast: rolling MAE (Mbps, lower is better)")
    a.set_xlabel("simulated hour")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1, 0]
    rec = result.anomaly_records
    if len(rec):
        for i, (start_h, dur, cdn, region) in enumerate(INCIDENTS):
            sub = rec[(rec.cdn == cdn) & (rec.region == region)]
            a.plot(sub.tick / 120, sub.zscore.clip(upper=30), lw=0.8, label=f"{cdn}/{region}")
            a.axvspan(start_h, start_h + dur / 60, color=red, alpha=0.15, label="injected incident" if i == 0 else None)
        a.axhline(4, color="black", ls="--", lw=1, label="alert threshold (z = 4)")
    a.set_title("3. CDN/region anomaly score (z) for incident groups")
    a.set_xlabel("simulated hour")
    a.legend(fontsize=7, ncol=2)
    a.grid(alpha=0.3)

    a = ax[1, 1]
    for col, label, style in [("rec_hit@10_hybrid", "hybrid (content + MF)", dict(color=blue, lw=2.4)),
                              ("rec_hit@10_content", "content-based", dict(color="#4a9a5b", lw=1.8)),
                              ("rec_hit@10_mf", "matrix factorisation (BPR)", dict(color="#7b5ea7", lw=1.8)),
                              ("rec_hit@10_popularity", "most popular", dict(color=orange, lw=2, ls="--")),
                              ("rec_hit@10_random", "random", dict(color=grey, lw=1.5, ls=":"))]:
        if col in h:
            a.plot(h["hour"], h[col], label=label, **style)
    a.set_title("4. Recommendations: rolling HitRate@10")
    a.set_xlabel("simulated hour")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    fig.suptitle("StreamPulse: real-time ML on streaming-media telemetry", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)