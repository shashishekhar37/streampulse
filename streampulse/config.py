"""Shared constants for the simulated streaming platform."""

from __future__ import annotations

TICK_SECONDS = 30  # each active session sends one heartbeat every 30 s

REGIONS = ["northeast", "southeast", "midwest", "mountain", "west"]
DEVICES = ["smart_tv", "set_top_box", "mobile", "web", "tablet"]
CDNS = ["cdn_a", "cdn_b", "cdn_c"]
GENRES = ["drama", "comedy", "sports", "news", "kids", "documentary", "action", "reality"]
BITRATE_LADDER = [0.8, 1.6, 3.0, 5.0, 8.0, 12.0]  # Mbps, adaptive-bitrate (ABR) rungs

# Behavioural personas (hidden ground truth used to test the clustering task).
#   genre_pref : relative taste for each genre in GENRES order
#   devices    : probability of each device in DEVICES order
#   hours      : relative activity for each hour of day 0..23
#   patience   : tolerance to buffering (higher = abandons less)
#   watch_frac : (alpha, beta) of the Beta distribution for how much of a title they intend to watch
PERSONAS = {
    "binge_watcher": dict(
        genre_pref=[5, 2, 0.5, 0.3, 0.3, 1.5, 4, 3],
        devices=[0.45, 0.25, 0.1, 0.15, 0.05],
        hours=[3, 2, 1, .5, .2, .2, .2, .3, .5, .5, .6, .7, .8, .8, .8, .9, 1, 1.5, 2.5, 3.5, 4, 4.5, 4.5, 4],
        patience=1.3, watch_frac=(5, 1.5)),
    "sports_fan": dict(
        genre_pref=[0.5, 0.8, 8, 1.5, 0.2, 0.5, 1.5, 0.3],
        devices=[0.4, 0.3, 0.2, 0.05, 0.05],
        hours=[.5, .3, .2, .1, .1, .2, .4, .6, .6, .6, .8, 1, 1.5, 1.5, 1.5, 1.5, 2, 2.5, 3.5, 4.5, 5, 4.5, 3, 1.5],
        patience=0.6, watch_frac=(4, 2)),
    "family": dict(
        genre_pref=[0.8, 3, 0.5, 0.3, 6, 1.5, 0.8, 1],
        devices=[0.5, 0.15, 0.05, 0.05, 0.25],
        hours=[.1, .1, .1, .1, .1, .3, 1, 2, 2, 1.5, 1.5, 1.5, 1.5, 1.5, 2, 3, 3.5, 3.5, 3.5, 3, 2, 1, .4, .2],
        patience=1.0, watch_frac=(3, 2)),
    "news_casual": dict(
        genre_pref=[0.5, 1, 1, 7, 0.2, 3, 0.5, 0.5],
        devices=[0.1, 0.1, 0.45, 0.3, 0.05],
        hours=[.2, .1, .1, .1, .3, 1.5, 4, 5, 4, 2.5, 1.5, 1.5, 2.5, 1.5, 1, 1, 1.2, 2, 2.5, 2, 1.5, 1, .6, .3],
        patience=0.8, watch_frac=(1.5, 3)),
}
PERSONA_SHARE = {"binge_watcher": 0.3, "sports_fan": 0.25, "family": 0.25, "news_casual": 0.2}

# Injected CDN incidents: (sim hour from start, duration minutes, cdn, region).
# These are the ground truth for the anomaly-detection task.
INCIDENTS = [
    (13.0, 20, "cdn_a", "northeast"),
    (20.0, 30, "cdn_b", "west"),
    (33.5, 25, "cdn_b", "southeast"),
    (45.0, 40, "cdn_c", "midwest"),
]

# Concept drift: from this sim hour on, viewers become less tolerant of buffering
# (e.g. after a competitor launch) and evening network congestion gets worse.
DRIFT_HOUR = 24.0
