"""StreamPulse: real-time machine learning on video-streaming telemetry.

The package simulates a video streaming platform (viewers, titles, CDNs, playback
heartbeats) and runs five ML systems on the live event stream:

    tasks.abandonment  - supervised classification with delayed labels
    tasks.forecasting  - online regression (next-interval bandwidth)
    tasks.anomaly      - unsupervised anomaly detection on CDN/region QoE
    tasks.recommender  - online matrix factorisation (BPR) for "what to watch"
    tasks.segmentation - online clustering of viewers into behaviour segments
"""

__version__ = "1.0.0"
