# StreamPulse Interview Guide

This guide is for presenting StreamPulse in AI/ML engineer interviews: what to say, which
concepts it shows, the design decisions behind it, and answers to likely questions.

---

## 1. How to describe it (and how not to)

**Describe it as a personal project built on simulated data.** Interviewers for ML roles
ask follow-ups such as "how much data?", "who used it?", "what was the business impact?"
and "how did you get access to the data?". A project presented as production work at an
employer falls apart under those questions, and can fail a background or reference
check. Presented honestly, it's a strong project: it shows you can build the whole
system, measure it properly, and find and fix real ML problems.

If you work in media or telecom, connect it to your domain truthfully: *"In my domain,
video quality and viewer drop-off are big topics, so I built a project to learn how
you'd do real-time ML on playback telemetry."*

### Resume entry (Projects section)

**StreamPulse: Real-Time ML Platform for Video-Streaming Telemetry** · Python, scikit-learn, NumPy, pandas
- Built an end-to-end streaming ML system that ingests simulated playback telemetry (~11K events/s on a laptop) through schema validation, a dead-letter queue and 30-second event-time micro-batches, and runs 5 models online.
- Predicted viewer abandonment with an online logistic regression trained from delayed labels (ROC-AUC 0.88 vs 0.80 for a frozen batch model); detected concept drift with Page-Hinkley and input drift with PSI.
- Detected CDN/region quality incidents without labels using peer-adjusted EWMA z-scores and Isolation Forest (3 of 4 injected incidents caught within 30 s); cut false alerts from 1,174 to 11 with variance floors and debouncing.
- Compared recommenders online (BPR matrix factorisation, content-based, hybrid, popularity): content-based doubled HitRate@10 over popularity (0.27 vs 0.13) under sparse interaction data.
- Clustered viewers into behaviour segments with online MiniBatchKMeans (ARI 0.98 against ground-truth personas); shipped models through a versioned registry and a REST scoring API, with 10 automated tests.

### 60-second pitch

> "I built StreamPulse to learn real-time ML end to end. It simulates a video streaming
> platform: viewers, titles, CDNs, and the 30-second heartbeats a player sends with bitrate,
> throughput and buffering. Events flow through a queue, get validated, and are grouped
> into 30-second windows by event time. On that stream I run five models: abandonment
> prediction, bandwidth forecasting, CDN incident detection, recommendations and viewer
> segmentation. Every model is evaluated prequentially, meaning it predicts before it sees
> the answer, and compared against a baseline. The most interesting parts were the problems
> I found: a label definition that capped accuracy, a forecaster that looked accurate but
> was only tracking the current hour, and an anomaly detector that started at over a
> thousand false alerts. I'll walk through any of them."

---

## 2. ML concepts covered, and where they live

| Concept | Where | One-line explanation |
|---|---|---|
| Synthetic data with ground truth | `simulator.py` | Hidden personas, incidents and drift make honest evaluation possible |
| Streaming ingestion, producer/consumer | `ingest.QueueSource` | Decouples data arrival from processing, like Kafka |
| Data validation, dead-letter queue | `ingest.validate`, `DeadLetterQueue` | Bad events are quarantined, not crashes or silent corruption |
| Event time vs processing time, tumbling windows, watermark | `ingest.MicroBatcher` | Windows use the timestamp inside the event; a window closes when a later one starts |
| Micro-batching | `pipeline.py` | Vectorised prediction per 30 s window (the Spark Structured Streaming idea) |
| Online feature store, training/serving parity | `feature_store.py` | One code path computes features for both learning and the API |
| Stateful features (Welford-style running stats, EWMA, lags) | `feature_store.py` | Features from history without storing history |
| Supervised classification, class imbalance | `tasks/abandonment.py` | 2.2% positives; positive weighting |
| Delayed labels, prequential evaluation | `tasks/abandonment.py` | Score now, learn when the outcome is known |
| Online learning (`partial_fit`, SGD) | `tasks/models.py` | Models update continuously |
| Baselines | every task | Frozen model, rule, persistence, popularity, random |
| Metrics: ROC-AUC, PR-AUC, Brier | `monitoring.RollingWindow` | Ranking vs calibration; PR-AUC suits rare events |
| Regression, time-series lags, Huber loss | `tasks/forecasting.py` | Robust to throughput spikes |
| Target/feature engineering for stationarity | `models.ResidualForecaster` | Predict deviation from the session mean |
| Unsupervised anomaly detection | `tasks/anomaly.py` | Learn "normal", flag deviations |
| Seasonality removal by peer comparison | `tasks/anomaly.py` | Subtract the median of all groups each window |
| Isolation Forest, sliding-window retraining | `tasks/anomaly.py` | Batch model retrained hourly inside a stream |
| Alert debouncing, precision/recall/delay | `tasks/anomaly.py` | Two consecutive windows required |
| Recommender systems, implicit feedback | `tasks/recommender.py` | Play and watch-time signals, no ratings |
| Matrix factorisation, BPR, negative sampling | `tasks/recommender.py` | Pairwise ranking loss |
| Content-based and hybrid recommenders, cold start | `tasks/recommender.py` | Genre affinity works with little history |
| HitRate@K, NDCG@K | `tasks/recommender.py` | Ranking metrics |
| Clustering, feature scaling, ARI, silhouette | `tasks/segmentation.py` | Online KMeans with external and internal metrics |
| Concept drift detection (Page-Hinkley) | `monitoring.py` | Loss trend test |
| Data drift (PSI) | `monitoring.py` | Distribution shift of inputs |
| Model registry, versioning, rollback | `registry.py` | LATEST pointer and metadata |
| Model serving, input validation, imputation | `serve.py` | REST API with error handling |
| Testing ML systems | `tests/` | Unit, property and end-to-end tests |

---

## 3. One 30-second window, step by step

1. **Ingest.** The simulator yields events. A producer thread puts them in a queue; the
   consumer validates each one. Bad events (missing fields, negative throughput, unknown
   CDN) go to the DLQ with a reason.
2. **Window.** Events are grouped by event-time window. When an event from a later window
   arrives, the current window is closed and processed.
3. **Session starts.** Each is added to the feature store. The recommender first checks
   whether the chosen title was in its top 10 for that viewer, then learns from it.
4. **Heartbeats.** Session state is updated: stall time, switches, throughput lags.
   Aggregates per CDN × region are computed for the window.
5. **Abandonment.** Every heartbeat is scored now, and the prediction is parked until its
   label is known.
6. **Forecast.** Last window's forecasts are compared to this window's actual throughput,
   the model learns, and the next window is forecast.
7. **Anomaly.** Each CDN × region aggregate is compared to its peers and its own history.
   Two anomalous windows in a row raise an alert.
8. **Session ends.** Pending abandonment predictions get their labels; viewer profiles and
   recommender feedback are updated.
9. **Timeouts.** Predictions whose viewer is still watching become label 0. The detector
   gets this window's mean loss.
10. **Housekeeping.** Clustering refreshes every 2 hours, metrics are logged hourly, and
    models are published to the registry every 6 hours.

---

## 4. Design decisions and lessons

These came up during development. Each makes a good "tell me about a problem you
solved" story (situation → what you saw → what you did → result).

**a) The label definition capped accuracy.** The first label was "abandons within the next
90 seconds". Heartbeats *before* a stall were then labelled positive even though nothing
in them predicted the stall. Offline, even a logistic regression fitted in-sample topped
out near 0.25 PR-AUC. Shortening the horizon to about 60 s made the label match what the
features can see. *Lesson: check the label's ceiling before tuning the model.*

**b) A forecaster that only looked accurate.** The bandwidth model's error was fine in the
stream, but the served model predicted 3.8 Mbps for a viewer at 20 Mbps. Every example in
a window shares the same hour and windows arrive in time order. A constant-learning-rate
SGD therefore let its intercept and hour weights track the current level instead of
learning a daily pattern: right for the current hour, wrong for others. The fix was to
predict the deviation from the session's own mean using relative features. Stream accuracy
stayed the same (MAE 4.03) and the served prediction became 19.9 Mbps. *Lesson: online
evaluation can hide non-stationary shortcuts. Test the served artifact on held-out
conditions.*

**c) 1,174 false alerts to 11.** Quiet CDN/region groups almost never rebuffer, so their
variance was near zero and one stalled viewer made a huge z-score. The fixes: a floor on
the standard deviation, at least 3 active streams per group, and two consecutive windows
before alerting. Peer adjustment (subtracting the median across groups) stops evening
congestion from looking like an incident.

**d) The drift response made things worse.** After Page-Hinkley flags drift, boosting the
learning rate seemed sensible. Measured PR-AUC: no boost 0.281, 2× 0.275, 3× 0.266,
5× 0.258. Continuous learning already adapts, so the boost only added noise. Drift
detection now raises an alert by default, and the boost is optional. *Lesson: measure the
"obvious" improvement.*

**e) Collaborative filtering lost to simple content features.** BPR matrix factorisation
reached HitRate@10 of only 0.15–0.17. With about 7 sessions per viewer there isn't enough
interaction data. A content-based model (genre affinity × popularity) reached 0.268, close
to the 0.283 that a model knowing the simulator's true preferences would get. *Lesson:
match the model to the data volume; hybrids and content features address sparsity and
cold start.*

**f) Clustering needed scaling.** Unscaled profiles let average watch time dominate the
distance, and personas merged (ARI 0.46). Standardising features, together with more
history per viewer, raised ARI to 0.98.

**g) Ranking and calibration differ.** Weighting positives 5× helps ranking (ROC/PR-AUC)
but inflates probabilities: online Brier is 0.027 against 0.021 for the unweighted rule.
If a business rule uses "probability > 0.3", calibrate first (Platt scaling or isotonic
regression on recent labelled data).

**h) PSI needs a seasonal reference.** PSI on throughput goes from 0.003 to 0.48 within a
day because the reference was night-time traffic. Comparing to the same hour yesterday
would isolate real drift from the daily cycle. PSI is also uninformative for zero-inflated
features such as stall time.

---

## 5. Likely interview questions

**Why online learning instead of retraining nightly?** Behaviour changes within hours (new
releases, live events, outages). Online learning adapts continuously and costs little per
update. The frozen model shows the gap: after the drift it averages 0.789 ROC-AUC against
0.845 online. In practice many teams combine both: periodic full retrains plus online
updates or frequent fine-tuning.

**What is prequential evaluation?** Test-then-train: each example is predicted *before*
the model learns from it, so every metric is an honest out-of-sample estimate on data in
time order. No random train/test split, so no leakage from the future.

**How do you handle delayed labels?** Predictions wait in a pending buffer keyed by
session. A label resolves when the session ends (abandoned or not) or when the horizon
passes. Only then do evaluation and learning happen. In production the delay can be days
(fraud chargebacks, churn), and the same buffer pattern applies with a store such as
Redis or a feature table.

**Why PR-AUC and not accuracy?** With 2.2% positives, predicting "never quits" scores
97.8% accuracy and catches nothing. PR-AUC measures how well positives are ranked; its
random baseline equals the positive rate (0.022), so 0.281 is about 13× random.

**How did you handle class imbalance?** Sample weights on positives (5×). Alternatives:
undersampling negatives, focal loss, threshold tuning. Weighting distorts probabilities,
so calibrate afterwards if probabilities are used directly.

**Concept drift vs data drift?** Data drift: P(X) changes, for example throughput
distributions at peak time (caught by PSI). Concept drift: P(y|X) changes, meaning the same
features lead to a different outcome (viewers start quitting on quality drops). Concept
drift is detected by monitoring model loss (Page-Hinkley) once labels arrive.

**Why event time instead of arrival time?** Network delays and retries reorder events.
Windows by event time give correct aggregates. Late events past the watermark are routed
to the DLQ here; production systems use allowed lateness and window updates.

**How do you avoid training/serving skew?** One `FeatureStore` computes features for
training and serving, and the model object carries its feature names and scaler. The API
validates names, rejects unknown ones and reports imputed ones.

**Why Isolation Forest and a z-score?** The z-score is interpretable and fast but assumes
roughly normal residuals per metric. Isolation Forest catches unusual *combinations* of
metrics without distribution assumptions, but it's harder to explain and needs periodic
retraining. Here IForest had fewer false alerts (7 vs 11) but lower window recall.

**How do you evaluate an unsupervised detector?** Inject known incidents, then measure
incident-level recall, detection delay and false alerts per day. These are the numbers an
on-call team cares about, beyond window-level precision and recall.

**Explain BPR.** For a viewer u, a watched item i and a sampled unwatched item j, maximise
log σ(score(u,i) − score(u,j)). It optimises ranking directly, which suits implicit
feedback where missing doesn't mean disliked.

**How do you solve cold start?** Popularity for brand-new users, content features (genre
affinity) after a few plays, and collaborative signals once history accumulates. A hybrid
blends them. New titles need content embeddings or exploration (bandits).

**How would you productionise this?** Kafka for ingestion; Flink or Spark Structured
Streaming for windows and features; Redis or Feast as the online feature store; MLflow as
the registry; FastAPI or KServe for serving; Prometheus and Grafana for metrics and alerts;
shadow deployment, then an A/B test on a business metric (fewer abandoned sessions, more
watch time) before full rollout.

**How would you scale it?** Partition the stream by session ID so each worker owns its
sessions' state; aggregate CDN × region counts with a keyed shuffle; serve models
statelessly behind a load balancer; keep feature state in a distributed key-value store.

**What would you do next?** Calibrate the abandonment model; try gradient-boosted trees
with periodic retraining; add a "traffic vanished" signal for incidents; seasonal PSI
references; contextual bandits for recommendations to balance exploration.

---

## 6. Running a live demo

1. `python run.py --hours 12`: point out the live hourly table (online vs frozen, alerts, DLQ).
2. Open `outputs/dashboard.png`: drift at hour 24, incidents as red bands, recommender comparison.
3. `python serve.py --selftest`: a smooth session scores low risk; a stalling one scores 0.99.
4. Open `streampulse/pipeline.py` and walk through the 8 numbered steps.
5. Tell one lesson from section 4, usually (b) or (c).
