"""Task 5: viewer segmentation (unsupervised clustering, updated online).

Every 2 simulated hours, each viewer with 3+ sessions gets a behaviour profile
(genre mix, time-of-day mix, average watch length, completion and abandonment rate,
share of live content). MiniBatchKMeans.partial_fit updates 4 cluster centres
incrementally, so segments evolve as behaviour data accumulates.

Evaluation uses the simulator's hidden personas, which a real system would not have:
    adjusted Rand index (ARI): agreement between clusters and personas (1 = perfect, 0 = random)
    silhouette score         : how well separated the clusters are, needs no labels
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ..config import GENRES
from ..feature_store import FeatureStore

HOUR_BUCKETS = ["night", "morning", "afternoon", "evening"]


class SegmentationTask:
    name = "segmentation"

    def __init__(self, n_clusters: int = 4, every_ticks: int = 240, seed: int = 0) -> None:
        self.km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=256, n_init=3)
        self.every = every_ticks
        self.scaler = StandardScaler()  # features have different units; scale before distances
        self.personas: dict[int, str] = {}
        self.fitted = False

    def on_session_start(self, ev: dict) -> None:
        if "_persona" in ev:
            self.personas[ev["viewer_id"]] = ev["_persona"]

    def _profiles(self, fs: FeatureStore) -> tuple[list[int], np.ndarray]:
        ids, rows = [], []
        for vid in list(fs.viewers):
            p = fs.viewer_profile(vid)
            if p is not None:
                ids.append(vid)
                rows.append(p)
        return ids, (np.vstack(rows) if rows else np.empty((0, 0)))

    def maybe_update(self, tick: int, fs: FeatureStore) -> None:
        if tick == 0 or tick % self.every:
            return
        ids, X = self._profiles(fs)
        if len(ids) < 50:
            return
        self.scaler.partial_fit(X)
        Xs = self.scaler.transform(X)
        for _ in range(3):
            self.km.partial_fit(Xs)
        self.fitted = True

    def summary(self, fs: FeatureStore) -> dict:
        ids, X = self._profiles(fs)
        if not self.fitted or len(ids) < 50:
            return {}
        Xs = self.scaler.transform(X)
        labels = self.km.predict(Xs)
        truth = [self.personas.get(v, "unknown") for v in ids]
        segments = []
        for c in range(self.km.n_clusters):
            m = labels == c
            if not m.any():
                continue
            prof = X[m].mean(axis=0)
            ng = len(GENRES)
            top_genres = [GENRES[i] for i in np.argsort(-prof[:ng])[:2]]
            when = HOUR_BUCKETS[int(np.argmax(prof[ng:ng + 4]))]
            personas, counts = np.unique(np.array(truth)[m], return_counts=True)
            segments.append({
                "segment": c, "viewers": int(m.sum()),
                "description": f"{' + '.join(top_genres)}, mostly {when}",
                "avg_watch_min": round(float(np.expm1(prof[ng + 4])), 1),
                "completion_rate": round(float(prof[ng + 5]), 3),
                "abandon_rate": round(float(prof[ng + 6]), 3),
                "dominant_true_persona": str(personas[np.argmax(counts)]),
                "purity": round(float(counts.max() / m.sum()), 3),
            })
        return {"viewers_clustered": len(ids),
                "adjusted_rand_index": round(float(adjusted_rand_score(truth, labels)), 3),
                "silhouette": round(float(silhouette_score(Xs, labels, sample_size=min(2000, len(X)),
                                                           random_state=0)), 3),
                "segments": segments}