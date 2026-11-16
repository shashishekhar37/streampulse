"""Task 4: "what to watch next" recommendations, four models compared online.

Implicit feedback: we never see star ratings, only behaviour. Pressing play on a title
is a positive signal; watching most of it is a stronger one.

Models
    popularity : most-played titles, with time decay so trends can change
    mf         : collaborative filtering by matrix factorisation trained with BPR
                 (Bayesian Personalised Ranking). Each viewer u and title i get an
                 8-number embedding; score = p_u . q_i + b_i. For a watched title i and
                 a random title j, nudge embeddings so score(u, i) > score(u, j).
                 Each update also replays a few of the viewer's past titles.
    content    : content-based. Score = log(popularity) + 2 * log(viewer's affinity for
                 the title's genre), where affinity comes from the genres they watched.
    hybrid     : content score + MF score. Content handles new and light viewers,
                 MF adds what genre alone can't capture.

Evaluation is prequential: when a session starts we first ask "was the chosen title in
each model's top 10?" (HitRate@10, NDCG@10), and only then learn from it.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from ..config import GENRES

MODELS = ("popularity", "mf", "content", "hybrid")


class RecommenderTask:
    name = "recommender"

    def __init__(self, n_viewers: int = 1000, n_titles: int = 400, k: int = 8, lr: float = 0.05,
                 reg: float = 0.01, n_neg: int = 8, replay: int = 5, top_k: int = 10, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.P = self.rng.normal(0, 0.1, (n_viewers, k))
        self.Q = self.rng.normal(0, 0.1, (n_titles, k))
        self.b = np.zeros(n_titles)
        self.pop = np.zeros(n_titles)
        self.genre_counts = np.zeros((n_viewers, len(GENRES)))
        self.title_genre = np.full(n_titles, -1)
        self.history: dict[int, list[int]] = defaultdict(list)
        self.lr, self.reg, self.n_neg, self.replay, self.top_k = lr, reg, n_neg, replay, top_k
        self.n_titles = n_titles
        self.hits = {m: deque(maxlen=3000) for m in MODELS}
        self.ndcg = {m: deque(maxlen=3000) for m in MODELS}
        self.cold = {m: deque(maxlen=3000) for m in MODELS}  # first session of a viewer

    # ------------------------------------------------------------ scoring
    def _content_scores(self, u: int) -> np.ndarray:
        aff = (self.genre_counts[u] + 1) / (self.genre_counts[u].sum() + len(GENRES))
        g = np.where(self.title_genre >= 0, self.title_genre, 0)
        known = self.title_genre >= 0
        return np.log1p(self.pop) + 2 * np.where(known, np.log(aff[g]), np.log(1 / len(GENRES)))

    def scores(self, u: int) -> dict[str, np.ndarray]:
        mf = self.Q @ self.P[u] + self.b
        content = self._content_scores(u)
        return {"popularity": self.pop, "mf": mf, "content": content,
                "hybrid": content + (self.Q @ self.P[u])}

    def recommend(self, u: int, model: str = "hybrid") -> np.ndarray:
        s = self.scores(u)[model]
        top = np.argpartition(-s, self.top_k)[: self.top_k]
        return top[np.argsort(-s[top])]

    # ------------------------------------------------------------- events
    def on_session_start(self, ev: dict) -> None:
        u, i = ev["viewer_id"], ev["title_id"]
        if "genre" in ev:
            self.title_genre[i] = GENRES.index(ev["genre"])
        first = not self.history[u]
        for name, s in self.scores(u).items():  # evaluate BEFORE learning
            top = np.argpartition(-s, self.top_k)[: self.top_k]
            ranked = top[np.argsort(-s[top])]
            pos = np.flatnonzero(ranked == i)
            hit = 1.0 if len(pos) else 0.0
            self.hits[name].append(hit)
            self.ndcg[name].append(1.0 / np.log2(pos[0] + 2) if len(pos) else 0.0)
            if first:
                self.cold[name].append(hit)
        self._bpr_update(u, i, 1.0)  # then learn
        for j in self.rng.choice(self.history[u], size=min(self.replay, len(self.history[u])), replace=False) \
                if self.history[u] else []:
            self._bpr_update(u, int(j), 0.5)
        self.history[u].append(i)
        self.pop *= 0.9995
        self.pop[i] += 1
        if self.title_genre[i] >= 0:
            self.genre_counts[u, self.title_genre[i]] += 1

    def on_session_end(self, ev: dict) -> None:
        frac = ev["watch_s"] / max(ev["duration_s"], 1)
        if frac >= 0.5:  # watched most of it: stronger positive signal
            self._bpr_update(ev["viewer_id"], ev["title_id"], min(frac, 1.0))

    def _bpr_update(self, u: int, i: int, weight: float) -> None:
        for j in self.rng.integers(self.n_titles, size=self.n_neg):
            if j == i:
                continue
            pu, qi, qj = self.P[u].copy(), self.Q[i].copy(), self.Q[j].copy()
            x = pu @ (qi - qj) + self.b[i] - self.b[j]
            g = weight / (1 + np.exp(x))  # gradient of log-sigmoid(x)
            self.P[u] += self.lr * (g * (qi - qj) - self.reg * pu)
            self.Q[i] += self.lr * (g * pu - self.reg * qi)
            self.Q[j] += self.lr * (-g * pu - self.reg * qj)
            self.b[i] += self.lr * (g - self.reg * self.b[i])
            self.b[j] += self.lr * (-g - self.reg * self.b[j])

    # ---------------------------------------------------------- reporting
    def report(self) -> dict:
        out = {}
        for m in MODELS:
            if self.hits[m]:
                out[f"rec_hit@10_{m}"] = float(np.mean(self.hits[m]))
                out[f"rec_ndcg@10_{m}"] = float(np.mean(self.ndcg[m]))
        out["rec_hit@10_random"] = self.top_k / self.n_titles
        return out

    def summary(self) -> dict:
        return {"cold_start_hit@10": {m: round(float(np.mean(self.cold[m])), 3) for m in MODELS if self.cold[m]},
                "viewers_seen": len(self.history)}