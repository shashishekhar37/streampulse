"""Small wrappers that bundle an online scaler with an incremental sklearn model.

They are what the registry saves and what serve.py loads, so a single object
carries everything needed to score raw features: scaler, model, feature names.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler


class OnlineClassifier:
    def __init__(self, feature_names: list[str], positive_weight: float = 5.0,
                 eta0: float = 0.005, seed: int = 0) -> None:
        self.feature_names = list(feature_names)
        self.positive_weight = positive_weight
        self.eta0, self.seed = eta0, seed
        self.scaler = StandardScaler()
        self.clf = self._new()
        self.fitted = False
        self.n_trained = 0

    def _new(self) -> SGDClassifier:
        return SGDClassifier(loss="log_loss", alpha=1e-4, learning_rate="constant",
                             eta0=self.eta0, random_state=self.seed)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.full(len(X), 0.02)
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]

    def learn(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, update_scaler: bool = True) -> None:
        if update_scaler:
            self.scaler.partial_fit(X)
        w = np.where(y == 1, self.positive_weight, 1.0)
        Xs = self.scaler.transform(X)
        for _ in range(epochs):
            self.clf.partial_fit(Xs, y, classes=np.array([0, 1]), sample_weight=w)
        self.fitted = True
        self.n_trained += len(y)

    def boost_learning_rate(self, factor: float) -> None:
        """Temporarily learn faster (used right after drift is detected)."""
        self.clf.set_params(eta0=self.eta0 * factor)

    def coefficients(self) -> dict[str, float]:
        if not self.fitted:
            return {}
        return dict(zip(self.feature_names, self.clf.coef_[0].round(3).tolist()))


class OnlineRegressor:
    def __init__(self, feature_names: list[str], eta0: float = 0.002, seed: int = 0) -> None:
        self.feature_names = list(feature_names)
        self.scaler = StandardScaler()
        self.reg = SGDRegressor(loss="huber", epsilon=0.5, alpha=1e-5, learning_rate="constant",
                                eta0=eta0, random_state=seed)
        self.fitted = False
        self.n_trained = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return X[:, 0]  # fall back to "same as last value"
        return self.reg.predict(self.scaler.transform(X))

    def learn(self, X: np.ndarray, y: np.ndarray) -> None:
        self.scaler.partial_fit(X)
        self.reg.partial_fit(self.scaler.transform(X), y)
        self.fitted = True
        self.n_trained += len(y)

class ResidualForecaster(OnlineRegressor):
    """Forecast log-throughput as  session_mean + learned deviation.

    Why not regress the raw level directly? In a stream, every example in a 30 s window
    shares the same hour and arrives in time order. A constant-learning-rate SGD model
    then lets its intercept and hour features "track" the current level instead of
    learning a real daily pattern: right for the current hour, wrong for any other.
    Predicting the deviation from the session's own average uses only relative
    features (lags minus session mean), which behave the same at every hour.
    """

    RELATIVE = ("lag1", "lag2", "lag3", "ewma", "group_mean_now")
    KEEP = ("lag_diff", "beats_log")

    def __init__(self, feature_names: list[str], eta0: float = 0.002, seed: int = 0) -> None:
        super().__init__(feature_names, eta0, seed)
        self._mean = self.feature_names.index("session_mean")
        self._rel = [self.feature_names.index(n) for n in self.RELATIVE]
        self._keep = [self.feature_names.index(n) for n in self.KEEP]

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([X[:, self._rel] - X[:, [self._mean]], X[:, self._keep]])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return X[:, 0]
        return X[:, self._mean] + self.reg.predict(self.scaler.transform(self._transform(X)))

    def learn(self, X: np.ndarray, y: np.ndarray) -> None:
        R = self._transform(X)
        self.scaler.partial_fit(R)
        self.reg.partial_fit(self.scaler.transform(R), y - X[:, self._mean])
        self.fitted = True
        self.n_trained += len(y)
