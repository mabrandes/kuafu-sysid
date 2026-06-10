"""Persistence baselines: predict horizon h from a lagged endog value.

period_steps controls which baseline:
  1                    -> prev_step (repeat last observed value)
  steps_per_day        -> prev_day (same slot yesterday)
  steps_per_week       -> prev_week (same slot last week)

For horizon h (1-indexed) the forecast for t+h is the endog value at
t + h - period == endog.shift(max(period - h, 0)) evaluated at t. The
forecast reads the endog series directly (passed to predict), so it works at
any configured lag (no need to inflate lag to cover a weekly period).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from kuafu_sysid.models.base import Forecaster


class Persistence(Forecaster):
    EXT = "joblib"
    requires_fit = False

    def __init__(self, period_steps: int, horizon: int, endog: str, **_ignored):
        self.period_steps = int(period_steps)
        self.horizon = int(horizon)
        self.endog = endog

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "Persistence":
        """No-op. X: (n_samples, n_features), Y: (n_samples, horizon H) — accepted
        for a uniform interface but unused; a persistence baseline learns nothing."""
        return self  # nothing to learn

    def predict(self, X: pd.DataFrame, endog: pd.Series | None = None) -> np.ndarray:
        """Predict all horizons from the endog series alone. X: (n_samples, …) — only
        its index/length are used; ``endog`` is the full target series (required).
        Returns (len(X), horizon H): step h = endog shifted by ``period_steps``."""
        if endog is None:
            raise ValueError(
                "Persistence baselines need the endog series; call predict(X, endog=...)."
            )
        cols = []
        for h in range(1, self.horizon + 1):
            k = max(self.period_steps - h, 0)  # h > period -> repeat the origin value
            cols.append(endog.shift(k).reindex(X.index).to_numpy(float))
        return np.column_stack(cols)

    def save(self, path: Path) -> None:
        joblib.dump({"period_steps": self.period_steps, "horizon": self.horizon, "endog": self.endog}, path)

    @classmethod
    def load(cls, path: Path) -> "Persistence":
        b = joblib.load(path)
        return cls(period_steps=b["period_steps"], horizon=b["horizon"], endog=b["endog"])
