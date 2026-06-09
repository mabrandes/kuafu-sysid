"""Persistence baselines: predict horizon h from a lagged endog value.

period_steps controls which baseline:
  1                    -> prev_step (repeat last observed value)
  steps_per_day        -> prev_day (same slot yesterday)
  steps_per_week       -> prev_week (same slot last week)

For horizon h (1-indexed) the forecast for t+h is the endog value at
t + h - period, i.e. column {endog}_lag_{period - h} when available; for
(period - h) < 0 or a missing column it falls back to {endog}_lag_0.
Requires the endog lag columns to be present in X (config lag >= period - 1).
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
        return self  # nothing to learn

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for h in range(1, self.horizon + 1):
            k = self.period_steps - h
            name = f"{self.endog}_lag_{k}"
            if k < 0 or name not in X.columns:
                name = f"{self.endog}_lag_0"
            cols.append(X[name].to_numpy(float))
        return np.column_stack(cols)

    def save(self, path: Path) -> None:
        joblib.dump({"period_steps": self.period_steps, "horizon": self.horizon, "endog": self.endog}, path)

    @classmethod
    def load(cls, path: Path) -> "Persistence":
        b = joblib.load(path)
        return cls(period_steps=b["period_steps"], horizon=b["horizon"], endog=b["endog"])
