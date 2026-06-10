"""LightGBM adapter (NaN-native). One regressor per horizon; optional quantile."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from kuafu_sysid.models.base import Forecaster


class Lgbm(Forecaster):
    EXT = "joblib"

    def __init__(self, n_estimators: int = 400, num_leaves: int = 31,
                 learning_rate: float = 0.05, quantile: float | None = None, **_ignored):
        objective = "quantile" if quantile is not None else "regression"
        self._kwargs = dict(n_estimators=n_estimators, num_leaves=num_leaves,
                            learning_rate=learning_rate, objective=objective,
                            alpha=quantile if quantile is not None else 0.5, verbose=-1)
        self._models: list[LGBMRegressor] = []
        self._columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "Lgbm":
        self._columns = list(X.columns)
        Xv = X.to_numpy(float)
        self._models = []
        for col in Y.columns:
            y = Y[col]
            m = y.notna().to_numpy()
            est = LGBMRegressor(**self._kwargs)
            est.fit(Xv[m], y[m].to_numpy(float))
            self._models.append(est)
        return self

    def predict(self, X: pd.DataFrame, endog=None) -> np.ndarray:
        Xv = X.reindex(columns=self._columns).to_numpy(float)
        return np.column_stack([est.predict(Xv) for est in self._models])

    def save(self, path: Path) -> None:
        joblib.dump({"models": self._models, "columns": self._columns}, path)

    @classmethod
    def load(cls, path: Path) -> "Lgbm":
        blob = joblib.load(path)
        obj = cls()
        obj._models = blob["models"]
        obj._columns = blob["columns"]
        return obj
