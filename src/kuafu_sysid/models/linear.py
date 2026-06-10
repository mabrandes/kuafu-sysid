"""Ridge multi-output linear adapter (dense; drops NaN rows)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from kuafu_sysid.models.base import Forecaster


class Linear(Forecaster):
    EXT = "joblib"

    def __init__(self, alpha: float = 1.0, **_ignored):
        self.alpha = alpha
        self._model = MultiOutputRegressor(Ridge(alpha=alpha))
        self._columns: list[str] | None = None
        self._n_out: int = 0

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "Linear":
        self._columns = list(X.columns)
        self._n_out = Y.shape[1]
        mask = X.notna().all(axis=1) & Y.notna().all(axis=1)
        self._model.fit(X.loc[mask].to_numpy(float), Y.loc[mask].to_numpy(float))
        return self

    def predict(self, X: pd.DataFrame, endog=None) -> np.ndarray:
        Xv = X.reindex(columns=self._columns)
        complete = Xv.notna().all(axis=1).to_numpy()
        out = np.full((len(Xv), self._n_out), np.nan)
        if complete.any():
            out[complete] = self._model.predict(Xv.loc[complete].to_numpy(float))
        return out

    def save(self, path: Path) -> None:
        joblib.dump({"model": self._model, "columns": self._columns, "n_out": self._n_out}, path)

    @classmethod
    def load(cls, path: Path) -> "Linear":
        blob = joblib.load(path)
        obj = cls()
        obj._model = blob["model"]
        obj._columns = blob["columns"]
        obj._n_out = blob["n_out"]
        return obj
