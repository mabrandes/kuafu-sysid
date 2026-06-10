"""Ridge multi-output linear adapter (dense; drops NaN rows).

By default the regularisation strength ``alpha`` is chosen by **k-fold
cross-validation** (``RidgeCV``) per output — pass an explicit ``alpha`` to skip
CV and use a fixed value.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.multioutput import MultiOutputRegressor

from kuafu_sysid.models.base import Forecaster


class Linear(Forecaster):
    EXT = "joblib"

    def __init__(self, alpha: float | None = None,
                 alphas=(0.01, 0.1, 1.0, 10.0, 100.0), cv: int = 5, **_ignored):
        # alpha=None -> cross-validate alpha over `alphas` with `cv` folds (RidgeCV);
        # alpha=<float> -> fixed Ridge (no CV).
        self.alpha = alpha
        self.alphas = tuple(alphas)
        self.cv = cv
        base = Ridge(alpha=alpha) if alpha is not None else RidgeCV(alphas=self.alphas, cv=cv)
        self._model = MultiOutputRegressor(base)
        self._columns: list[str] | None = None
        self._n_out: int = 0
        self.alpha_: list[float] | None = None   # CV-chosen alpha per horizon (if RidgeCV)

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "Linear":
        """Fit one Ridge per horizon. X: (n_samples, n_features); Y: (n_samples,
        horizon H). Rows with any NaN in X or Y are dropped (dense model). With
        ``alpha=None`` each horizon's alpha is selected by ``cv``-fold CV."""
        self._columns = list(X.columns)
        self._n_out = Y.shape[1]
        mask = X.notna().all(axis=1) & Y.notna().all(axis=1)
        self._model.fit(X.loc[mask].to_numpy(float), Y.loc[mask].to_numpy(float))
        if self.alpha is None:   # record the CV-selected alpha per output
            self.alpha_ = [float(est.alpha_) for est in self._model.estimators_]
        return self

    def predict(self, X: pd.DataFrame, endog=None) -> np.ndarray:
        """Predict all horizons. X: (n_samples, n_features) — reindexed to the
        training column order. Returns ndarray (n_samples, horizon H); rows with
        any NaN feature yield a full row of NaN (dense model can't score them)."""
        Xv = X.reindex(columns=self._columns)
        complete = Xv.notna().all(axis=1).to_numpy()
        out = np.full((len(Xv), self._n_out), np.nan)
        if complete.any():
            out[complete] = self._model.predict(Xv.loc[complete].to_numpy(float))
        return out

    def save(self, path: Path) -> None:
        joblib.dump({"model": self._model, "columns": self._columns,
                     "n_out": self._n_out, "alpha_": self.alpha_}, path)

    @classmethod
    def load(cls, path: Path) -> "Linear":
        blob = joblib.load(path)
        obj = cls()
        obj._model = blob["model"]
        obj._columns = blob["columns"]
        obj._n_out = blob["n_out"]
        obj.alpha_ = blob.get("alpha_")
        return obj
