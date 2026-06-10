"""LightGBM adapter (NaN-native). One regressor per (quantile, horizon).

Trains a set of quantiles (always including the 0.5 median, which is the point
forecast ``predict()`` returns); ``predict_quantiles()`` exposes the full set so
callers can draw an uncertainty band (e.g. 0.1 / 0.9).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from kuafu_sysid.models.base import Forecaster


class Lgbm(Forecaster):
    EXT = "joblib"

    def __init__(self, n_estimators: int = 1000, num_leaves: int = 31,
                 learning_rate: float = 0.05, quantiles=(0.5,),
                 early_stopping_rounds: int = 50, val_fraction: float = 0.2,
                 eval_log: int = 0, **_ignored):
        # quantiles always include the 0.5 median (the point forecast). Extra
        # quantiles (e.g. 0.1, 0.9) form an uncertainty band via predict_quantiles().
        self.quantiles = tuple(sorted({0.5, *(float(q) for q in quantiles)}))
        self._base = dict(n_estimators=n_estimators, num_leaves=num_leaves,
                          learning_rate=learning_rate, objective="quantile", verbose=-1)
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.eval_log = eval_log
        self._models: dict[float, list[LGBMRegressor]] = {}   # quantile -> [model per horizon]
        self._columns: list[str] | None = None
        self.best_iteration_: int | None = None

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "Lgbm":
        """Fit one LightGBM per (quantile, horizon) — len(quantiles)·H models.
        X: (n_samples, n_features) — NaN allowed (tree-native); Y: (n_samples,
        horizon H). Per model, rows with a NaN target are dropped and the last
        ``val_fraction`` is held out for early stopping."""
        self._columns = list(X.columns)
        Xv = X.to_numpy(float)
        self._models = {q: [] for q in self.quantiles}
        best_iters = []
        for q in self.quantiles:
            for col in Y.columns:
                y = Y[col]
                m = y.notna().to_numpy()
                Xc, yc = Xv[m], y[m].to_numpy(float)
                est = LGBMRegressor(alpha=q, **self._base)
                n = len(Xc)
                if self.early_stopping_rounds and self.val_fraction and n >= 20:
                    cut = max(1, int(n * (1 - self.val_fraction)))   # time-ordered val tail
                    cbs = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
                    if self.eval_log:
                        cbs.append(lgb.log_evaluation(self.eval_log))
                    est.fit(Xc[:cut], yc[:cut],
                            eval_set=[(Xc[:cut], yc[:cut]), (Xc[cut:], yc[cut:])],
                            eval_names=["train", "val"], callbacks=cbs)
                    if est.best_iteration_:
                        best_iters.append(est.best_iteration_)
                else:
                    est.fit(Xc, yc)
                self._models[q].append(est)
        self.best_iteration_ = int(np.median(best_iters)) if best_iters else None
        return self

    def _predict_quantile(self, X: pd.DataFrame, q: float) -> np.ndarray:
        Xv = X.reindex(columns=self._columns).to_numpy(float)
        return np.column_stack([est.predict(Xv) for est in self._models[q]])

    def feature_importances(self) -> pd.Series:
        imp = np.mean([m.feature_importances_ for m in self._models[0.5]], axis=0)
        return pd.Series(imp, index=self._columns)

    def predict(self, X: pd.DataFrame, endog=None) -> np.ndarray:
        """Point forecast = the 0.5 median. X: (n_samples, n_features) -> (n_samples,
        horizon H). Use ``predict_quantiles`` for the full band."""
        return self._predict_quantile(X, 0.5)

    def predict_quantiles(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        """Map each trained quantile -> (n_samples, horizon H) prediction array."""
        return {q: self._predict_quantile(X, q) for q in self.quantiles}

    def save(self, path: Path) -> None:
        joblib.dump({"models": self._models, "columns": self._columns,
                     "quantiles": self.quantiles}, path)

    @classmethod
    def load(cls, path: Path) -> "Lgbm":
        blob = joblib.load(path)
        obj = cls()
        obj._models = blob["models"]
        obj._columns = blob["columns"]
        obj.quantiles = tuple(blob.get("quantiles", (0.5,)))
        return obj
