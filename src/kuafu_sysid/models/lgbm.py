"""LightGBM adapter (NaN-native). One regressor per horizon; optional quantile."""
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
                 learning_rate: float = 0.05, quantile: float | None = None,
                 early_stopping_rounds: int = 50, val_fraction: float = 0.2,
                 eval_log: int = 0, **_ignored):
        objective = "quantile" if quantile is not None else "regression"
        # n_estimators is an upper bound; early stopping per horizon picks best_iteration_.
        self._kwargs = dict(n_estimators=n_estimators, num_leaves=num_leaves,
                            learning_rate=learning_rate, objective=objective,
                            alpha=quantile if quantile is not None else 0.5, verbose=-1)
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.eval_log = eval_log    # 0 = silent; N>0 logs train/val every N rounds (per horizon)
        self._models: list[LGBMRegressor] = []
        self._columns: list[str] | None = None
        self.best_iteration_: int | None = None   # median best_iteration across horizons

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "Lgbm":
        """Fit one LightGBM per horizon (H models). X: (n_samples, n_features) —
        NaN allowed (tree-native); Y: (n_samples, horizon H). Per column, rows with
        a NaN target are dropped and the last ``val_fraction`` is held out for early
        stopping."""
        self._columns = list(X.columns)
        Xv = X.to_numpy(float)
        self._models = []
        best_iters = []
        for col in Y.columns:
            y = Y[col]
            m = y.notna().to_numpy()
            Xc, yc = Xv[m], y[m].to_numpy(float)
            est = LGBMRegressor(**self._kwargs)
            n = len(Xc)
            if self.early_stopping_rounds and self.val_fraction and n >= 20:
                cut = max(1, int(n * (1 - self.val_fraction)))  # time-ordered val tail
                callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
                if self.eval_log:
                    callbacks.append(lgb.log_evaluation(self.eval_log))
                est.fit(Xc[:cut], yc[:cut],
                        eval_set=[(Xc[:cut], yc[:cut]), (Xc[cut:], yc[cut:])],
                        eval_names=["train", "val"], callbacks=callbacks)
                if est.best_iteration_:
                    best_iters.append(est.best_iteration_)
            else:
                est.fit(Xc, yc)
            self._models.append(est)
        self.best_iteration_ = int(np.median(best_iters)) if best_iters else None
        return self

    def feature_importances(self) -> pd.Series:
        # average the per-horizon models' importances
        imp = np.mean([m.feature_importances_ for m in self._models], axis=0)
        return pd.Series(imp, index=self._columns)

    def predict(self, X: pd.DataFrame, endog=None) -> np.ndarray:
        """Predict all horizons. X: (n_samples, n_features) — reindexed to the
        training columns, NaN allowed. Stacks the H per-horizon models' outputs ->
        (n_samples, horizon H); each uses its own early-stopped best_iteration_."""
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
