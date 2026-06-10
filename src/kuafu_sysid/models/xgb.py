"""XGBoost multi-output adapter (NaN-native) with validation + early stopping."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from kuafu_sysid.models.base import Forecaster


class Xgb(Forecaster):
    EXT = "json"

    def __init__(self, n_estimators: int = 1000, max_depth: int = 6,
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, early_stopping_rounds: int = 50,
                 val_fraction: float = 0.2, eval_log: int = 0, **_ignored):
        # n_estimators is an upper bound; early stopping on a held-out validation
        # tail picks the actual number of trees (best_iteration).
        # eval_log: 0 = silent; N>0 prints train/validation RMSE every N rounds.
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.eval_log = eval_log
        self._model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=subsample, colsample_bytree=colsample_bytree,
            tree_method="hist", random_state=42,
        )
        self._columns: list[str] | None = None
        self.best_iteration_: int | None = None
        self.evals_result_: dict | None = None   # {'validation_0': train, 'validation_1': val}

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "Xgb":
        self._columns = list(X.columns)
        m = Y.notna().all(axis=1)  # complete targets; X may contain NaN (tree-native)
        Xv, Yv = X.loc[m].to_numpy(float), Y.loc[m].to_numpy(float)
        n = len(Xv)
        if self.early_stopping_rounds and self.val_fraction and n >= 20:
            cut = max(1, int(n * (1 - self.val_fraction)))   # time-ordered: val = tail of train
            self._model.set_params(early_stopping_rounds=self.early_stopping_rounds)
            self._model.fit(
                Xv[:cut], Yv[:cut],
                eval_set=[(Xv[:cut], Yv[:cut]), (Xv[cut:], Yv[cut:])],  # [train, validation]
                verbose=self.eval_log,   # 0=silent; N prints validation_0/1 RMSE every N rounds
            )
            self.best_iteration_ = int(getattr(self._model, "best_iteration", None) or 0)
            self.evals_result_ = self._model.evals_result()
        else:
            self._model.fit(Xv, Yv)   # too few rows to hold out a validation set
        return self

    def feature_importances(self) -> pd.Series:
        return pd.Series(self._model.feature_importances_, index=self._columns)

    def predict(self, X: pd.DataFrame, endog=None) -> np.ndarray:
        Xa = X.reindex(columns=self._columns).to_numpy(float)
        if self.best_iteration_ is not None:
            return self._model.predict(Xa, iteration_range=(0, self.best_iteration_ + 1))
        return self._model.predict(Xa)

    def save(self, path: Path) -> None:
        self._model.save_model(str(path))
        meta = {"columns": self._columns, "best_iteration": self.best_iteration_}
        Path(path).with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Xgb":
        obj = cls()
        obj._model = XGBRegressor()
        obj._model.load_model(str(path))
        meta = json.loads(Path(path).with_suffix(".meta.json").read_text(encoding="utf-8"))
        obj._columns = meta["columns"]
        obj.best_iteration_ = meta.get("best_iteration")
        return obj
