"""XGBoost multi-output adapter (NaN-native)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from kuafu_sysid.models.base import Forecaster


class Xgb(Forecaster):
    EXT = "json"

    def __init__(self, n_estimators: int = 400, max_depth: int = 6,
                 learning_rate: float = 0.05, **_ignored):
        self._model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, tree_method="hist",
        )
        self._columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "Xgb":
        self._columns = list(X.columns)
        m = Y.notna().all(axis=1)  # need complete targets; X may contain NaN (native)
        self._model.fit(X.loc[m].to_numpy(float), Y.loc[m].to_numpy(float))
        return self

    def predict(self, X: pd.DataFrame, endog=None) -> np.ndarray:
        return self._model.predict(X.reindex(columns=self._columns).to_numpy(float))

    def save(self, path: Path) -> None:
        self._model.save_model(str(path))
        Path(path).with_suffix(".cols").write_text("\n".join(self._columns), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Xgb":
        obj = cls()
        obj._model = XGBRegressor()
        obj._model.load_model(str(path))
        obj._columns = Path(path).with_suffix(".cols").read_text(encoding="utf-8").splitlines()
        return obj
