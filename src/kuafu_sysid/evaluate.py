"""Load a pinned model and forecast / evaluate on new data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kuafu_sysid.config import SelectionConfig
from kuafu_sysid.features import FeatureSpec, build_features
from kuafu_sysid.metrics import per_horizon_metrics
from kuafu_sysid.store import ModelStore


@dataclass
class FittedForecaster:
    model: object
    recipe: dict

    def _spec(self) -> FeatureSpec:
        r = self.recipe
        return FeatureSpec(endog=r["endog"], exog=tuple(r["exog"]),
                           exog_with_lag=tuple(r["exog_with_lag"]),
                           forecast_exog=tuple(r["forecast_exog"]))

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        """Match the training resolution: downsample to the recipe's dt_min when the
        eval data is finer (no-op if it's already at that step)."""
        dt = self.recipe.get("dt_min")
        if dt:
            native = round(df.index.to_series().diff().median().total_seconds() / 60)
            if native < dt:
                from kuafu_sysid.features import resample_df
                df = resample_df(df, dt, self.recipe.get("resample_agg", "mean"))
        return df

    def features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        r = self.recipe
        df = self._prep(df)
        return build_features(df, self._spec(), r["lag"], r["horizon"], r["dt_min"], r["time_features"])

    def _clip(self, a):
        cm = self.recipe.get("clip_min")
        return a if cm is None else np.maximum(a, cm)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._prep(df)
        X, _ = build_features(df, self._spec(), self.recipe["lag"], self.recipe["horizon"],
                              self.recipe["dt_min"], self.recipe["time_features"])
        X = X.reindex(columns=self.recipe["feature_columns"])
        pred = self._clip(self.model.predict(X, endog=df[self.recipe["endog"]]))
        return pd.DataFrame(pred, index=X.index, columns=self.recipe["target_columns"])

    def predict_quantiles(self, df: pd.DataFrame) -> dict:
        """Map quantile -> wide forecast DataFrame. Only models that produce
        quantiles (LGBM with a band) support this; others raise."""
        if not hasattr(self.model, "predict_quantiles"):
            raise ValueError(f"{self.recipe['method']} has no quantiles "
                             "(train an LGBM with a `quantiles` band)")
        df = self._prep(df)
        X, _ = build_features(df, self._spec(), self.recipe["lag"], self.recipe["horizon"],
                              self.recipe["dt_min"], self.recipe["time_features"])
        X = X.reindex(columns=self.recipe["feature_columns"])
        cols, idx = self.recipe["target_columns"], X.index
        return {q: pd.DataFrame(self._clip(a), index=idx, columns=cols)
                for q, a in self.model.predict_quantiles(X).items()}


@dataclass
class EvalResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    period: tuple | None = None   # (first, last) timestamp the metrics were scored over


def load_forecaster(sel: SelectionConfig, role: str) -> FittedForecaster:
    rs = sel.roles[role]
    store = ModelStore(sel.store_root)
    model, recipe = store.load(rs.target, rs.method, rs.feature_hash, rs.train_start, rs.train_end)
    return FittedForecaster(model=model, recipe=recipe)


def _eval_fc(fc: FittedForecaster, data: pd.DataFrame, start, end) -> EvalResult:
    """Score one fitted forecaster on ``data`` over the ``[start:end]`` window."""
    _, Y = fc.features(data)
    pred = fc.predict(data)
    yp = pred if (start is None and end is None) else pred.loc[start:end]
    yt = Y.reindex(yp.index)
    metrics = per_horizon_metrics(yt, yp.to_numpy())
    period = (yp.index.min(), yp.index.max()) if len(yp) else None
    return EvalResult(metrics=metrics, predictions=pred, period=period)


def evaluate(sel: SelectionConfig, role: str, data: pd.DataFrame,
             start=None, end=None) -> EvalResult:
    """Evaluate a pinned model on ``data``.

    Features are built on the full ``data`` (so lags have history), but the
    metrics are scored only over ``[start:end]`` — pass ``start=train_end`` to
    score on genuinely out-of-sample rows the model never trained on. The
    returned ``predictions`` cover the full ``data`` regardless of the window.

    Requires ``method`` to be set on the role; for a blank method (all models)
    use :func:`evaluate_all`.
    """
    return _eval_fc(load_forecaster(sel, role), data, start, end)


def evaluate_all(sel: SelectionConfig, role: str, data: pd.DataFrame,
                 start=None, end=None) -> dict[str, EvalResult]:
    """Evaluate one or all trained models for a role, keyed by method name.

    If the role pins a ``method`` only that one is evaluated; if ``method`` is
    blank, every method in the target's ``_latest.json`` is evaluated (each at
    its latest hash). Same scoring semantics as :func:`evaluate`.
    """
    rs = sel.roles[role]
    store = ModelStore(sel.store_root)
    methods = [rs.method] if rs.method else store.list_methods(rs.target)
    out: dict[str, EvalResult] = {}
    for m in methods:
        model, recipe = store.load(rs.target, m, rs.feature_hash, rs.train_start, rs.train_end)
        out[m] = _eval_fc(FittedForecaster(model=model, recipe=recipe), data, start, end)
    return out
