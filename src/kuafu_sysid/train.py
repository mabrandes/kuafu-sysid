"""Train all configured models from a TrainConfig and save them."""
from __future__ import annotations

import numpy as np
import pandas as pd

from kuafu_sysid.config import TrainConfig
from kuafu_sysid.features import build_features, feature_hash, normalize_lags
from kuafu_sysid.metrics import per_horizon_metrics
from kuafu_sysid.models import get_model
from kuafu_sysid.store import ModelStore


def _steps_per(dt_min: int) -> tuple[int, int]:
    spd = int(round(24 * 60 / dt_min))
    return spd, spd * 7


def _split(X, Y, split):
    n = len(X)
    if split < 0:                       # test-before-train (backtest)
        cut = int(round(-split * n))
        return X.iloc[cut:], Y.iloc[cut:], X.iloc[:cut], Y.iloc[:cut]
    cut = int(round((1 - split) * n))   # test = last split fraction
    return X.iloc[:cut], Y.iloc[:cut], X.iloc[cut:], Y.iloc[cut:]


def train(cfg: TrainConfig) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(cfg.data_path).sort_index()
    if cfg.train_start or cfg.train_end:
        df = df.loc[cfg.train_start:cfg.train_end]
    dt_min = cfg.dt_min or int(round(df.index.to_series().diff().median().total_seconds() / 60))
    spd, spw = _steps_per(dt_min)

    X, Y = build_features(df, cfg.spec, cfg.lag, cfg.horizon, dt_min, cfg.time_features)
    X_tr, Y_tr, X_te, Y_te = _split(X, Y, cfg.split)

    fhash = feature_hash(cfg.spec, cfg.lag, cfg.horizon, dt_min, cfg.time_features)
    recipe_base = {
        "target": cfg.target, "endog": cfg.spec.endog,
        "exog": list(cfg.spec.exog), "exog_with_lag": list(cfg.spec.exog_with_lag),
        "forecast_exog": list(cfg.spec.forecast_exog),
        "lag": list(normalize_lags(cfg.lag)), "horizon": cfg.horizon, "dt_min": dt_min,
        "time_features": cfg.time_features, "feature_hash": fhash,
        "train_start": cfg.train_start, "train_end": cfg.train_end,
        "feature_columns": list(X.columns), "target_columns": list(Y.columns),
    }
    store = ModelStore(cfg.store_root)
    results: dict[str, pd.DataFrame] = {}
    for method in cfg.models:
        model = get_model(method, horizon=cfg.horizon, endog=cfg.spec.endog,
                          steps_per_day=spd, steps_per_week=spw)
        model.fit(X_tr, Y_tr)
        pred = model.predict(X_te, endog=df[cfg.spec.endog])
        results[method] = per_horizon_metrics(Y_te, pred)
        if cfg.save:
            store.save(cfg.target, method, model, {**recipe_base, "method": method},
                       cfg.train_start, cfg.train_end)
    return results
