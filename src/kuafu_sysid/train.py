"""Train all configured models from a TrainConfig and save them."""
from __future__ import annotations

import time

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


def _save_model_plots(art_path, method, metrics_df, model, preds_df, actual) -> None:
    """Save horizon / timeseries / feature-importance PNGs next to the artefact."""
    import matplotlib
    matplotlib.use("Agg")   # headless: write files, never pop a window
    import matplotlib.pyplot as plt

    from kuafu_sysid.plots import (
        plot_feature_importance, plot_horizon_metrics, plot_learning_curve, plot_timeseries,
    )
    d = art_path.parent
    stem = art_path.with_suffix("").name   # method_hash_start_end

    if getattr(model, "evals_result_", None):   # XGB train-vs-validation (over/underfitting)
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_learning_curve(model, ax=ax)
        fig.savefig(d / f"{stem}_learning_curve.png", bbox_inches="tight"); plt.close(fig)

    axes = plot_horizon_metrics({method: metrics_df}, which=("rmse", "mae", "r2"))  # 3-panel
    fig = axes[0].figure
    fig.savefig(d / f"{stem}_horizon.png", bbox_inches="tight"); plt.close(fig)

    if len(preds_df):
        w0 = preds_df.index.min()
        w1 = min(w0 + pd.Timedelta(days=14), preds_df.index.max())   # readable window
        fig, ax = plt.subplots(figsize=(11, 4))
        plot_timeseries(actual, preds_df, step=1, start=w0, end=w1, ax=ax)
        fig.savefig(d / f"{stem}_timeseries.png", bbox_inches="tight"); plt.close(fig)

    if model.feature_importances() is not None:   # tree models only
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_feature_importance(model, ax=ax)
        fig.savefig(d / f"{stem}_importance.png", bbox_inches="tight"); plt.close(fig)


def train(cfg: TrainConfig, verbose: bool = True,
          tree_eval_log: int = 0, save_plots: bool = True) -> dict[str, pd.DataFrame]:
    """Train every model in ``cfg.models`` and (optionally) save them.

    ``verbose=False`` silences the per-model progress prints. ``tree_eval_log=N``
    (N>0) streams XGB/LGBM train-vs-validation RMSE every N boosting rounds — like
    XGBoost's ``verbose=True`` (note LGBM prints per horizon, so it's chattier).
    ``save_plots`` (default True, only when ``cfg.save``) writes a horizon,
    timeseries, and (tree-only) feature-importance PNG next to each saved model.
    """
    def log(msg: str) -> None:
        if verbose:
            print(f"[sysid] {msg}", flush=True)

    log(f"target '{cfg.target}': loading {cfg.data_path}")
    df = pd.read_parquet(cfg.data_path).sort_index()
    if cfg.train_start or cfg.train_end:
        df = df.loc[cfg.train_start:cfg.train_end]
    dt_min = cfg.dt_min or int(round(df.index.to_series().diff().median().total_seconds() / 60))
    spd, spw = _steps_per(dt_min)
    log(f"data: {len(df)} rows  {df.index.min():%Y-%m-%d}..{df.index.max():%Y-%m-%d}  (dt={dt_min}min)")

    X, Y = build_features(df, cfg.spec, cfg.lag, cfg.horizon, dt_min, cfg.time_features)
    X_tr, Y_tr, X_te, Y_te = _split(X, Y, cfg.split)
    fhash = feature_hash(cfg.spec, cfg.lag, cfg.horizon, dt_min, cfg.time_features)
    log(f"features: {X.shape[1]} cols (lags={list(normalize_lags(cfg.lag))}), "
        f"horizon={cfg.horizon}  | train={len(X_tr)} test={len(X_te)} rows  | hash={fhash}")

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
    n = len(cfg.models)
    for i, method in enumerate(cfg.models, 1):
        log(f"[{i}/{n}] training {method} ...")
        t0 = time.perf_counter()
        model = get_model(method, horizon=cfg.horizon, endog=cfg.spec.endog,
                          steps_per_day=spd, steps_per_week=spw, eval_log=tree_eval_log)
        model.fit(X_tr, Y_tr)
        pred = model.predict(X_te, endog=df[cfg.spec.endog])
        results[method] = per_horizon_metrics(Y_te, pred)
        dt = time.perf_counter() - t0
        bi = getattr(model, "best_iteration_", None)   # trees kept after early stopping
        es = f", early-stopped @ {bi} trees" if bi else ""
        log(f"[{i}/{n}] {method}: mean RMSE={results[method]['rmse'].mean():.3f}  ({dt:.1f}s{es})")
        if cfg.save:
            art = store.save(cfg.target, method, model, {**recipe_base, "method": method},
                             cfg.train_start, cfg.train_end)
            if save_plots:
                preds_df = pd.DataFrame(pred, index=X_te.index, columns=list(Y.columns))
                _save_model_plots(art, method, results[method], model,
                                  preds_df, df[cfg.spec.endog])
    log(f"done: {n} model(s) "
        + ("saved to " + str(store.root / cfg.target) if cfg.save else "(not saved)"))
    return results
