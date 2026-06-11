"""Train all configured models from a TrainConfig and save them."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from kuafu_sysid.config import TrainConfig
from kuafu_sysid.features import build_features, feature_hash, normalize_lags, resample_df
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


def _save_model_plots(art_path, model, X_te=None, target_columns=None,
                      actual=None, step: int = 1, clip_min=None) -> None:
    """Per-model PNGs next to the artefact: learning curve (XGB), feature importance
    (tree models), and an uncertainty-band timeseries (quantile LGBM).
    Cross-model comparisons live in _save_compare_plots."""
    import matplotlib
    matplotlib.use("Agg")   # headless: write files, never pop a window
    import matplotlib.pyplot as plt

    from kuafu_sysid.plots import plot_feature_importance, plot_forecast_band, plot_learning_curve
    d = art_path.parent
    stem = art_path.with_suffix("").name   # method_hash_start_end

    if getattr(model, "evals_result_", None):   # XGB train-vs-validation (over/underfitting)
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_learning_curve(model, ax=ax)
        fig.savefig(d / f"{stem}_learning_curve.png", bbox_inches="tight"); plt.close(fig)

    if model.feature_importances() is not None:   # tree models only
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_feature_importance(model, ax=ax)
        fig.savefig(d / f"{stem}_importance.png", bbox_inches="tight"); plt.close(fig)

    # quantile LGBM: measured vs median + uncertainty band at `step`
    if (X_te is not None and hasattr(model, "predict_quantiles")
            and len(getattr(model, "quantiles", (0.5,))) > 1):
        qp = model.predict_quantiles(X_te)
        if clip_min is not None:
            qp = {q: np.maximum(a, clip_min) for q, a in qp.items()}
        qs = sorted(qp)
        wide = lambda a: pd.DataFrame(a, index=X_te.index, columns=target_columns)
        w0 = X_te.index.min()
        w1 = min(w0 + pd.Timedelta(days=4), X_te.index.max())
        fig, ax = plt.subplots(figsize=(11, 4))
        plot_forecast_band(actual, wide(qp[0.5]), wide(qp[qs[0]]), wide(qp[qs[-1]]),
                           step=step, start=w0, end=w1, ax=ax)
        fig.savefig(d / f"{stem}_band.png", bbox_inches="tight"); plt.close(fig)


def _save_compare_plots(d, prefix, results, preds_by_method, actual,
                        step: int, days: int = 4) -> None:
    """One combined horizon plot (RMSE/MAE/R², all models) and one combined
    timeseries plot (all models at horizon `step`, over a short window). Named
    `{prefix}_*` where prefix = {feature_hash}_{train_start}_{train_end}, matching
    the per-model artefact naming."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from kuafu_sysid.plots import plot_horizon_metrics, plot_timeseries_compare

    axes = plot_horizon_metrics(results, which=("rmse", "mae", "r2"))   # all models overlaid
    fig = axes[0].figure
    fig.savefig(d / f"{prefix}_horizon_compare.png", bbox_inches="tight"); plt.close(fig)

    any_preds = next(iter(preds_by_method.values()))
    w0 = any_preds.index.min()
    w1 = min(w0 + pd.Timedelta(days=days), any_preds.index.max())   # fewer days = clearer
    fig, ax = plt.subplots(figsize=(12, 4.5))
    plot_timeseries_compare(actual, preds_by_method, step=step, start=w0, end=w1, ax=ax)
    fig.savefig(d / f"{prefix}_timeseries_compare.png", bbox_inches="tight"); plt.close(fig)


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
    native = int(round(df.index.to_series().diff().median().total_seconds() / 60))
    dt_min = cfg.dt_min or native              # target working resolution
    if dt_min > native:                        # dt_min coarser than data -> downsample
        df = resample_df(df, dt_min, cfg.resample_agg)
        log(f"downsampled {native}min -> {dt_min}min (agg={cfg.resample_agg})")
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
        "clip_min": cfg.clip_min, "resample_agg": cfg.resample_agg,
        "feature_columns": list(X.columns), "target_columns": list(Y.columns),
    }
    store = ModelStore(cfg.store_root)
    results: dict[str, pd.DataFrame] = {}
    preds_by_method: dict[str, pd.DataFrame] = {}
    step12h = max(1, min(int(round(12 * 60 / dt_min)), cfg.horizon))   # 12 h ahead
    n = len(cfg.models)
    for i, method in enumerate(cfg.models, 1):
        log(f"[{i}/{n}] training {method} ...")
        t0 = time.perf_counter()
        model = get_model(method, horizon=cfg.horizon, endog=cfg.spec.endog,
                          steps_per_day=spd, steps_per_week=spw, eval_log=tree_eval_log,
                          quantiles=cfg.quantiles, cv=cfg.linear_cv)
        model.fit(X_tr, Y_tr)
        pred = model.predict(X_te, endog=df[cfg.spec.endog])
        if cfg.clip_min is not None:
            pred = np.maximum(pred, cfg.clip_min)   # e.g. PV can't be negative
        results[method] = per_horizon_metrics(Y_te, pred)
        preds_by_method[method] = pd.DataFrame(pred, index=X_te.index, columns=list(Y.columns))
        dt = time.perf_counter() - t0
        bi = getattr(model, "best_iteration_", None)   # trees kept after early stopping
        es = f", early-stopped @ {bi} trees" if bi else ""
        log(f"[{i}/{n}] {method}: mean RMSE={results[method]['rmse'].mean():.3f}  ({dt:.1f}s{es})")
        if cfg.save:
            art = store.save(cfg.target, method, model, {**recipe_base, "method": method},
                             cfg.train_start, cfg.train_end)
            if save_plots:
                _save_model_plots(art, model, X_te=X_te, target_columns=list(Y.columns),
                                  actual=df[cfg.spec.endog], step=step12h, clip_min=cfg.clip_min)
    if cfg.save and save_plots and results:
        prefix = f"{fhash}_{cfg.train_start}_{cfg.train_end}"   # match per-model naming
        _save_compare_plots(store.root / cfg.target, prefix, results,
                            preds_by_method, df[cfg.spec.endog], step=step12h)
    log(f"done: {n} model(s) "
        + ("saved to " + str(store.root / cfg.target) if cfg.save else "(not saved)"))
    return results
