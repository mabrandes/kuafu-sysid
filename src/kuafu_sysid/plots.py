"""Plain-matplotlib evaluation plots (projects can restyle)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_error_by_horizon(metrics_by_method: dict[str, pd.DataFrame], metric: str = "rmse", ax=None):
    """One line per method: a single metric vs horizon step."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for name, m in metrics_by_method.items():
        ax.plot(m.index, m[metric], marker="o", label=name)
    ax.set_xlabel("Horizon step")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} by horizon")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_horizon_metrics(metrics, which=("rmse", "mae", "r2"), axes=None):
    """One panel per metric (RMSE / MAE / R² by default) vs. horizon step.

    ``metrics`` is a per-horizon DataFrame (e.g. ``evaluate(...).metrics``) or a
    ``{name: DataFrame}`` dict to overlay several models.
    """
    if isinstance(metrics, pd.DataFrame):
        metrics = {"model": metrics}
    if axes is None:
        _, axes = plt.subplots(1, len(which), figsize=(4.6 * len(which), 3.8))
    axes = np.atleast_1d(axes)
    for ax, met in zip(axes, which):
        for name, df in metrics.items():
            ax.plot(df.index, df[met], marker="o", ms=3, label=name)
        ax.set_xlabel("Horizon step")
        ax.set_title(met.upper())
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    return axes


def plot_feature_importance(model, top: int = 30, ax=None):
    """Horizontal bar chart of a tree model's top-``top`` feature importances.

    Works for `Xgb`/`Lgbm` (averaged across horizons for LGBM). Raises for models
    without importances (Linear/KNN/persistence)."""
    imp = model.feature_importances()
    if imp is None:
        raise ValueError("this model has no feature importances (not a tree model)")
    imp = imp.sort_values(ascending=False).head(top).iloc[::-1]   # largest at top
    if ax is None:
        _, ax = plt.subplots(figsize=(8, min(0.3 * len(imp) + 1, 11)))
    ax.barh(imp.index.astype(str), imp.to_numpy())
    ax.set_xlabel("importance")
    ax.set_title(f"Feature importance (top {len(imp)})")
    ax.tick_params(axis="y", labelsize=7)
    return ax


def plot_timeseries(actual: pd.Series, predictions: pd.DataFrame, step=1,
                    start=None, end=None, ax=None):
    """Plot measured vs. forecast over calendar (delivery) time.

    ``actual``       — the endog series (e.g. ``df[endog]``).
    ``predictions``  — wide forecast frame from ``evaluate()`` / ``load_forecaster``
                       (index = forecast origin, columns ``{endog}_h_1..H``).
    ``step``         — a horizon step (int) or several (e.g. ``[1, 4, 24, 96]``).
                       The step-``h`` forecast issued at origin ``t`` is for delivery
                       time ``t + h``, so its index is shifted forward by ``h`` steps
                       to line up with the measured series.
    ``start``/``end`` zoom to a date window (recommended — the full series is dense).
    """
    steps = [step] if isinstance(step, int) else list(step)
    dt = predictions.index.to_series().diff().median()
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    a = actual if (start is None and end is None) else actual.loc[start:end]
    ax.plot(a.index, a.to_numpy(), label="measured", lw=1.6, color="black", zorder=len(steps) + 2)
    for s in steps:
        pred = pd.Series(predictions.iloc[:, s - 1].to_numpy(), index=predictions.index + s * dt)
        p = pred if (start is None and end is None) else pred.loc[start:end]
        ax.plot(p.index, p.to_numpy(), lw=1.0, alpha=0.85, label=f"forecast h={s}")
    ax.set_ylabel(predictions.columns[steps[0] - 1].rsplit("_h_", 1)[0])
    ax.set_title("Measured vs. forecast" + (f" — horizon step {steps[0]}" if len(steps) == 1
                                            else f" — horizon steps {steps}"))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_forecast_origin(actual: pd.Series, predictions: pd.DataFrame, origin,
                         lower: pd.DataFrame | None = None, upper: pd.DataFrame | None = None,
                         ax=None):
    """Plot the whole forecast *trajectory* issued at one ``origin`` time.

    Fixes the forecast origin and walks every horizon h_1..H, so for a PV model
    with a 1-day horizon, ``origin`` = 08:00 shows that morning's full day-ahead
    forecast vs. what actually happened. ``predictions`` is a wide frame (index =
    origin, columns ``{endog}_h_1..H``); optional ``lower``/``upper`` (same shape,
    e.g. LGBM quantiles) draw an uncertainty band around the trajectory.
    """
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None and predictions.index.tz is not None:
        origin = origin.tz_localize(predictions.index.tz)
    if origin not in predictions.index:   # snap to the nearest available origin
        origin = predictions.index[predictions.index.get_indexer([origin], method="nearest")[0]]
    dt = predictions.index.to_series().diff().median()
    deliv = [origin + (h + 1) * dt for h in range(predictions.shape[1])]   # h_1..h_H delivery times

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    a = actual.loc[origin:deliv[-1]]
    ax.plot(a.index, a.to_numpy(), color="black", lw=1.6, label="measured")
    if lower is not None and upper is not None:
        ax.fill_between(deliv, lower.loc[origin].to_numpy(), upper.loc[origin].to_numpy(),
                        alpha=0.2, color="tab:blue", label="band")
    ax.plot(deliv, predictions.loc[origin].to_numpy(), color="tab:blue", marker=".",
            ms=4, lw=1.2, label=f"forecast issued {origin:%Y-%m-%d %H:%M}")
    ax.axvline(origin, color="grey", ls="--", lw=1)
    ax.set_ylabel(predictions.columns[0].rsplit("_h_", 1)[0])
    ax.set_title(f"Forecast trajectory issued at {origin:%Y-%m-%d %H:%M}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def _issue_origins(predictions: pd.DataFrame, hour: int, minute: int = 0,
                   start=None, end=None) -> pd.DatetimeIndex:
    idx = predictions.index
    o = idx[(idx.hour == hour) & (idx.minute == minute)]
    if start is not None or end is not None:
        o = pd.Series(0, index=o).loc[start:end].index
    if len(o) == 0:
        raise ValueError(f"no forecasts issued at {hour:02d}:{minute:02d}")
    return o


def plot_issue_timeseries(actual: pd.Series, predictions: pd.DataFrame, hour: int = 8,
                          minute: int = 0, start=None, end=None, ax=None):
    """Stitch together every forecast *issued at* ``hour:minute`` into one continuous
    operational series vs. measured. Each issue covers the next H steps (e.g. an
    08:00 day-ahead PV forecast covers 08:00→08:00 D+1), so consecutive daily issues
    abut into a gapless line — the forecast you'd actually have run on."""
    dt = predictions.index.to_series().diff().median()
    origins = _issue_origins(predictions, hour, minute, start, end)
    pieces = [pd.Series(predictions.loc[o].to_numpy(),
                        index=[o + (h + 1) * dt for h in range(predictions.shape[1])])
              for o in origins]
    fc = pd.concat(pieces).sort_index()
    fc = fc[~fc.index.duplicated(keep="last")]
    a = actual.loc[fc.index.min():fc.index.max()]
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(a.index, a.to_numpy(), color="black", lw=1.4, label="measured")
    ax.plot(fc.index, fc.to_numpy(), color="tab:blue", lw=1.0, alpha=0.85,
            label=f"forecast issued {hour:02d}:{minute:02d}")
    ax.set_ylabel(predictions.columns[0].rsplit("_h_", 1)[0])
    ax.set_title(f"All {hour:02d}:{minute:02d} forecasts (stitched) vs measured")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_issue_profile(actual: pd.Series, predictions: pd.DataFrame, hour: int = 8,
                       minute: int = 0, start=None, end=None, band=(0.05, 0.95), ax=None):
    """Average the forecast issued at ``hour:minute`` and the matching measurements
    over all days → mean lead-time profile (e.g. the typical 08:00→08:00 D+1 PV day).
    x-axis is hours ahead of the issue time. ``band`` (e.g. ``(0.05, 0.95)``) shades
    the empirical quantile spread of the forecast **across days** (the day-to-day
    range, computed from the data — not the model's quantiles); pass None to hide it."""
    dt = predictions.index.to_series().diff().median()
    H = predictions.shape[1]
    origins = _issue_origins(predictions, hour, minute, start, end)
    fc = np.array([predictions.loc[o].to_numpy() for o in origins], dtype=float)        # (days, H)
    meas = np.array([actual.reindex([o + (h + 1) * dt for h in range(H)]).to_numpy()
                     for o in origins], dtype=float)
    lead_h = [(h + 1) * dt.total_seconds() / 3600 for h in range(H)]
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    if band is not None:
        lo = np.nanquantile(fc, band[0], axis=0)
        hi = np.nanquantile(fc, band[1], axis=0)
        ax.fill_between(lead_h, lo, hi, alpha=0.2, color="tab:blue",
                        label=f"forecast {int(band[0]*100)}–{int(band[1]*100)}% across days")
    ax.plot(lead_h, np.nanmean(meas, axis=0), color="black", lw=1.6, marker=".", label="measured (mean)")
    ax.plot(lead_h, np.nanmean(fc, axis=0), color="tab:blue", lw=1.4, marker=".", label="forecast (mean)")
    ax.set_xlabel(f"hours after {hour:02d}:{minute:02d}")
    ax.set_ylabel(predictions.columns[0].rsplit("_h_", 1)[0])
    ax.set_title(f"Mean {hour:02d}:{minute:02d} day-ahead profile (n={len(origins)} days)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_forecast_band(actual: pd.Series, median: pd.DataFrame, lower: pd.DataFrame,
                       upper: pd.DataFrame, step: int = 1, start=None, end=None, ax=None):
    """Measured vs. median forecast with a shaded uncertainty band, at one horizon
    ``step``. ``median``/``lower``/``upper`` are wide quantile frames (same shape,
    index = forecast origin, columns ``{endog}_h_1..H``) — e.g. the 0.5 / 0.1 / 0.9
    outputs of ``Lgbm.predict_quantiles``. All are shifted forward by ``step`` to
    align with the measured series at delivery time."""
    dt = median.index.to_series().diff().median()

    def _deliv(df):
        s = pd.Series(df.iloc[:, step - 1].to_numpy(), index=df.index + step * dt)
        return s if (start is None and end is None) else s.loc[start:end]

    med, lo, hi = _deliv(median), _deliv(lower), _deliv(upper)
    a = actual if (start is None and end is None) else actual.loc[start:end]
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(med.index, lo.to_numpy(), hi.to_numpy(), alpha=0.25,
                    color="tab:orange", label="uncertainty band")
    ax.plot(med.index, med.to_numpy(), color="tab:orange", lw=1.1, label="forecast (median)")
    ax.plot(a.index, a.to_numpy(), color="black", lw=1.5, label="measured")
    ax.set_ylabel(median.columns[step - 1].rsplit("_h_", 1)[0])
    ax.set_title(f"Measured vs. forecast + band — horizon step {step}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_timeseries_compare(actual: pd.Series, preds_by_model: dict[str, pd.DataFrame],
                            step: int = 1, start=None, end=None, ax=None):
    """Overlay several models' forecasts at one horizon ``step`` vs. the measured
    series (one line per model). ``preds_by_model`` maps a label to that model's
    wide prediction frame (as returned by ``evaluate``/``load_forecaster``)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    a = actual if (start is None and end is None) else actual.loc[start:end]
    ax.plot(a.index, a.to_numpy(), label="measured", lw=1.8, color="black",
            zorder=len(preds_by_model) + 2)
    ylabel = ""
    for name, preds in preds_by_model.items():
        dt = preds.index.to_series().diff().median()
        s = pd.Series(preds.iloc[:, step - 1].to_numpy(), index=preds.index + step * dt)
        sp = s if (start is None and end is None) else s.loc[start:end]
        ax.plot(sp.index, sp.to_numpy(), lw=1.0, alpha=0.85, label=name)
        ylabel = preds.columns[step - 1].rsplit("_h_", 1)[0]
    ax.set_ylabel(ylabel)
    ax.set_title(f"Measured vs. forecast — horizon step {step}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_learning_curve(model, ax=None):
    """Train vs. validation RMSE per boosting round for an XGB model.

    Pass a freshly-trained ``Xgb`` (its ``evals_result_`` holds the curves;
    ``validation_0`` = train, ``validation_1`` = the held-out validation tail).
    """
    ev = getattr(model, "evals_result_", None)
    if not ev:
        raise ValueError("model has no evals_result_ (train an Xgb with early stopping first)")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    metric = next(iter(ev["validation_0"]))            # e.g. "rmse"
    ax.plot(ev["validation_0"][metric], label="train")
    ax.plot(ev["validation_1"][metric], label="validation")
    bi = getattr(model, "best_iteration_", None)
    if bi is not None:
        ax.axvline(bi, ls="--", color="grey", lw=1, label=f"best @ {bi}")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel(metric.upper())
    ax.set_title("XGB learning curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax
