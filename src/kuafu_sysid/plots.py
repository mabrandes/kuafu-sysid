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
