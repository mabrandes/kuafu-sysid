"""Plain-matplotlib evaluation plots (projects can restyle)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_error_by_horizon(metrics_by_method: dict[str, pd.DataFrame], metric: str = "rmse", ax=None):
    """One line per method: error vs horizon step."""
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


def plot_timeseries(actual: pd.Series, predictions: pd.DataFrame, step: int = 1,
                    start=None, end=None, ax=None):
    """Plot measured vs. forecast over calendar (delivery) time for one horizon step.

    ``actual``       — the endog series (e.g. ``df[endog]``).
    ``predictions``  — wide forecast frame from ``evaluate()`` / ``load_forecaster``
                       (index = forecast origin, columns ``{endog}_h_1..H``).
    ``step``         — which horizon step to show (1 = next step ahead).
    The step-``h`` forecast issued at origin ``t`` is for delivery time ``t + h``,
    so the forecast index is shifted forward by ``h`` steps to line up with actuals.
    ``start``/``end`` zoom to a date window (recommended — full series is dense).
    """
    col = predictions.columns[step - 1]
    dt = predictions.index.to_series().diff().median()
    pred = pd.Series(predictions[col].to_numpy(), index=predictions.index + step * dt)
    a = actual if (start is None and end is None) else actual.loc[start:end]
    p = pred if (start is None and end is None) else pred.loc[start:end]
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    ax.plot(a.index, a.to_numpy(), label="measured", lw=1.0)
    ax.plot(p.index, p.to_numpy(), label=f"forecast (h={step})", lw=1.0, alpha=0.8)
    ax.set_ylabel(col.rsplit("_h_", 1)[0])
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
