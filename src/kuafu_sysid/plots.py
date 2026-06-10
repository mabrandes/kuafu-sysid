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
