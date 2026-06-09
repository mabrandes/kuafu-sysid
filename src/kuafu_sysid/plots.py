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
