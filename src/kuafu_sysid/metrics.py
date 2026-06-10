"""Per-horizon forecast error metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_horizon_metrics(Y: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """RMSE / MAE / R² / bias for each horizon step (1-indexed). NaN rows ignored.

    R² = 1 − SS_res/SS_tot (fraction of variance explained; 1 = perfect, 0 = no
    better than predicting the mean, <0 = worse than the mean).
    """
    actual = Y.to_numpy(float)
    err = pred - actual
    rows = []
    for h in range(actual.shape[1]):
        m = ~np.isnan(err[:, h])
        e = err[m, h]
        a = actual[m, h]
        ss_tot = float(np.sum((a - a.mean()) ** 2)) if e.size else 0.0
        r2 = 1.0 - float(np.sum(e ** 2)) / ss_tot if ss_tot > 0 else np.nan
        rows.append({
            "horizon": h + 1,
            "rmse": float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan,
            "mae": float(np.mean(np.abs(e))) if e.size else np.nan,
            "r2": r2,
            "bias": float(np.mean(e)) if e.size else np.nan,
            "n": int(e.size),
        })
    return pd.DataFrame(rows).set_index("horizon")
