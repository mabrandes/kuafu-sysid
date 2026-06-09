"""Per-horizon forecast error metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_horizon_metrics(Y: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """RMSE / MAE / bias for each horizon step (1-indexed). NaN rows ignored."""
    actual = Y.to_numpy(float)
    err = pred - actual
    rows = []
    for h in range(actual.shape[1]):
        e = err[:, h]
        e = e[~np.isnan(e)]
        rows.append({
            "horizon": h + 1,
            "rmse": float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan,
            "mae": float(np.mean(np.abs(e))) if e.size else np.nan,
            "bias": float(np.mean(e)) if e.size else np.nan,
            "n": int(e.size),
        })
    return pd.DataFrame(rows).set_index("horizon")
