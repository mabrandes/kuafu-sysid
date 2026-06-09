"""Feature engineering for direct multi-step forecasting."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    """Names the columns that drive a forecast (all from the source frame)."""
    endog: str                          # target column
    exog: tuple[str, ...] = ()          # current-step drivers (no lag)
    exog_with_lag: tuple[str, ...] = () # exog that also get lagged copies
    forecast_exog: tuple[str, ...] = () # known-ahead exog (aligned to future slots)


def normalize_lags(lag) -> tuple[int, ...]:
    """int N -> (0..N-1); sequence -> sorted unique ints."""
    if isinstance(lag, (int, np.integer)):
        return tuple(range(int(lag)))
    return tuple(sorted({int(x) for x in lag}))


def feature_hash(spec: FeatureSpec, lag, horizon: int, dt_min, time_features: dict) -> str:
    """Stable 6-char sha1 over the full feature recipe."""
    payload = {
        "endog": spec.endog,
        "exog": list(spec.exog),
        "exog_with_lag": list(spec.exog_with_lag),
        "forecast_exog": list(spec.forecast_exog),
        "lag": list(normalize_lags(lag)),
        "horizon": int(horizon),
        "dt_min": dt_min,
        "time_features": {
            "enabled": bool(time_features.get("enabled", False)),
            "holidays_country": time_features.get("holidays_country"),
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:6]
