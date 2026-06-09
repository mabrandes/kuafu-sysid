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


def add_time_features(index: pd.DatetimeIndex, holidays_country: str | None) -> pd.DataFrame:
    """Cyclical hour/day-of-year/day-of-week features + optional holiday flag."""
    idx = pd.DatetimeIndex(index)
    hour = idx.hour + idx.minute / 60.0
    doy = idx.dayofyear
    dow = idx.dayofweek
    out = pd.DataFrame(index=idx)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    if holidays_country:
        import holidays as _hol
        cal = _hol.country_holidays(holidays_country, years=sorted(set(idx.year)))
        out["is_holiday"] = idx.normalize().isin(pd.to_datetime(list(cal.keys())).tz_localize(idx.tz)).astype(int)
    return out


def build_features(df: pd.DataFrame, spec: FeatureSpec, lag, horizon: int, dt_min,
                   time_features: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X feature matrix, Y target matrix) for direct multi-step regression.

    Columns produced:
      Y: {endog}_h_1 .. _h_H
      X: {endog}_lag_k, {exog_with_lag}_lag_k, current {exog}, {forecast_exog}_fc_h, time features
    Rows are NOT dropped here; downstream model adapters handle NaN per their kind.
    """
    lags = normalize_lags(lag)
    X = pd.DataFrame(index=df.index)

    # endog lags
    for k in lags:
        X[f"{spec.endog}_lag_{k}"] = df[spec.endog].shift(k)
    # exog_with_lag lags
    for col in spec.exog_with_lag:
        for k in lags:
            X[f"{col}_lag_{k}"] = df[col].shift(k)
    # current-step exog
    for col in spec.exog:
        X[col] = df[col]
    # known-ahead exog aligned to future slots
    for col in spec.forecast_exog:
        for h in range(horizon):
            X[f"{col}_fc_{h}"] = df[col].shift(-h)
    # time features
    if time_features.get("enabled"):
        X = X.join(add_time_features(df.index, time_features.get("holidays_country")))

    # targets
    Y = pd.DataFrame(index=df.index)
    for h in range(1, horizon + 1):
        Y[f"{spec.endog}_h_{h}"] = df[spec.endog].shift(-h)

    return X, Y
