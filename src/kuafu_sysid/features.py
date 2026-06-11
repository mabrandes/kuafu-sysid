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


#: canonical time-feature settings (all off by default)
_TF_DEFAULTS = {
    "time_of_day": False,    # hour of day
    "day_of_week": False,    # weekday
    "day_of_year": False,    # seasonality
    "holidays_country": None,
    "encoding": "cyclical",  # "cyclical" (sin/cos) or "onehot" (dummies)
}


def resolve_time_features(time_features: dict | None) -> dict:
    """Normalise a time_features config into the canonical settings dict.

    Per-component toggles (``time_of_day``, ``day_of_week``, ``day_of_year``),
    a ``holidays_country`` (or ``None``), and an ``encoding`` of ``"cyclical"``
    or ``"onehot"``. The legacy ``enabled: true`` turns the three cyclical
    components on (per-component keys still override it).
    """
    tf = dict(time_features or {})
    out = dict(_TF_DEFAULTS)
    if "enabled" in tf:
        on = bool(tf["enabled"])
        out["time_of_day"] = out["day_of_week"] = out["day_of_year"] = on
    for key in ("time_of_day", "day_of_week", "day_of_year"):
        if key in tf:
            out[key] = bool(tf[key])
    out["holidays_country"] = tf.get("holidays_country")
    out["encoding"] = tf.get("encoding", "cyclical")
    if out["encoding"] not in ("cyclical", "onehot"):
        raise ValueError(
            f"time_features.encoding must be 'cyclical' or 'onehot', got {out['encoding']!r}"
        )
    return out


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
        "time_features": resolve_time_features(time_features),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:6]


def resample_df(df: pd.DataFrame, minutes: int, agg="mean") -> pd.DataFrame:
    """Downsample a tz-aware time series to ``minutes`` resolution before feature
    building. ``agg`` is either a single rule applied to all columns (``"mean"``
    or ``"sum"``) or a ``{column: rule}`` map — columns not listed default to
    ``"mean"``. Use ``"sum"`` for accumulated quantities (e.g. PV energy per step)
    and ``"mean"`` for rates/levels (irradiance, temperature, price, load)."""
    r = df.resample(f"{int(minutes)}min")
    if isinstance(agg, str):
        return r.agg(agg)
    return r.agg({c: agg.get(c, "mean") for c in df.columns})


def add_time_features(index: pd.DatetimeIndex, time_features: dict | None) -> pd.DataFrame:
    """Calendar features for the chosen components and encoding.

    cyclical: ``hour_sin/cos`` (time_of_day), ``dow_sin/cos`` (day_of_week),
    ``doy_sin/cos`` (day_of_year).
    onehot:   ``tod_0..23`` (hour), ``dow_0..6`` (weekday), ``month_1..12``
    (a tractable stand-in for day_of_year). Holidays always add ``is_holiday``.
    """
    r = resolve_time_features(time_features)
    idx = pd.DatetimeIndex(index)
    cyc = r["encoding"] == "cyclical"
    cols: dict[str, np.ndarray] = {}

    if r["time_of_day"]:
        if cyc:
            hour = idx.hour + idx.minute / 60.0
            cols["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
            cols["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        else:
            for h in range(24):
                cols[f"tod_{h}"] = (idx.hour == h).astype(int)
    if r["day_of_week"]:
        dow = idx.dayofweek
        if cyc:
            cols["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
            cols["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        else:
            for d in range(7):
                cols[f"dow_{d}"] = (dow == d).astype(int)
    if r["day_of_year"]:
        if cyc:
            doy = idx.dayofyear
            cols["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
            cols["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        else:
            for m in range(1, 13):
                cols[f"month_{m}"] = (idx.month == m).astype(int)
    if r["holidays_country"]:
        import holidays as _hol
        cal = _hol.country_holidays(r["holidays_country"], years=sorted(set(idx.year)))
        cols["is_holiday"] = idx.normalize().isin(
            pd.to_datetime(list(cal.keys())).tz_localize(idx.tz)
        ).astype(int)
    return pd.DataFrame(cols, index=idx)  # built at once (no fragmentation)


def build_features(df: pd.DataFrame, spec: FeatureSpec, lag, horizon: int, dt_min,
                   time_features: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X feature matrix, Y target matrix) for direct multi-step regression.

    Columns produced:
      Y: {endog}_h_1 .. _h_H
      X: {endog}_lag_k, {exog_with_lag}_lag_k, current {exog}, {forecast_exog}_fc_h, time features
    Rows are NOT dropped here; downstream model adapters handle NaN per their kind.
    """
    lags = normalize_lags(lag)
    cols: dict[str, pd.Series] = {}

    for k in lags:                                  # endog lags
        cols[f"{spec.endog}_lag_{k}"] = df[spec.endog].shift(k)
    for col in spec.exog_with_lag:                  # exog lags
        for k in lags:
            cols[f"{col}_lag_{k}"] = df[col].shift(k)
    for col in spec.exog:                           # current-step exog
        cols[col] = df[col]
    for col in spec.forecast_exog:                  # known-ahead exog (future slots)
        for h in range(horizon):
            cols[f"{col}_fc_{h}"] = df[col].shift(-h)

    # build the matrix at once to avoid DataFrame fragmentation, then append time features
    X = pd.DataFrame(cols, index=df.index)
    tf = add_time_features(df.index, time_features)
    if not tf.empty:
        X = pd.concat([X, tf], axis=1)

    Y = pd.DataFrame(
        {f"{spec.endog}_h_{h}": df[spec.endog].shift(-h) for h in range(1, horizon + 1)},
        index=df.index,
    )
    return X, Y
