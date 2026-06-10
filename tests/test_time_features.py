import pandas as pd

from kuafu_sysid.features import (
    FeatureSpec, add_time_features, feature_hash, resolve_time_features,
)


def _idx(n=200, freq="h"):
    return pd.date_range("2025-01-01", periods=n, freq=freq, tz="Europe/Zurich")


def test_resolve_legacy_enabled_turns_on_cyclical_sets():
    r = resolve_time_features({"enabled": True, "holidays_country": "CH"})
    assert r["time_of_day"] and r["day_of_week"] and r["day_of_year"]
    assert r["holidays_country"] == "CH" and r["encoding"] == "cyclical"


def test_per_component_toggle_selects_only_requested():
    tf = add_time_features(_idx(), {"day_of_week": True})
    assert list(tf.columns) == ["dow_sin", "dow_cos"]          # only weekday
    # nothing else
    assert not any(c.startswith("hour") or c.startswith("doy") for c in tf.columns)


def test_per_component_override_beats_enabled():
    # enabled turns all on, but explicit day_of_year:false removes seasonality
    tf = add_time_features(_idx(), {"enabled": True, "day_of_year": False})
    assert "hour_sin" in tf.columns and "dow_sin" in tf.columns
    assert "doy_sin" not in tf.columns


def test_onehot_encoding_columns():
    tf = add_time_features(_idx(), {"time_of_day": True, "day_of_week": True,
                                    "day_of_year": True, "encoding": "onehot"})
    assert "tod_0" in tf.columns and "tod_23" in tf.columns      # 24 hour dummies
    assert "dow_0" in tf.columns and "dow_6" in tf.columns       # 7 weekday dummies
    assert "month_1" in tf.columns and "month_12" in tf.columns  # 12 month dummies
    # each row sums to exactly one per group
    assert (tf[[f"tod_{h}" for h in range(24)]].sum(axis=1) == 1).all()


def test_holidays_only():
    tf = add_time_features(_idx(), {"holidays_country": "CH"})
    assert list(tf.columns) == ["is_holiday"]


def test_invalid_encoding_raises():
    import pytest
    with pytest.raises(ValueError, match="encoding"):
        add_time_features(_idx(), {"time_of_day": True, "encoding": "bogus"})


def test_feature_hash_sensitive_to_time_feature_choices():
    spec = FeatureSpec(endog="y")
    base = dict(spec=spec, lag=4, horizon=3, dt_min=60)
    h_cyc = feature_hash(time_features={"time_of_day": True, "encoding": "cyclical"}, **base)
    h_oh = feature_hash(time_features={"time_of_day": True, "encoding": "onehot"}, **base)
    h_dow = feature_hash(time_features={"day_of_week": True, "encoding": "cyclical"}, **base)
    assert len({h_cyc, h_oh, h_dow}) == 3   # encoding and component both change the hash
