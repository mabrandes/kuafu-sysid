import numpy as np
import pandas as pd

from kuafu_sysid.features import FeatureSpec, add_time_features, build_features


def _frame(n=200, freq="h"):
    idx = pd.date_range("2025-01-01", periods=n, freq=freq, tz="Europe/Zurich")
    rng = np.arange(n, dtype=float)
    return pd.DataFrame({"y": rng, "a": rng * 2, "b": rng + 1}, index=idx)


def test_add_time_features_columns():
    tf = add_time_features(_frame().index, {"enabled": True, "holidays_country": "CH"})
    for c in ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos", "is_holiday"]:
        assert c in tf.columns
    assert tf["is_holiday"].iloc[0] == 1  # 2025-01-01 is a CH holiday


def test_build_features_shapes_and_columns():
    df = _frame()
    spec = FeatureSpec(endog="y", exog=("b",), exog_with_lag=("a",), forecast_exog=("b",))
    X, Y = build_features(df, spec, lag=3, horizon=2, dt_min=60,
                          time_features={"enabled": True, "holidays_country": "CH"})
    assert list(Y.columns) == ["y_h_1", "y_h_2"]
    for c in ["y_lag_0", "y_lag_2", "a_lag_0", "a_lag_2", "b", "b_fc_0", "b_fc_1", "hour_sin"]:
        assert c in X.columns
    assert X.index.equals(Y.index)


def test_build_features_target_alignment():
    df = _frame(freq="h")
    spec = FeatureSpec(endog="y")
    X, Y = build_features(df, spec, lag=1, horizon=2, dt_min=60,
                          time_features={"enabled": False, "holidays_country": None})
    pos0 = df.index.get_loc(Y.index[0])
    assert Y["y_h_1"].iloc[0] == df["y"].iloc[pos0 + 1]
    assert Y["y_h_2"].iloc[0] == df["y"].iloc[pos0 + 2]
