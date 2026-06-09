from kuafu_sysid.features import FeatureSpec, normalize_lags, feature_hash


def test_normalize_lags_int_and_sequence():
    assert normalize_lags(4) == (0, 1, 2, 3)
    assert normalize_lags([48, 0, 0, 24]) == (0, 24, 48)


def test_feature_hash_is_stable_and_6_chars():
    spec = FeatureSpec(endog="y", exog=("a",), exog_with_lag=(), forecast_exog=("a",))
    h1 = feature_hash(spec, lag=4, horizon=3, dt_min=60, time_features={"enabled": True, "holidays_country": "CH"})
    h2 = feature_hash(spec, lag=[0, 1, 2, 3], horizon=3, dt_min=60, time_features={"enabled": True, "holidays_country": "CH"})
    assert h1 == h2 and len(h1) == 6  # int 4 and [0..3] normalize identically


def test_feature_hash_changes_with_recipe():
    spec = FeatureSpec(endog="y", exog=("a",), exog_with_lag=(), forecast_exog=())
    base = feature_hash(spec, lag=4, horizon=3, dt_min=60, time_features={"enabled": False, "holidays_country": None})
    assert base != feature_hash(spec, lag=5, horizon=3, dt_min=60, time_features={"enabled": False, "holidays_country": None})
    assert base != feature_hash(spec, lag=4, horizon=4, dt_min=60, time_features={"enabled": False, "holidays_country": None})
