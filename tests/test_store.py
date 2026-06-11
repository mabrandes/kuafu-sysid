import numpy as np
import pandas as pd

from kuafu_sysid.models.linear import Linear
from kuafu_sysid.store import ModelStore


def _recipe():
    return {
        "target": "da_price", "method": "Linear", "endog": "y",
        "exog": [], "exog_with_lag": [], "forecast_exog": [],
        "lag": [0, 1], "horizon": 2, "dt_min": 60,
        "time_features": {"enabled": False, "holidays_country": None},
        "feature_hash": "abc123", "train_start": "2025-01-01", "train_end": "2025-02-01",
        "feature_columns": ["y_lag_0", "y_lag_1"], "target_columns": ["y_h_1", "y_h_2"],
    }


def _fitted():
    X = pd.DataFrame(np.random.RandomState(0).randn(50, 2), columns=["y_lag_0", "y_lag_1"])
    Y = pd.DataFrame(np.random.RandomState(1).randn(50, 2), columns=["y_h_1", "y_h_2"])
    return Linear().fit(X, Y), X


def test_save_creates_artefact_sidecar_and_latest(tmp_path):
    store = ModelStore(tmp_path)
    model, _ = _fitted()
    store.save("da_price", "Linear", model, _recipe(), "2025-01-01", "2025-02-01")
    d = tmp_path / "da_price"
    assert (d / "Linear_abc123_2025-01-01_2025-02-01.joblib").exists()
    assert (d / "Linear_abc123_2025-01-01_2025-02-01_config.json").exists()
    assert (d / "_latest.json").exists()


def test_load_by_hash_roundtrip(tmp_path):
    store = ModelStore(tmp_path)
    model, X = _fitted()
    store.save("da_price", "Linear", model, _recipe(), "2025-01-01", "2025-02-01")
    loaded, recipe = store.load("da_price", "Linear", feature_hash="abc123")
    assert recipe["feature_hash"] == "abc123"
    assert np.allclose(np.nan_to_num(model.predict(X)), np.nan_to_num(loaded.predict(X)))


def test_load_falls_back_to_latest(tmp_path):
    store = ModelStore(tmp_path)
    model, _ = _fitted()
    store.save("da_price", "Linear", model, _recipe(), "2025-01-01", "2025-02-01")
    _, recipe = store.load("da_price", "Linear")  # no hash/dates -> _latest.json
    assert recipe["method"] == "Linear"


def test_no_hash_uses_latest_even_with_dates(tmp_path):
    # blank feature_hash -> most recently trained model, NOT the alphabetical pick
    store = ModelStore(tmp_path)
    model, _ = _fitted()
    r1 = {**_recipe(), "feature_hash": "zzzzzz"}
    store.save("da_price", "Linear", model, r1, "2025-01-01", "2025-02-01")
    r2 = {**_recipe(), "feature_hash": "aaaaaa"}
    store.save("da_price", "Linear", model, r2, "2025-01-01", "2025-02-01")  # saved last = latest
    _, recipe = store.load("da_price", "Linear",
                           train_start="2025-01-01", train_end="2025-02-01")
    assert recipe["feature_hash"] == "aaaaaa"   # latest, not alphabetical (zzzzzz)
