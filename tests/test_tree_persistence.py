import numpy as np
import pandas as pd

from kuafu_sysid.models.xgb import Xgb
from kuafu_sysid.models.lgbm import Lgbm
from kuafu_sysid.models.persistence import Persistence


def _xy(n=200, p=4, h=3):
    X = pd.DataFrame(np.random.RandomState(0).randn(n, p), columns=[f"f{i}" for i in range(p)])
    Y = pd.DataFrame(np.random.RandomState(1).randn(n, h), columns=[f"y_h_{i+1}" for i in range(h)])
    return X, Y


def test_xgb_roundtrip(tmp_path):
    X, Y = _xy()
    m = Xgb(n_estimators=10).fit(X, Y)
    out = m.predict(X)
    assert out.shape == (len(X), 3)
    p = tmp_path / f"m.{Xgb.EXT}"
    m.save(p)
    assert np.allclose(out, Xgb.load(p).predict(X), atol=1e-5)


def test_lgbm_roundtrip(tmp_path):
    X, Y = _xy()
    m = Lgbm(n_estimators=10).fit(X, Y)
    out = m.predict(X)
    assert out.shape == (len(X), 3)
    p = tmp_path / f"m.{Lgbm.EXT}"
    m.save(p)
    assert np.allclose(out, Lgbm.load(p).predict(X), atol=1e-5)


def test_persistence_prev_step_repeats_last_value():
    idx = pd.date_range("2025-01-01", periods=10, freq="h", tz="Europe/Zurich")
    X = pd.DataFrame({"y_lag_0": np.arange(10.0)}, index=idx)
    m = Persistence(period_steps=1, horizon=3, endog="y")
    out = m.predict(X)
    assert out.shape == (10, 3)
    assert np.array_equal(out[5], [5.0, 5.0, 5.0])  # last value repeated across horizon


def test_persistence_prev_day_uses_offset_lag():
    idx = pd.date_range("2025-01-01", periods=50, freq="h", tz="Europe/Zurich")
    cols = {f"y_lag_{k}": np.full(50, float(k)) for k in range(24)}
    X = pd.DataFrame(cols, index=idx)
    m = Persistence(period_steps=24, horizon=2, endog="y")
    out = m.predict(X)
    # h=1 -> lag 23 (value 23), h=2 -> lag 22 (value 22)
    assert np.array_equal(out[0], [23.0, 22.0])
