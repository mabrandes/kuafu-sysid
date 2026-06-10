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


def test_xgb_uses_validation_early_stopping():
    X, Y = _xy(n=300)
    m = Xgb(n_estimators=500).fit(X, Y)
    # validation tail was held out and monitored -> learning curves + best_iteration
    assert m.evals_result_ is not None
    assert {"validation_0", "validation_1"} <= set(m.evals_result_)  # [train, validation]
    assert m.best_iteration_ is not None and m.best_iteration_ < 500


def test_lgbm_uses_validation_early_stopping():
    X, Y = _xy(n=300)
    m = Lgbm(n_estimators=500).fit(X, Y)
    assert m.best_iteration_ is not None and m.best_iteration_ < 500


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
    endog = pd.Series(np.arange(10.0), index=idx)
    X = pd.DataFrame(index=idx)  # persistence ignores X, reads endog
    m = Persistence(period_steps=1, horizon=3, endog="y")
    out = m.predict(X, endog=endog)
    assert out.shape == (10, 3)
    assert np.array_equal(out[5], [5.0, 5.0, 5.0])  # origin value repeated across horizon


def test_persistence_prev_day_uses_offset():
    idx = pd.date_range("2025-01-01", periods=50, freq="h", tz="Europe/Zurich")
    endog = pd.Series(np.arange(50.0), index=idx)  # endog[i] == i
    m = Persistence(period_steps=24, horizon=2, endog="y")
    out = m.predict(pd.DataFrame(index=idx), endog=endog)
    # at origin position 30: h=1 -> t+1-24 = pos 7; h=2 -> t+2-24 = pos 8
    assert np.array_equal(out[30], [7.0, 8.0])


def test_persistence_prev_week_works_without_huge_lag():
    # weekly period (168 h) with no lag columns at all — reads endog directly
    idx = pd.date_range("2025-01-01", periods=400, freq="h", tz="Europe/Zurich")
    endog = pd.Series(np.arange(400.0), index=idx)
    m = Persistence(period_steps=168, horizon=2, endog="y")
    out = m.predict(pd.DataFrame(index=idx), endog=endog)
    # at origin position 200: h=1 -> 200+1-168 = pos 33; h=2 -> pos 34
    assert np.array_equal(out[200], [33.0, 34.0])
