import numpy as np
import pandas as pd

from kuafu_sysid.models.linear import Linear
from kuafu_sysid.models.knn import Knn


def _xy(n=120, p=4, h=3, with_nan=False):
    X = pd.DataFrame(np.random.RandomState(0).randn(n, p), columns=[f"f{i}" for i in range(p)])
    Y = pd.DataFrame(np.random.RandomState(1).randn(n, h), columns=[f"y_h_{i+1}" for i in range(h)])
    if with_nan:
        X.iloc[0, 0] = np.nan  # one incomplete row
    return X, Y


def test_linear_fit_predict_shape():
    X, Y = _xy()
    m = Linear().fit(X, Y)
    out = m.predict(X)
    assert out.shape == (len(X), Y.shape[1])


def test_linear_nan_row_predicts_nan():
    X, Y = _xy(with_nan=True)
    m = Linear().fit(X, Y)
    out = m.predict(X)
    assert np.isnan(out[0]).all()       # incomplete row -> NaN
    assert not np.isnan(out[1]).any()   # complete row -> value


def test_linear_save_load_roundtrip(tmp_path):
    X, Y = _xy()
    m = Linear().fit(X, Y)
    p = tmp_path / f"m.{Linear.EXT}"
    m.save(p)
    m2 = Linear.load(p)
    assert np.allclose(np.nan_to_num(m.predict(X)), np.nan_to_num(m2.predict(X)))


def test_knn_fit_predict_shape():
    X, Y = _xy()
    out = Knn(n_neighbors=5).fit(X, Y).predict(X)
    assert out.shape == (len(X), Y.shape[1])
