import numpy as np
import pandas as pd

from kuafu_sysid.metrics import per_horizon_metrics


def test_per_horizon_metrics_perfect():
    Y = pd.DataFrame({"y_h_1": [1.0, 2, 3], "y_h_2": [4.0, 5, 6]})
    pred = Y.to_numpy()
    m = per_horizon_metrics(Y, pred)
    assert list(m.index) == [1, 2]                # horizon steps
    assert (m["rmse"] == 0).all() and (m["mae"] == 0).all()


def test_per_horizon_metrics_known_error():
    Y = pd.DataFrame({"y_h_1": [0.0, 0.0]})
    pred = np.array([[1.0], [3.0]])               # errors 1 and 3
    m = per_horizon_metrics(Y, pred)
    assert np.isclose(m.loc[1, "mae"], 2.0)
    assert np.isclose(m.loc[1, "rmse"], np.sqrt((1 + 9) / 2))
    assert np.isclose(m.loc[1, "bias"], 2.0)      # mean(pred - actual)
