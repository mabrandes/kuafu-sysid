import numpy as np
import pandas as pd

from kuafu_sysid.config import SelectionConfig, RoleSpec, TrainConfig
from kuafu_sysid.features import FeatureSpec
from kuafu_sysid.evaluate import load_forecaster, evaluate
from kuafu_sysid.train import train


def _setup(tmp_path, clip_min=None):
    idx = pd.date_range("2025-01-01", periods=400, freq="h", tz="Europe/Zurich")
    df = pd.DataFrame({"y": np.sin(np.arange(400) / 5.0), "a": np.cos(np.arange(400) / 5.0)}, index=idx)
    data_path = tmp_path / "data.parquet"
    df.to_parquet(data_path)
    cfg = TrainConfig(
        target="t", spec=FeatureSpec(endog="y", exog=("a",)),
        data_path=str(data_path), lag=24, horizon=4, dt_min=60,
        train_start="2025-01-01", train_end="2025-01-15", split=-0.2,
        models=["Linear"], time_features={"enabled": False, "holidays_country": None},
        clip_min=clip_min, store_root=str(tmp_path / "models"), save=True,
    )
    train(cfg)
    sel = SelectionConfig(store_root=str(tmp_path / "models"), roles={
        "price": RoleSpec(target="t", method="Linear", train_start="2025-01-01", train_end="2025-01-15"),
    })
    return df, sel


def test_clip_min_floors_predictions(tmp_path):
    # sine target spans negatives; clip_min=0 must floor all forecasts at 0
    df, sel = _setup(tmp_path, clip_min=0.0)
    out = load_forecaster(sel, "price").predict(df).to_numpy()
    out = out[~np.isnan(out)]
    assert (out >= 0).all() and out.min() == 0.0   # some were clipped


def test_load_forecaster_predict_wide(tmp_path):
    df, sel = _setup(tmp_path)
    fc = load_forecaster(sel, "price")
    out = fc.predict(df)
    assert list(out.columns) == ["y_h_1", "y_h_2", "y_h_3", "y_h_4"]
    assert out.index.equals(df.index)


def test_evaluate_returns_metrics(tmp_path):
    df, sel = _setup(tmp_path)
    res = evaluate(sel, "price", df)
    assert list(res.metrics.index) == [1, 2, 3, 4]
    assert (res.metrics["rmse"] >= 0).all()
    assert "r2" in res.metrics.columns


def test_evaluate_metric_window_scores_subset(tmp_path):
    df, sel = _setup(tmp_path)
    cut = df.index[len(df) // 2]
    res = evaluate(sel, "price", df, start=cut)          # metrics only after `cut`
    assert int(res.metrics["n"].iloc[0]) < len(df)        # fewer rows scored
    assert res.predictions.index.equals(df.index)         # predictions still cover all data
