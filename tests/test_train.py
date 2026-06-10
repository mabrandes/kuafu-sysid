import numpy as np
import pandas as pd

from kuafu_sysid.config import TrainConfig
from kuafu_sysid.features import FeatureSpec
from kuafu_sysid.train import train


def _make_parquet(tmp_path):
    idx = pd.date_range("2025-01-01", periods=400, freq="h", tz="Europe/Zurich")
    df = pd.DataFrame({"y": np.sin(np.arange(400) / 5.0), "a": np.cos(np.arange(400) / 5.0)}, index=idx)
    p = tmp_path / "data.parquet"
    df.to_parquet(p)
    return p


def _cfg(tmp_path, data_path):
    return TrainConfig(
        target="t", spec=FeatureSpec(endog="y", exog=("a",)),
        data_path=str(data_path), lag=24, horizon=4, dt_min=60,
        train_start="2025-01-01", train_end="2025-01-15", split=-0.2,
        models=["Linear", "Persistence_prev_day", "XGB"],
        time_features={"enabled": True, "holidays_country": "CH"},
        store_root=str(tmp_path / "models"), save=True,
    )


def test_train_writes_artefacts_and_returns_metrics(tmp_path):
    data_path = _make_parquet(tmp_path)
    metrics = train(_cfg(tmp_path, data_path))
    assert set(metrics) == {"Linear", "Persistence_prev_day", "XGB"}
    # Linear should beat naive prev_day RMSE at h=1 on a smooth signal
    assert metrics["Linear"].loc[1, "rmse"] < metrics["Persistence_prev_day"].loc[1, "rmse"]
    # artefacts exist
    d = tmp_path / "models" / "t"
    assert any(f.name.startswith("Linear_") and f.suffix == ".joblib" for f in d.iterdir())
    assert (d / "_latest.json").exists()
    # plots saved next to the models (save_plots default True)
    pngs = [f.name for f in d.glob("*.png")]
    assert any(n.startswith("XGB_") and n.endswith("_importance.png") for n in pngs)  # tree-only
    assert any(n.endswith("_horizon.png") for n in pngs)
    assert any(n.endswith("_timeseries.png") for n in pngs)
    assert not any(n.startswith("Persistence") and n.endswith("_importance.png") for n in pngs)


def test_train_no_plots_when_disabled(tmp_path):
    data_path = _make_parquet(tmp_path)
    train(_cfg(tmp_path, data_path), save_plots=False)
    assert not list((tmp_path / "models" / "t").glob("*.png"))
