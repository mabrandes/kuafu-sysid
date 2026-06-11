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
    # per-model: feature importance (tree-only) + XGB learning curve
    assert any(n.startswith("XGB_") and n.endswith("_importance.png") for n in pngs)
    assert any(n.startswith("XGB_") and n.endswith("_learning_curve.png") for n in pngs)
    assert not any(n.startswith("Persistence") and n.endswith("_importance.png") for n in pngs)
    # combined comparison plots (named by feature_hash + train window, all models overlaid)
    assert any(n.endswith("_horizon_compare.png") for n in pngs)
    assert any(n.endswith("_timeseries_compare.png") for n in pngs)
    assert not any(n.startswith("t_") for n in pngs)   # no longer prefixed by target


def test_train_downsamples_when_dt_min_coarser(tmp_path):
    # native data is hourly; dt_min=180 (coarser) -> downsample to a 3h grid
    data_path = _make_parquet(tmp_path)   # 400 hourly rows
    cfg = TrainConfig(
        target="t", spec=FeatureSpec(endog="y", exog=("a",)),
        data_path=str(data_path), lag=8, horizon=2, dt_min=180,
        train_start="2025-01-01", train_end="2025-01-15", split=-0.2,
        models=["Linear"], resample_agg="mean",
        time_features={"enabled": False, "holidays_country": None},
        store_root=str(tmp_path / "models"), save=True,
    )
    train(cfg, save_plots=False)
    import json, glob
    recipe = json.loads(open(glob.glob(str(tmp_path / "models" / "t" / "*_config.json"))[0]).read())
    assert recipe["dt_min"] == 180          # working resolution is the (coarser) dt_min


def test_train_no_plots_when_disabled(tmp_path):
    data_path = _make_parquet(tmp_path)
    train(_cfg(tmp_path, data_path), save_plots=False)
    assert not list((tmp_path / "models" / "t").glob("*.png"))
