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
    # combined comparison plots (one per target, all models overlaid)
    assert "t_horizon_compare.png" in pngs
    assert "t_timeseries_compare.png" in pngs


def test_train_resamples_before_features(tmp_path):
    # native data is hourly; resample_min=180 -> 3h grid, so steps_per_day=8
    data_path = _make_parquet(tmp_path)   # 400 hourly rows
    cfg = TrainConfig(
        target="t", spec=FeatureSpec(endog="y", exog=("a",)),
        data_path=str(data_path), lag=8, horizon=2, dt_min=None,
        train_start="2025-01-01", train_end="2025-01-15", split=-0.2,
        models=["Linear"], resample_min=180, resample_agg="mean",
        time_features={"enabled": False, "holidays_country": None},
        store_root=str(tmp_path / "models"), save=True,
    )
    train(cfg, save_plots=False)
    import json, glob
    recipe = json.loads(open(glob.glob(str(tmp_path / "models" / "t" / "*_config.json"))[0]).read())
    assert recipe["dt_min"] == 180          # effective timestep is the resample target


def test_train_no_plots_when_disabled(tmp_path):
    data_path = _make_parquet(tmp_path)
    train(_cfg(tmp_path, data_path), save_plots=False)
    assert not list((tmp_path / "models" / "t").glob("*.png"))
