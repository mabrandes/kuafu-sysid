def test_public_api_exports():
    import kuafu_sysid as ks
    for name in ["train", "evaluate", "load_forecaster", "TrainConfig",
                 "SelectionConfig", "ModelStore", "plot_error_by_horizon",
                 "plot_horizon_metrics", "plot_learning_curve", "plot_timeseries",
                 "plot_timeseries_compare", "plot_feature_importance", "plot_forecast_band"]:
        assert hasattr(ks, name)


def test_plot_timeseries_multistep_and_horizon_metrics():
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np, pandas as pd
    from kuafu_sysid import plot_timeseries, plot_horizon_metrics
    idx = pd.date_range("2025-01-01", periods=200, freq="h", tz="Europe/Zurich")
    actual = pd.Series(np.arange(200.0), index=idx)
    preds = pd.DataFrame({f"y_h_{h}": np.arange(200.0) for h in range(1, 5)}, index=idx)
    ax = plot_timeseries(actual, preds, step=[1, 4], start="2025-01-01", end="2025-01-05")
    assert ax is not None
    m = pd.DataFrame({"rmse": [1.0, 2], "mae": [.5, 1], "r2": [.9, .8]}, index=[1, 2])
    axes = plot_horizon_metrics(m, which=("rmse", "mae", "r2"))
    assert len(axes) == 3


def test_plot_forecast_origin():
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np, pandas as pd
    from kuafu_sysid import plot_forecast_origin
    idx = pd.date_range("2025-07-01", periods=200, freq="h", tz="Europe/Zurich")
    actual = pd.Series(np.arange(200.0), index=idx)
    preds = pd.DataFrame({f"y_h_{h}": np.arange(200.0) + h for h in range(1, 5)}, index=idx)
    ax = plot_forecast_origin(actual, preds, origin="2025-07-02 08:00")
    assert ax is not None


def test_plot_issue_timeseries_and_profile():
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np, pandas as pd
    from kuafu_sysid import plot_issue_timeseries, plot_issue_profile
    idx = pd.date_range("2025-07-01", periods=240, freq="h", tz="Europe/Zurich")  # 10 days
    actual = pd.Series(np.sin(np.arange(240) / 3.0), index=idx)
    preds = pd.DataFrame({f"y_h_{h}": np.sin((np.arange(240) + h) / 3.0) for h in range(1, 25)},
                         index=idx)  # 24h horizon
    assert plot_issue_timeseries(actual, preds, hour=8) is not None   # stitched
    assert plot_issue_profile(actual, preds, hour=8, band=None) is not None      # no band
    assert plot_issue_profile(actual, preds, hour=8, band=(0.05, 0.95)) is not None  # empirical band


def test_plot_timeseries_returns_axes():
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np, pandas as pd
    from kuafu_sysid import plot_timeseries
    idx = pd.date_range("2025-01-01", periods=48, freq="h", tz="Europe/Zurich")
    actual = pd.Series(np.arange(48.0), index=idx)
    preds = pd.DataFrame({"y_h_1": np.arange(48.0), "y_h_2": np.arange(48.0)}, index=idx)
    ax = plot_timeseries(actual, preds, step=2, start="2025-01-01", end="2025-01-02")
    assert ax is not None


def test_plot_error_by_horizon_returns_axes():
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    from kuafu_sysid import plot_error_by_horizon
    m = pd.DataFrame({"rmse": [1.0, 2.0], "mae": [0.5, 1.0]}, index=[1, 2])
    ax = plot_error_by_horizon({"Linear": m})
    assert ax is not None
