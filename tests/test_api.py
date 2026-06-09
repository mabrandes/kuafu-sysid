def test_public_api_exports():
    import kuafu_sysid as ks
    for name in ["train", "evaluate", "load_forecaster", "TrainConfig",
                 "SelectionConfig", "ModelStore", "plot_error_by_horizon"]:
        assert hasattr(ks, name)


def test_plot_error_by_horizon_returns_axes():
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    from kuafu_sysid import plot_error_by_horizon
    m = pd.DataFrame({"rmse": [1.0, 2.0], "mae": [0.5, 1.0]}, index=[1, 2])
    ax = plot_error_by_horizon({"Linear": m})
    assert ax is not None
