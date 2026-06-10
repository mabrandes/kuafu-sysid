import textwrap

import pytest

from kuafu_sysid.config import TrainConfig, SelectionConfig


def test_train_config_from_yaml(tmp_path):
    p = tmp_path / "train.yaml"
    p.write_text(textwrap.dedent("""
        target: da_price
        data:
          path: data/processed/entsoe_CH.parquet
          endog: CH_da_price_eur_per_mwh
          exog: [CH_load_mw]
          exog_with_lag: [CH_load_mw]
          forecast_exog: []
        dt_min: 60
        lag: 48
        horizon: 24
        train: {start: 2025-01-01, end: 2025-02-01, split: -0.1}
        models: [Linear, XGB]
        time_features: {enabled: true, holidays_country: CH}
        store: {root: models}
        save: true
    """), encoding="utf-8")
    cfg = TrainConfig.from_yaml(p)
    assert cfg.target == "da_price"
    assert cfg.spec.endog == "CH_da_price_eur_per_mwh"
    assert cfg.models == ["Linear", "XGB"]
    assert cfg.horizon == 24
    # YAML parses bare dates to datetime.date; they must be normalised to ISO strings
    assert cfg.train_start == "2025-01-01" and isinstance(cfg.train_start, str)


def test_lags_list_overrides_lag_count(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        "target: t\ndata: {path: x, endog: y}\nhorizon: 2\nlag: 96\n"
        "lags: [0, 1, 96, 672]\nmodels: [Linear]\n", encoding="utf-8")
    cfg = TrainConfig.from_yaml(p)
    assert cfg.lag == [0, 1, 96, 672]   # explicit list wins over lag: 96


def test_quantiles_parsed(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("target: t\ndata: {path: x, endog: y}\nhorizon: 2\nlag: 2\n"
                 "models: [LGBM]\nquantiles: [0.1, 0.9]\n", encoding="utf-8")
    assert TrainConfig.from_yaml(p).quantiles == (0.1, 0.9)


def test_empty_lags_falls_back_to_lag_count(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        "target: t\ndata: {path: x, endog: y}\nhorizon: 2\nlag: 96\n"
        "lags: []\nmodels: [Linear]\n", encoding="utf-8")
    cfg = TrainConfig.from_yaml(p)
    assert cfg.lag == 96                # empty list -> use the count


def test_train_config_rejects_unknown_model(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("target: t\ndata: {path: x, endog: y}\nhorizon: 2\nlag: 2\nmodels: [Nope]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown model"):
        TrainConfig.from_yaml(p)


def test_selection_config_from_yaml(tmp_path):
    p = tmp_path / "sel.yaml"
    p.write_text(textwrap.dedent("""
        store: {root: models}
        roles:
          da_price: {target: da_price, method: Linear, feature_hash: abc123,
                     train_start: 2025-01-01, train_end: 2025-02-01}
    """), encoding="utf-8")
    sel = SelectionConfig.from_yaml(p)
    assert sel.roles["da_price"].method == "Linear"
    assert sel.store_root == "models"
