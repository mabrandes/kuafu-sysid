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
