import pytest

from kuafu_sysid.models import MODEL_REGISTRY, get_model
from kuafu_sysid.models.persistence import Persistence


def test_registry_has_all_methods():
    for name in ["Linear", "KNN", "XGB", "LGBM",
                 "Persistence_prev_step", "Persistence_prev_day", "Persistence_prev_week"]:
        assert name in MODEL_REGISTRY


def test_get_model_persistence_periods():
    ctx = dict(horizon=4, endog="y", steps_per_day=24, steps_per_week=168)
    assert get_model("Persistence_prev_step", **ctx).period_steps == 1
    assert get_model("Persistence_prev_day", **ctx).period_steps == 24
    assert get_model("Persistence_prev_week", **ctx).period_steps == 168


def test_get_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("Nope", horizon=1, endog="y")
