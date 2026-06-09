"""Model registry: method string -> Forecaster adapter."""
from __future__ import annotations

from kuafu_sysid.models.base import Forecaster
from kuafu_sysid.models.knn import Knn
from kuafu_sysid.models.lgbm import Lgbm
from kuafu_sysid.models.linear import Linear
from kuafu_sysid.models.persistence import Persistence
from kuafu_sysid.models.xgb import Xgb

MODEL_REGISTRY: dict[str, type[Forecaster]] = {
    "Linear": Linear,
    "KNN": Knn,
    "XGB": Xgb,
    "LGBM": Lgbm,
    "Persistence_prev_step": Persistence,
    "Persistence_prev_day": Persistence,
    "Persistence_prev_week": Persistence,
}

_PERSISTENCE_PERIOD = {
    "Persistence_prev_step": "step",
    "Persistence_prev_day": "day",
    "Persistence_prev_week": "week",
}


def get_model(method: str, *, horizon: int, endog: str,
              steps_per_day: int = 24, steps_per_week: int = 168, **params) -> Forecaster:
    """Instantiate an adapter. Persistence variants compute their period from
    steps_per_day / steps_per_week (caller passes these from dt_min)."""
    if method not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {method!r}. Choose from {sorted(MODEL_REGISTRY)}.")
    cls = MODEL_REGISTRY[method]
    if cls is Persistence:
        period = {"step": 1, "day": steps_per_day, "week": steps_per_week}[_PERSISTENCE_PERIOD[method]]
        return Persistence(period_steps=period, horizon=horizon, endog=endog)
    return cls(**params)
