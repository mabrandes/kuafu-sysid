"""kuafu_sysid — generic, config-driven forecasting framework."""

__version__ = "0.2.0"

from kuafu_sysid.config import SelectionConfig, TrainConfig
from kuafu_sysid.evaluate import EvalResult, FittedForecaster, evaluate, load_forecaster
from kuafu_sysid.features import FeatureSpec, build_features, feature_hash
from kuafu_sysid.models import MODEL_REGISTRY, get_model
from kuafu_sysid.plots import plot_error_by_horizon, plot_learning_curve, plot_timeseries
from kuafu_sysid.store import ModelStore
from kuafu_sysid.train import train

__all__ = [
    "TrainConfig", "SelectionConfig", "FeatureSpec", "build_features", "feature_hash",
    "MODEL_REGISTRY", "get_model", "ModelStore", "train",
    "evaluate", "load_forecaster", "FittedForecaster", "EvalResult",
    "plot_error_by_horizon", "plot_learning_curve", "plot_timeseries",
]
