"""Common contract for all forecasting model adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


# `ABC` = Abstract Base Class: this defines only the CONTRACT every model adapter
# must satisfy (fit/predict/save/load), not any behaviour. It deliberately has no
# `__init__`: an ABC isn't constructed directly, and each concrete adapter takes
# DIFFERENT constructor args (Ridge alpha, XGB n_estimators, persistence period, ...),
# so there's no meaningful shared __init__ to put here. Subclasses define their own.
class Forecaster(ABC):
    """Adapter interface. predict() returns shape (len(X), horizon)."""

    # Class-level (not instance) attributes: they're fixed per model FAMILY, not
    # per trained model. `EXT` is the on-disk artefact extension the store uses to
    # name and find the file — "json" for XGB's native format, "joblib" otherwise.
    # A subclass overrides it by redeclaring `EXT = "json"`.
    EXT: str = "joblib"          # artefact file extension
    requires_fit: bool = True    # baselines (e.g. persistence) set this False

    @abstractmethod
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "Forecaster": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame, endog: pd.Series | None = None) -> np.ndarray:
        """Return (len(X), horizon). ``endog`` (the full target series) is used
        only by baselines (e.g. persistence); regression adapters ignore it."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    # `load` is a @classmethod because it's an ALTERNATIVE CONSTRUCTOR: you call it
    # when you DON'T yet have an instance — `Xgb.load(path)` reads the file and
    # returns a brand-new fitted object. (A normal method takes `self`, i.e. needs
    # an existing instance — impossible here, the instance is what we're creating.)
    # It receives the class as `cls`, so `MODEL_REGISTRY[method].load(...)`
    # reconstructs the right subclass. `save` stays an instance method: it acts on
    # an already-trained model (`self`).
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Forecaster": ...

    def feature_importances(self) -> "pd.Series | None":
        """Importance per feature (indexed by column name), or None if the model
        has none (e.g. persistence baselines). Tree adapters override this."""
        return None
