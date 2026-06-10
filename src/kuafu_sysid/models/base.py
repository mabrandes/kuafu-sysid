"""Common contract for all forecasting model adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class Forecaster(ABC):
    """Adapter interface. predict() returns shape (len(X), horizon)."""

    EXT: str = "joblib"          # artefact file extension
    requires_fit: bool = True    # baselines set False

    @abstractmethod
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "Forecaster": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame, endog: pd.Series | None = None) -> np.ndarray:
        """Return (len(X), horizon). ``endog`` (the full target series) is used
        only by baselines (e.g. persistence); regression adapters ignore it."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Forecaster": ...
