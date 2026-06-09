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
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Forecaster": ...
