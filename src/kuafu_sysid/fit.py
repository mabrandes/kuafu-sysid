"""Model fitting entry points (placeholder skeleton).

Real RC / state-space / regression identification goes here.
"""

from __future__ import annotations

import pandas as pd


def fit_model(data: pd.DataFrame, target: str, **kwargs):
    """Fit a system model to ``data`` predicting ``target``.

    TODO: implement concrete identification (RC, state-space, regression).
    """
    raise NotImplementedError("fit_model not yet implemented")
