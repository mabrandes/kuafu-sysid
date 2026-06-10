"""Typed config dataclasses for training and model selection (YAML-backed)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from kuafu_sysid.features import FeatureSpec
from kuafu_sysid.models import MODEL_REGISTRY


def _datestr(v):
    """YAML parses bare ISO dates (2025-01-01) into datetime.date; normalise to
    an ISO string so downstream tz-aware index slicing and filename stems work."""
    if v is None:
        return None
    return v.isoformat() if hasattr(v, "isoformat") else str(v)


@dataclass
class TrainConfig:
    target: str
    spec: FeatureSpec
    data_path: str
    lag: object                 # int or list[int]
    horizon: int
    dt_min: int | None = None
    train_start: str | None = None
    train_end: str | None = None
    split: float = -0.1
    models: list[str] = field(default_factory=list)
    time_features: dict = field(default_factory=lambda: {"enabled": False, "holidays_country": None})
    store_root: str = "models"
    save: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        d = raw["data"]
        spec = FeatureSpec(
            endog=d["endog"],
            exog=tuple(d.get("exog", []) or []),
            exog_with_lag=tuple(d.get("exog_with_lag", []) or []),
            forecast_exog=tuple(d.get("forecast_exog", []) or []),
        )
        models = list(raw.get("models", []))
        for m in models:
            if m not in MODEL_REGISTRY:
                raise ValueError(f"Unknown model {m!r}. Choose from {sorted(MODEL_REGISTRY)}.")
        train = raw.get("train", {}) or {}
        # `lags` (explicit list) overrides `lag` (count -> 0..lag-1) when non-empty.
        lags = raw.get("lags") or []
        lag = list(lags) if lags else raw["lag"]
        return cls(
            target=raw["target"], spec=spec, data_path=d["path"],
            lag=lag, horizon=int(raw["horizon"]), dt_min=raw.get("dt_min"),
            train_start=_datestr(train.get("start")), train_end=_datestr(train.get("end")),
            split=float(train.get("split", -0.1)), models=models,
            time_features=raw.get("time_features", {"enabled": False, "holidays_country": None}),
            store_root=(raw.get("store", {}) or {}).get("root", "models"),
            save=bool(raw.get("save", True)),
        )


@dataclass
class RoleSpec:
    target: str
    method: str
    feature_hash: str | None = None
    train_start: str | None = None
    train_end: str | None = None


@dataclass
class SelectionConfig:
    store_root: str
    roles: dict[str, RoleSpec]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SelectionConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        roles = {}
        for name, spec in raw["roles"].items():
            rs = RoleSpec(**spec)
            rs.train_start = _datestr(rs.train_start)
            rs.train_end = _datestr(rs.train_end)
            roles[name] = rs
        return cls(store_root=(raw.get("store", {}) or {}).get("root", "models"), roles=roles)
