"""Typed config dataclasses for training and model selection (YAML-backed)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from kuafu_sysid.features import FeatureSpec
from kuafu_sysid.models import MODEL_REGISTRY


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
        return cls(
            target=raw["target"], spec=spec, data_path=d["path"],
            lag=raw["lag"], horizon=int(raw["horizon"]), dt_min=raw.get("dt_min"),
            train_start=train.get("start"), train_end=train.get("end"),
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
        roles = {name: RoleSpec(**spec) for name, spec in raw["roles"].items()}
        return cls(store_root=(raw.get("store", {}) or {}).get("root", "models"), roles=roles)
