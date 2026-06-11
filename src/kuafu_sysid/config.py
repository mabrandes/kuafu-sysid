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


# `@dataclass` auto-generates the boilerplate (__init__, __repr__, __eq__) from the
# class-level field annotations below. So `TrainConfig(target=..., spec=..., ...)`
# just works without us writing an __init__, and `print(cfg)` shows every field.
# Fields with a value (e.g. `dt_min: ... = None`) are optional with that default;
# fields without one (target, spec, data_path, lag, horizon) are required.
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
    # `field(default_factory=...)` is required for MUTABLE defaults (list/dict): a
    # plain `models: list = []` would share ONE list across all instances. The
    # factory is called per-instance to give each its own fresh list/dict.
    models: list[str] = field(default_factory=list)
    time_features: dict = field(default_factory=lambda: {"enabled": False, "holidays_country": None})
    quantiles: tuple = ()       # LGBM uncertainty band, e.g. (0.1, 0.9); 0.5 always added
    linear_cv: bool = True      # Linear: cross-validate alpha (RidgeCV) or use a fixed Ridge
    clip_min: float | None = None   # clamp predictions to >= clip_min (e.g. 0 for PV); None = off
    store_root: str = "models"
    save: bool = True

    # `@classmethod` makes this an ALTERNATIVE CONSTRUCTOR: it receives the class
    # itself as `cls` (not an instance `self`), and is called on the class —
    # `TrainConfig.from_yaml("cfg.yaml")` — to build and return a new instance.
    # (A normal method takes `self` and needs an instance to call.) Using `cls`
    # (rather than hard-coding `TrainConfig`) means subclasses get the right type.
    # The return annotation is the string "TrainConfig" because the class isn't
    # fully defined yet at read time (forward reference; `from __future__ import
    # annotations` at the top makes all annotations lazy strings anyway).
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
        # `cls(...)` == `TrainConfig(...)` here — calls the dataclass-generated
        # __init__ with the parsed values and returns the new instance.
        return cls(
            target=raw["target"], spec=spec, data_path=d["path"],
            lag=lag, horizon=int(raw["horizon"]), dt_min=raw.get("dt_min"),
            train_start=_datestr(train.get("start")), train_end=_datestr(train.get("end")),
            split=float(train.get("split", -0.1)), models=models,
            time_features=raw.get("time_features", {"enabled": False, "holidays_country": None}),
            quantiles=tuple(raw.get("quantiles") or ()),
            linear_cv=bool(raw.get("linear_cv", True)),
            clip_min=raw.get("clip_min"),
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
            # `**spec` unpacks the dict into keyword arguments, i.e.
            # RoleSpec(target=..., method=..., ...). Keys must match field names;
            # missing optional keys fall back to the dataclass defaults.
            rs = RoleSpec(**spec)
            rs.train_start = _datestr(rs.train_start)
            rs.train_end = _datestr(rs.train_end)
            roles[name] = rs
        return cls(store_root=(raw.get("store", {}) or {}).get("root", "models"), roles=roles)
