"""Self-describing model store: hash-named artefacts + JSON sidecar + _latest.json."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from kuafu_sysid.models import MODEL_REGISTRY


class ModelStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _dir(self, target: str) -> Path:
        d = self.root / target
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _stem(method: str, feature_hash: str, start: str, end: str) -> str:
        return f"{method}_{feature_hash}_{start}_{end}"

    def save(self, target, method, model, recipe: dict, train_start, train_end) -> Path:
        d = self._dir(target)
        stem = self._stem(method, recipe["feature_hash"], train_start, train_end)
        model.save(d / f"{stem}.{model.EXT}")
        (d / f"{stem}_config.json").write_text(json.dumps(recipe, indent=2, default=str), encoding="utf-8")
        latest_path = d / "_latest.json"
        latest = json.loads(latest_path.read_text(encoding="utf-8")) if latest_path.exists() else {}
        latest[method] = stem
        latest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        latest_path.write_text(json.dumps(latest, indent=2), encoding="utf-8")
        return d / f"{stem}.{model.EXT}"

    def _resolve_stem(self, d: Path, method, feature_hash, train_start, train_end) -> str:
        # No hash pinned -> always the most recently trained model for this method
        # (the `_latest.json` pointer); train_start/train_end are ignored here.
        if not feature_hash:
            latest_path = d / "_latest.json"
            if not latest_path.exists():
                raise FileNotFoundError(f"No trained models in {d} (no _latest.json)")
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            if method not in latest:
                raise FileNotFoundError(f"No trained {method} in {d} (_latest.json)")
            return latest[method]
        # Hash pinned -> exact stem (with dates) or newest file for that hash.
        if train_start and train_end:
            return self._stem(method, feature_hash, train_start, train_end)
        hits = sorted(d.glob(f"{method}_{feature_hash}_*_config.json"))
        if not hits:
            raise FileNotFoundError(f"No model {method}/{feature_hash} in {d}")
        return hits[-1].name.replace("_config.json", "")

    def list_methods(self, target) -> list[str]:
        """Method names that have a trained model for ``target`` (from _latest.json)."""
        latest_path = self.root / target / "_latest.json"
        if not latest_path.exists():
            raise FileNotFoundError(f"No trained models in {self.root / target}")
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        return [m for m in latest if m != "updated_at"]

    def load(self, target, method, feature_hash=None, train_start=None, train_end=None):
        d = self.root / target
        stem = self._resolve_stem(d, method, feature_hash, train_start, train_end)
        recipe = json.loads((d / f"{stem}_config.json").read_text(encoding="utf-8"))
        cls = MODEL_REGISTRY[method]
        model = cls.load(d / f"{stem}.{cls.EXT}")
        return model, recipe
