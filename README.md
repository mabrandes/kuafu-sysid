# kuafu-sysid

Generic, config-driven forecasting framework (direct multi-step regression)
shared across `kuafu_*` projects.

## Usage

```python
from kuafu_sysid import TrainConfig, SelectionConfig, train, evaluate, load_forecaster

# train + save every configured model
metrics = train(TrainConfig.from_yaml("config/sysid_train.yaml"))

# evaluate a pinned model on new data
sel = SelectionConfig.from_yaml("config/sysid_select.yaml")
res = evaluate(sel, "da_price", df)

# get forecasts for the optimization
fc = load_forecaster(sel, "da_price")
forecast = fc.predict(df)   # wide DataFrame: index=origin, cols=endog_h_1..H
```

Models: `Linear`, `KNN`, `XGB`, `LGBM`, `Persistence_prev_step/day/week`.
Models are saved under `<store.root>/<target>/` with a feature-hash name, a
`_config.json` sidecar (full recipe), and a `_latest.json` pointer.

## Choosing lags & time-feature encoding

`lag` is the lag count (`0..lag-1`); set an explicit `lags` list to use only
selected lags (overrides `lag` when non-empty — e.g. `[0,1,2,95,96,672]` for
recent + daily + weekly structure). `time_features.encoding` is `cyclical`
(compact sin/cos) or `onehot` (dummies). For the trade-offs — cyclical vs.
one-hot per model type, and pruning lags for fast, data-efficient training —
see **[docs/modeling-guide.md](docs/modeling-guide.md)**.

## Install

Editable (during development):

```toml
[tool.uv.sources]
kuafu-sysid = { path = "../kuafu-sysid", editable = true }
```

Git (once pushed to GitHub):

```toml
kuafu-sysid = { git = "https://github.com/mabrandes/kuafu-sysid.git" }
```

## Layout

- `features.py` — `FeatureSpec`, `build_features`, `feature_hash`.
- `models/` — adapters (`Linear`, `Knn`, `Xgb`, `Lgbm`, `Persistence`) + registry.
- `store.py` — `ModelStore` (hash-named artefacts + sidecar + `_latest.json`).
- `config.py` — `TrainConfig`, `SelectionConfig` YAML loaders.
- `train.py` / `evaluate.py` — `train(cfg)`, `evaluate(sel, role, df)`, `load_forecaster`.
- `metrics.py` / `plots.py` — per-horizon error + plotting.
