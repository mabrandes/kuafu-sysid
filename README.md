# kuafu-sysid

Generic, config-driven forecasting framework (direct multi-step regression)
shared across `kuafu_*` projects.

## Example

A runnable end-to-end demo lives in [`examples/`](examples/): `example.ipynb`
generates a small synthetic dataset, then trains/evaluates from
[`examples/sysid_train.yaml`](examples/sysid_train.yaml) and
[`examples/sysid_select.yaml`](examples/sysid_select.yaml). Run the notebook from
the `examples/` directory.

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

`XGB`/`LGBM` train with a held-out **validation tail + early stopping**
(`n_estimators` is just an upper bound — training stops at `best_iteration`); the
`train()` log reports it and `plot_learning_curve()` shows the train-vs-validation
curve.

## Choosing lags, encoding & tree settings

`lag` is the lag count (`0..lag-1`); set an explicit `lags` list to use only
selected lags (overrides `lag` when non-empty — e.g. `[0,1,2,95,96,672]` for
recent + daily + weekly structure). `time_features.encoding` is `cyclical`
(compact sin/cos) or `onehot` (dummies). For the trade-offs — cyclical vs.
one-hot per model type, pruning lags for fast/data-efficient training, and what
`n_estimators` / `learning_rate` / **early stopping** / validation mean for the
tree models — see **[docs/modeling-guide.md](docs/modeling-guide.md)**.

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
