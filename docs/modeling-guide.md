# Modeling guide — encodings, lags & efficient training

Practical guidance for the config knobs that most affect accuracy, training
speed, and how much data you need.

## Features at a glance

| Feature | What it does | Where |
|---------|--------------|-------|
| **Time-feature toggles** | turn `time_of_day` / `day_of_week` / `day_of_year` / holidays on independently | `time_features:` in the train config · [§ encoding](#time-feature-encoding-cyclical-vs-one-hot) |
| **Encoding `cyclical` / `onehot`** | sin/cos pairs (compact) vs. category dummies (flexible) | `time_features.encoding` · [§ encoding](#time-feature-encoding-cyclical-vs-one-hot) |
| **Explicit `lags` list** | use a *selection* of lags (e.g. `[0,1,2,96,672]`) instead of dense `0..lag-1` → smaller, faster, data-efficient | `lags:` overrides `lag:` · [§ lags](#lags--efficient-training) |
| **Persistence from endog** | `prev_step`/`prev_day`/`prev_week` baselines read the target series directly, so they work at **any** `lag` (no need to inflate it) | automatic for `Persistence_*` models |
| **Validation + early stopping** | XGB/LGBM hold out a validation tail and stop at `best_iteration`; `n_estimators` is just an upper bound | `n_estimators`/`early_stopping_rounds`/`val_fraction` · [§ tree models](#tree-models-trees-learning-rate-validation--early-stopping) |
| **Learning curve** | `plot_learning_curve(model)` shows train vs. validation RMSE per round (over/underfitting) | [§ tree models](#tree-models-trees-learning-rate-validation--early-stopping) |
| **Feature importance** | `plot_feature_importance(model)` for XGB/LGBM (averaged across horizons for LGBM) | `model.feature_importances()` |
| **Saved diagnostics** | `train()` writes `<model>_{horizon,timeseries,importance,learning_curve}.png` next to each saved model | `save_plots=True` (default) |
| **Out-of-sample eval** | `evaluate(sel, role, df, start=train_end)` builds features on all of `df` but scores errors only on unseen rows | metrics: RMSE / MAE / **R²** / bias per horizon |
| **Eval plots** | `plot_horizon_metrics` (RMSE/MAE/R² panels) and `plot_timeseries(..., step=[1,4,24,96])` (forecast at several horizons vs measured) | — |
| **Progress prints** | `train()` logs data shape, feature count, and per-model RMSE / time / early-stop trees (`verbose=True`) | on by default |
| **Self-describing store** | each model saved by feature-`hash` + `_config.json` recipe + `_latest.json`, so config variants coexist and pin reproducibly | `store.root` + `sysid_select.yaml` |

## Time-feature encoding: cyclical vs. one-hot

`time_features.encoding` controls how calendar components (hour-of-day,
weekday, season) are represented.

**Cyclical (`sin`/`cos`)** — each component becomes **2 continuous columns**
(e.g. `hour_sin`, `hour_cos`). The pair lies on a circle, so the value *wraps*:
23:00 sits next to 00:00, Dec next to Jan. Compact (2 cols), smooth.

**One-hot (dummies)** — each category becomes its own **0/1 column**
(`tod_0..23`, `dow_0..6`, `month_1..12`). High-dimensional (24 + 7 + 12 = 43
columns vs. 6), no wrap, no smoothness — every hour/weekday/month is independent.

### Which to use for which model

| | Linear (Ridge) | Tree (XGB / LGBM) |
|---|---|---|
| **Cyclical** | Fits one smooth sinusoid per pair with 2 coefficients — great when the daily/seasonal shape is a smooth single-peak curve. Few parameters → needs little data, low overfit. Can't capture a sharp/multi-peak shape (only one harmonic). | Works fine; trees split the 2 continuous features. Compact, so fewer columns to search. Usually the better default for trees. |
| **One-hot** | Gives the model a free level per category → can fit an **arbitrary, non-smooth** profile (each hour its own coefficient). More flexible, but 43 extra parameters → needs more data and risks overfit; loses the 23↔0 adjacency. | Trees handle dummies natively (each split isolates a category), but high-cardinality one-hot fragments splits and adds many columns for little gain. Rarely beats cyclical here. |

**Rules of thumb**
- **Default to cyclical.** It's compact and works for both model families.
- Switch a Linear model to **one-hot** only when you specifically need a
  flexible, non-smooth per-hour/per-weekday profile and have enough data.
- For **trees**, prefer cyclical — one-hot mostly just inflates the feature
  matrix. (A plain ordinal hour would be even better for trees, but the library
  offers cyclical or one-hot only.)
- Encoding is part of the `feature_hash`, so cyclical and one-hot versions of
  the same target coexist as separate saved models — easy to A/B.

## Lags & efficient training

Every lag adds a column **per lagged series** — `lag: 96` creates 96
`{endog}_lag_*` columns, plus 96 for each entry in `exog_with_lag`. With a
multi-step horizon (e.g. 96) the design matrix gets very wide, which means:

- **more data needed** — a Linear model wants roughly *#rows ≫ #features*
  (a common rule is ≥ ~10 rows per feature) to estimate coefficients reliably;
- **slower training** and **higher overfit risk** (especially for Linear/KNN);
- diminishing returns — most lags carry little extra signal.

### Use a *selection* of lags, not all of them

The informative lags are usually:
- the **most recent** steps (`0, 1, 2, 3`) — current level + short trend,
- the **daily period** and its neighbours (`steps_per_day = 96` for 15-min:
  `95, 96, 97`),
- the **weekly period** and neighbours (`672, 673`) when weekly structure matters.

A list like `[0, 1, 2, 3, 95, 96, 97, 672]` captures recent + daily + weekly
structure with **~8 columns instead of 672** — far faster, needs far less data,
and often *more* accurate than the dense `0..671` because there's less noise to
overfit.

### The `lags` config field

```yaml
lag: 96         # the count: used only when `lags` is empty -> expands to 0..95
lags: []        # explicit list of lag steps; NON-EMPTY overrides `lag`
# e.g. recent + daily + weekly structure:
# lags: [0, 1, 2, 3, 95, 96, 97, 672]
```

- `lags` non-empty → exactly those lags are used (deduped, sorted).
- `lags` empty/absent → fall back to `lag` (the dense `0..lag-1`).

Persistence baselines (`prev_step/day/week`) read the endog series directly, so
**pruning lags never breaks them** — pick the smallest lag set that performs.

### Getting to "minimum data for best performance"

1. **Prune lags** with the `lags` list (biggest lever — cuts feature count ~100×).
2. **Trim exog** to the few genuinely predictive drivers; drop weak entries from
   `exog_with_lag` first (each one is multiplied by every lag).
3. **Prefer cyclical** time features (6 cols) over one-hot (43 cols).
4. **Shorten the train window** once the feature count is small — fewer features
   means a few weeks of data can suffice; watch the per-horizon RMSE vs. the
   persistence baselines and stop adding data when it stops improving.
5. Trees (XGB/LGBM) tolerate wide matrices and NaN better than Linear/KNN, so if
   you must keep many features, lean on them; if you want the smallest, fastest
   model, prune hard and use Linear.

Because every choice above feeds the `feature_hash`, you can train several
variants, compare their per-horizon metrics, and pin the best one in
`sysid_select.yaml` — without any of them overwriting each other.

## Tree models: trees, learning rate, validation & early stopping

`XGB` and `LGBM` are **gradient-boosted trees**: they build an ensemble of small
decision trees *one at a time*, where each new tree is fit to the errors
(residuals) the ensemble still makes. Two parameters govern this:

- **`n_estimators`** — the number of trees / boosting rounds. Each added tree
  reduces training error a bit more. Too few → *underfit* (not enough capacity);
  too many → *overfit* (the model starts memorising noise and generalises worse).
- **`learning_rate`** (shrinkage) — how much each tree contributes. Smaller steps
  generalise better but need *more* trees to reach the same fit. Typical pairing:
  a small learning rate (`0.05`) with a large `n_estimators` upper bound, then let
  early stopping decide how many trees are actually used.

### Early stopping (and why `n_estimators` is just an upper bound)

Rather than guess the right number of trees, we **hold out a validation set** and
watch its error after each new tree:

- after every boosting round, error is measured on the validation set;
- if it hasn't improved for **`early_stopping_rounds`** consecutive rounds, training
  stops;
- the model remembers **`best_iteration`** — the round with the lowest validation
  error — and predicts using only those trees.

So `n_estimators` is an *upper bound*: training usually stops well before it. This
**prevents overfitting** (you don't keep adding trees once they stop helping unseen
data) and **saves compute**. The progress log shows it, e.g.
`XGB: mean RMSE=2.41 (8.3s, early-stopped @ 137 trees)`.

### Where the validation set comes from

The validation set is the **last `val_fraction` of the training data** (a
time-ordered tail — the most recent rows). Crucially it is carved out of the
*training* split, **not** the test split, so the held-out test set stays clean and
the reported per-horizon metrics are honest (no early-stopping leakage).

```
|<--------------- training window --------------->|<---- test ---->|
|<------ fit trees ------>|<-- validation (val_fraction) -->|        (early stopping watches this)
```

### Defaults & knobs

| Param | Default | Meaning |
|-------|---------|---------|
| `n_estimators` | 1000 | max trees (upper bound; early stopping usually stops sooner) |
| `learning_rate` | 0.05 | contribution per tree |
| `early_stopping_rounds` | 50 | stop after this many rounds with no validation improvement |
| `val_fraction` | 0.2 | fraction of the train window held out to monitor |
| `max_depth` (XGB) / `num_leaves` (LGBM) | 6 / 31 | tree size (capacity per tree) |
| `subsample`, `colsample_bytree` (XGB) | 0.8 | row / column sampling per tree (regularisation) |

`LGBM` applies the same idea **per horizon step** (one model per step), so each
step early-stops independently. `Linear`/`KNN` have no boosting rounds, so early
stopping does not apply to them.

**Watch it live:** `train(cfg, tree_eval_log=N)` streams the train- and
validation-RMSE every `N` boosting rounds (like XGBoost's `verbose=True`):

```
[0]   validation_0-rmse:2.805   validation_1-rmse:3.003     # validation_0 = train, _1 = validation
[20]  validation_0-rmse:1.192   validation_1-rmse:1.580
[40]  validation_0-rmse:0.587   validation_1-rmse:1.346     # val bottoms out…
[sysid] XGB: mean RMSE=1.46  (…, early-stopped @ 44 trees)  # …then early stopping fires
```

`tree_eval_log=0` (default) is silent. LGBM prints per horizon, so it's chattier.

### Seeing the learning curve

A freshly-trained `Xgb` keeps the train-vs-validation error per round in
`evals_result_`; plot it to see under/overfitting and where early stopping fired:

```python
from kuafu_sysid import get_model, plot_learning_curve
from kuafu_sysid.features import build_features
X, Y = build_features(df, spec, lag, horizon, dt_min, time_features)
m = get_model("XGB", horizon=horizon, endog="pv").fit(X.iloc[:n_tr], Y.iloc[:n_tr])
plot_learning_curve(m)   # train vs validation RMSE, with the best-iteration marked
```

Reading it: the **train** curve keeps falling; the **validation** curve falls then
flattens (or rises) — the dashed line marks `best_iteration`, where validation
stopped improving. A big gap between the two curves means overfitting (reduce
`max_depth`/`num_leaves`, lower `learning_rate`, or add data).
