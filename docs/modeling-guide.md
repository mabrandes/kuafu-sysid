# Modeling guide — encodings, lags & efficient training

Practical guidance for the config knobs that most affect accuracy, training
speed, and how much data you need.

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
