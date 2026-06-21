---
name: kats-forecasting
description: Use Facebook/Meta Kats for forecasting with TimeSeriesData, classical models, Prophet, LSTM, LightGBM MLAR, global models, ensembles, temporal hierarchical reconciliation, hyperparameter/meta-learning helpers, prediction intervals where documented, plotting, backtesting, and leakage-safe temporal validation. Trigger when an agent needs to model prepared time-series data with Kats after applying forecasting-data-prep for frequency, horizon, splits, covariates, and anti-leakage checks, especially when working with legacy Kats 0.2.0 projects.
---

# Kats Forecasting

Use this skill after `forecasting-data-prep`. Kats is best for legacy or existing Kats projects that need `TimeSeriesData`, classical forecasting models, Prophet wrappers, ensembles, meta-learning utilities, global LightGBM-style forecasting, anomaly/change detection adjacency, or temporal hierarchical reconciliation.

Kats' public docs are old and incomplete relative to the GitHub `main` tree. Do not invent APIs from modern forecasting libraries; verify class names and parameters in official API docs or source before use.

## Minimum Install

```bash
python -m pip install --upgrade pip
python -m pip install kats
MINIMAL_KATS=1 python -m pip install kats
```

`MINIMAL_KATS=1` omits many dependencies and disables functionality. Kats 0.2.0 on PyPI is alpha-era software released in 2022 and declares Python 3.7/3.8 classifiers, so check environment compatibility before promising a runnable workflow.

## Data Contract

- Start from a `forecasting-data-prep` contract: sorted timestamps, no duplicate keys, documented frequency, temporal split cutoffs, horizon, known-future covariates, and leakage notes.
- Convert data to `kats.consts.TimeSeriesData` before modeling. Initialize from a pandas `DataFrame`, `Series`, or `DatetimeIndex`/value pair.
- Use the default `time` column or pass `time_col_name`; value columns can be a pandas `Series` for univariate data or `DataFrame` for multivariate data.
- Use `TimeSeriesData.validate_data(validate_frequency=True, validate_dimension=True)` when frequency and shape must be strict.
- Use `TimeSeriesData.interpolate()` only inside the training window or inside each backtest fold. Interpolation over validation/test target values leaks.
- Kats does not document a generic panel data container. For independent multiple series, loop over IDs or use documented global-model APIs that accept lists/dicts of `TimeSeriesData`.

## Model Selection

- Use simple baselines outside Kats or Kats lightweight trend/seasonal models before complex models.
- Use univariate local models: `ProphetModel`, `ARIMAModel`, `SARIMAModel`, `HoltWintersModel`, `ThetaModel`, `STLFModel`, `LinearModel`, `QuadraticModel`, `HarmonicRegressionModel`, `LSTMModel`, `SimpleHeuristicModel`, and `NeuralProphetModel` when dependencies and source support it.
- Use multivariate local models only when the target is jointly multivariate and the model documents it: `VARModel`, `BayesianVAR`.
- Use `MLARModel` or the `globalmodel` package only for documented multiple-series/global workflows.
- Use ensemble classes only after defining leakage-safe validation for weights: `MedianEnsembleModel`, `WeightedAvgEnsemble`, and Kats ensemble helpers.
- Use `TemporalHierarchicalModel` only for temporal aggregation reconciliation, not generic cross-sectional hierarchy.

Read `references/kats-model-map.md` before choosing among model families or claiming support.

## Workflow

1. Prepare data with `forecasting-data-prep`; preserve `freq`, horizon, gap, cutoff timestamps, series IDs, and covariate roles.
2. Split by time before any Kats interpolation, scaling, encoding, model selection, or hyperparameter tuning.
3. Build `TimeSeriesData(train_df, time_col_name=...)`; keep validation/test/future as separate dataframes or `TimeSeriesData`.
4. Choose a model and params class from official docs/source; construct as `ModelClass(train_ts, ParamsClass(...))`.
5. Fit with `model.fit(...)`.
6. Predict with documented forecast horizon arguments, usually `model.predict(steps=horizon, include_history=False, freq=freq)`.
7. Evaluate with a temporal holdout or rolling-origin backtest. Keep final test data untouched until final reporting.
8. Plot with `model.plot()` or `Model.plot(train_ts, forecast_df, include_history=True)` when supported.

## Python Pattern

```python
import pandas as pd
from kats.consts import TimeSeriesData
from kats.models.prophet import ProphetModel, ProphetParams

train_df = train_df.rename(columns={time_col: "time", target_col: "y"})
train_ts = TimeSeriesData(train_df[["time", "y"]])
train_ts.validate_data(validate_frequency=True, validate_dimension=True)

params = ProphetParams(seasonality_mode="additive")
model = ProphetModel(train_ts, params)
model.fit()

forecast = model.predict(steps=horizon, freq=freq, include_history=False)
# expected documented columns for many models: time, fcst, fcst_lower, fcst_upper
```

Use per-ID loops for independent panels unless using `MLARModel`/global-model APIs documented for multiple `TimeSeriesData` inputs.

## Validation, Metrics, and Diagnostics

- Kats docs mention backtesting and consolidated backtesting APIs but do not provide complete public Sphinx documentation for a universal backtesting interface. Prefer explicit temporal cutoffs when the API is unclear.
- Recommended metrics: MAE, RMSE, MASE/RMSSE, WAPE, sMAPE only when zeros are not an issue, bias/OPE, and interval coverage/width when intervals are emitted.
- Kats source uses metrics such as `smape`, `sbias`, and exceedance-style metrics in some global workflows; otherwise compute metrics explicitly from forecast `fcst` against held-out actuals.
- Probabilistic support is model-specific. Many local models return `fcst_lower`/`fcst_upper`; Bayesian VAR docs explicitly say confidence intervals are not yet implemented.
- Use `TimeSeriesData.plot(cols=[...])`, `model.plot()`, or the static model `plot()` helper where documented.
- For residual diagnostics, compute out-of-sample residuals from validation/backtests. Kats does not document one universal residual diagnostics API across forecasters.

Read `references/kats-data-validation.md` before using covariates, global models, backtesting, intervals, plotting, or diagnostics.

## Anti-Leakage Rules

- Never random split forecasting rows.
- Fit interpolation, missing-value handling, scalers, encoders, feature selection, hyperparameter tuning, and ensemble weights on train only or inside each temporal fold.
- Build lag/window inputs for `MLARModel`, LSTM, and global models using only history available before each cutoff.
- Use future covariates only when values are known at prediction time for every horizon step, or when they come from a separately validated forecast.
- Respect `steps`, `freq`, horizon, gap, timestamp cutoff, seasonal periods, lag windows, and temporal aggregation levels.
- Rebuild features, covariates, transformations, and ensemble weights separately per cutoff during backtesting.

## Common Errors

- Treating Kats as an actively modern library without checking the old 0.2.0/Python compatibility constraints.
- Passing raw pandas directly to models instead of `TimeSeriesData`.
- Assuming all models support multivariate targets, panel data, exogenous regressors, or intervals.
- Using `TimeSeriesData.interpolate()` before splitting.
- Training ensemble weights or meta-learning selection on validation/test periods.
- Confusing `VARModel`/`BayesianVAR` multivariate endogenous modeling with independent panel forecasting.
- Expecting documented covariate support for every model. Most local Kats models do not expose a consistent exogenous-regressor interface.

## References

- Read `references/kats-model-map.md` for documented models, source-only models, ensembling, global models, and limitations.
- Read `references/kats-data-validation.md` for `TimeSeriesData`, covariates, horizons, validation, metrics, plotting, and diagnostics.
- Read `references/official-sources.md` for official sources consulted.

## Ready Checklist

- `forecasting-data-prep` contract is complete and leakage risks are resolved or documented.
- Kats/Python/dependency compatibility is checked for the target environment.
- Data is converted to `TimeSeriesData` with explicit time/value columns and validated frequency/dimensions where required.
- Chosen model's official docs/source support the required univariate/multivariate/global/covariate/interval behavior.
- Validation uses temporal cutoffs/backtesting, not random splits.
- Metrics, plots, forecast intervals, and residual checks are computed on held-out periods only.
