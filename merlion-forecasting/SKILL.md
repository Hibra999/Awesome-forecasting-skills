---
name: merlion-forecasting
description: Use Salesforce Merlion for forecasting workflows after forecasting-data-prep, including TimeSeries/UnivariateTimeSeries data, DefaultForecaster, Arima, Sarima, ETS, Prophet, MSES, VectorAR, tree forecasters, DeepAR/Autoformer/ETSformer/Informer/Transformer, AutoETS/AutoProphet/AutoSarima, ForecasterEnsemble, exogenous regressors, uncertainty/error bars, ForecastEvaluator live-deployment validation, forecast metrics, plotting, and anti-leakage safeguards.
---

# Merlion Forecasting

Use this skill after `forecasting-data-prep`. Merlion is useful when you need a unified forecasting interface, default forecasters, ensembles/model selection, AutoML wrappers, exogenous regressors, live-deployment-style evaluation, or the same framework for forecasting and adjacent anomaly detection.

The official GitHub repo was archived on March 11, 2026 and the latest release shown in the repo is v2.0.2 from February 15, 2023. Treat Merlion as legacy/stable code and verify Python/dependency compatibility before promising a runnable workflow.

## Minimum Install

```bash
pip install salesforce-merlion
pip install "salesforce-merlion[all]"
```

Optional extras documented by the README include `dashboard`, `spark`, and `deep-learning`. For source installs, clone the repo and use `pip install -e .`. Some forecasting models depend on OpenMP/LightGBM; on conda, install `lightgbm` from `conda-forge` first when needed.

## Data Contract

- Start from a `forecasting-data-prep` contract: sorted timestamps, target column(s), frequency, temporal cutoffs, horizon, known-future regressors, and leakage notes.
- Convert pandas time-indexed data to `merlion.utils.TimeSeries` with `TimeSeries.from_pd(...)`; convert back with `.to_pd()`.
- Use `UnivariateTimeSeries` or one-column `TimeSeries` for univariate models. Use multicolumn `TimeSeries` for multivariate models and for selecting a target via `target_seq_index`.
- Merlion does not document a generic panel ID container for many independent series. Loop over IDs, use `ts_datasets` benchmark loaders, or build a validated wrapper externally.
- For models requiring even sampling, use a train-only `TemporalResample` transform with explicit granularity.
- Pass exogenous variables as a separate `TimeSeries` through `exog_data`; they must be known for all training/inference timestamps used by the model.

Read `references/merlion-data-validation.md` before using multivariate targets, exogenous variables, transforms, evaluators, uncertainty bands, or panel-like workflows.

## Model Selection

Use documented model/config class names exactly.

- Starter/default: `DefaultForecaster`.
- Univariate: `Arima`, `Sarima`, `ETS`, `Prophet`, `MSES`.
- Multivariate: `VectorAR`, `RandomForestForecaster`, `ExtraTreesForecaster`, `LGBMForecaster`.
- Deep learning: `DeepARForecaster`, `AutoformerForecaster`, `ETSformerForecaster`, `InformerForecaster`, `TransformerForecaster`.
- AutoML: `AutoETS`, `AutoProphet`, `AutoSarima`, plus `SeasonalityLayer` utilities.
- Ensembles: `ForecasterEnsemble` with combiners such as `Mean`, `Median`, `Max`, `ModelSelector`, and `MetricWeightedMean`.
- Factory aliases: `ModelFactory` supports documented/import-alias names such as `DefaultForecaster`, `Arima`, `ETS`, `MSES`, `Prophet`, `Sarima`, `VectorAR`, tree forecasters, deep forecasters, AutoML wrappers, and `ForecasterEnsemble`.

Read `references/merlion-model-map.md` before claiming exogenous, multivariate-output, deep-learning, AutoML, or uncertainty support.

## Forecasting Pattern

```python
from merlion.utils import TimeSeries
from merlion.models.defaults import DefaultForecaster, DefaultForecasterConfig
from merlion.evaluate.forecast import ForecastMetric

train_data = TimeSeries.from_pd(train_df.set_index(time_col)[[target_col]])
test_data = TimeSeries.from_pd(test_df.set_index(time_col)[[target_col]])

model = DefaultForecaster(DefaultForecasterConfig())
train_pred, train_err = model.train(train_data=train_data)

test_pred, test_err = model.forecast(time_stamps=test_data.time_stamps)
smape = ForecastMetric.sMAPE.value(ground_truth=test_data, predict=test_pred)
```

For explicit models, initialize `ModelClass(ConfigClass(...))`. For example, models that assume regular sampling often need `TemporalResample(granularity="1h")` and some need `max_forecast_steps`.

For exogenous regressors:

```python
model.train(train_data=train_y, exog_data=train_exog)
forecast, stderr = model.forecast(time_stamps=future_timestamps, exog_data=future_exog)
```

`future_exog` must include all required future timestamps and only variables known at prediction time.

## Validation, Metrics, and Plotting

- Use temporal holdouts or `ForecastEvaluator` to simulate deployment with `cadence`, `horizon`, `retrain_freq`, and optional `train_window`.
- Do not random split rows. Split by timestamp before fitting Merlion transforms, AutoML, ensembles, or exogenous transforms.
- Built-in forecast metrics include `MAE`, `MARRE`, `RMSE`, `sMAPE`, `RMSPE`, `MASE`, and `MSIS`.
- Add WAPE, bias, coverage, and business metrics externally when needed. Avoid percent metrics when actuals can be zero or near zero.
- Plot with `model.plot_forecast(...)`, `model.plot_forecast_plotly(...)`, or `get_figure(...)`. `plot_forecast_uncertainty=True` works only when the model returns uncertainty.
- Merlion does not document a universal residual diagnostics API. Compute validation/test residuals from `TimeSeries.to_pd()` outputs and inspect by horizon, time, and target variable.

## Probabilistic Forecasting

`train()` and `forecast()` return `(forecast, stderr)`; `stderr` may be `None` for models without uncertainty. `forecast(..., return_iqr=True)` returns `(forecast, lb, ub)` where supported. Use `ForecastMetric.MSIS` for interval quality when you have 95% lower/upper bounds.

Do not promise calibrated intervals for every model. Validate empirical coverage on temporal validation windows.

## Anti-Leakage Rules

- Never random split time-series observations.
- Fit `TemporalResample`, normalization, missing-value policies, AutoML model selection, ensemble combiners, and any custom feature generation on train only or inside each evaluator/backtest fold.
- Use lags/rolling features only from past values; apply group-wise `shift` before rolling when looping over IDs.
- Use exogenous variables only when known for all requested forecast timestamps and independent of the target being forecast.
- Keep `max_forecast_steps`, `time_stamps`, `horizon`, `cadence`, `retrain_freq`, `train_window`, frequency, and split cutoffs aligned.
- For live-deployment simulation, reinitialize/reset models per evaluator run and avoid tuning on final test residuals.

## Common Errors

- Passing raw pandas directly to models instead of `TimeSeries`.
- Ignoring the archived/legacy status and dependency constraints.
- Treating multivariate `TimeSeries` as panel data over independent IDs.
- Using univariate models without setting `target_seq_index` when forecasting one dimension of multivariate data.
- Forgetting to provide future `exog_data` at forecast time.
- Fitting transforms, AutoML, or ensemble selectors on validation/test data.
- Plotting uncertainty for models whose `stderr` is `None`.

## References

- Read `references/merlion-model-map.md` for supported forecasting models and capability caveats.
- Read `references/merlion-data-validation.md` for data formats, exogenous variables, evaluation, plotting, intervals, and diagnostics.
- Read `references/official-sources.md` for official sources consulted.

## Ready Checklist

- `forecasting-data-prep` contract is complete and anti-leakage risks are documented.
- pandas data is time-indexed, sorted, split by time, and converted to `TimeSeries`.
- Frequency, horizon, `max_forecast_steps`, transforms, and target variable are explicit.
- Chosen model supports the required univariate/multivariate/exogenous/uncertainty behavior.
- Validation uses temporal holdout or `ForecastEvaluator`, never random split.
- Metrics, plots, residuals, and interval coverage are computed on held-out periods only.
