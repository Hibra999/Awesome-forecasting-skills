# Merlion Data, Validation, and Evaluation

Use this reference when converting prepared data to Merlion and designing validation.

## Data Formats

Merlion models use:

- `UnivariateTimeSeries`: one named sequence.
- `TimeSeries`: one or more aligned univariates.

Typical conversion:

```python
from merlion.utils import TimeSeries

ts = TimeSeries.from_pd(df.set_index(time_col)[value_cols])
df_back = ts.to_pd()
```

Official `ts_datasets` loaders return pandas DataFrames plus metadata. Custom datasets should already be cleaned by `forecasting-data-prep`.

## Univariate, Multivariate, and Panels

- Univariate: pass a one-column `TimeSeries`.
- Multivariate target/context: pass a multicolumn `TimeSeries` and set `target_seq_index` when forecasting one variable.
- True multivariate output: use models whose `support_multivariate_output` is true.
- Multiple independent series/panel: Merlion forecasting docs do not document a generic panel ID container. Loop over IDs, use benchmark loaders, or write an external wrapper that preserves temporal splits per series.

## Frequency, Resampling, and Missing Values

Some models require evenly sampled data. Use `TemporalResample(granularity=...)` explicitly when needed, and fit that transform only on training data.

Merlion transforms can aggregate and impute values. Treat aggregation, interpolation, missing-value policies, scaling, and normalization as train-only preprocessing. If a transform learns parameters, do not fit it on validation/test periods.

## Forecast Horizon

Merlion forecasts by timestamp list or integer steps:

- `forecast(time_stamps=[...])`
- `forecast(time_stamps=n_steps)`

Some models require `max_forecast_steps`. Keep it at least as large as the requested horizon and aligned with validation/evaluator horizons.

`time_series_prev` can provide recent context when forecasting beyond the original training data. It must contain only observations available before the forecast creation time.

## Exogenous Regressors

Merlion's forecaster API accepts `exog_data` in `train()` and `forecast()`, but only some models actually use it.

Rules:

- Exogenous data is a separate `TimeSeries`.
- It must be sampled or resampled to align with endogenous timestamps.
- At inference, `exog_data` must include every requested forecast timestamp.
- If `time_series_prev` is supplied, exogenous data must cover those previous timestamps too.
- Exogenous variables must be known a priori and independent of the variable being forecasted.

Default exogenous handling in docs includes `MeanVarNormalize`, missing value policy `ZFill`, and aggregation policy `Mean`. Override deliberately and fit only on training data/folds.

## Training and Prediction

All forecasting models share:

```python
forecast_train, stderr_train = model.train(train_data=train_data)
forecast, stderr = model.forecast(time_stamps=test_data.time_stamps)
```

The training return is a forecast over training timestamps plus standard error where available. Use it for diagnostics carefully; model selection should rely on validation/evaluator output.

Save/load:

- `model.save(path)`
- `ModelClass.load(dirname=path)`
- `ModelFactory.load(name="ModelName", model_path=path)`

## Validation and Backtesting

Use temporal holdouts or `ForecastEvaluator`.

`ForecastEvaluatorConfig` controls:

- `horizon`: how far ahead to predict.
- `cadence`: how often to obtain predictions.
- `retrain_freq`: how often to retrain.
- `train_window`: maximum amount of recent history for retraining.

The evaluator simulates live deployment: train initially, forecast future windows, retrain at specified cadence, and evaluate against test values.

Do not random split. For model selection, reset/reinitialize models and rerun the evaluator or holdout workflow per candidate. Use final test only once.

## Metrics

Official `ForecastMetric` options:

- `MAE`
- `MARRE`
- `RMSE`
- `sMAPE`
- `RMSPE`
- `MASE`
- `MSIS`

Recommended additions outside Merlion:

- WAPE for business reporting.
- Mean error/bias for systematic over- or under-forecasting.
- Interval coverage and width when using stderr/IQR-based bounds.
- Horizon-wise metrics for multi-step forecasts.

Avoid percent metrics when actual values can be zero or near zero.

## Plotting and Diagnostics

Use:

- `model.get_figure(...)`
- `model.plot_forecast(...)`
- `model.plot_forecast_plotly(...)`

`plot_forecast_uncertainty=True` only works when the selected model returns uncertainty. Residual diagnostics are manual: convert held-out `TimeSeries` forecasts and actuals to pandas, compute residuals by horizon/time/target, and inspect bias, autocorrelation, outliers, and interval misses.

## Limitations to Surface

- The GitHub repository is archived and read-only as of March 11, 2026.
- Latest release shown by GitHub is v2.0.2 from February 15, 2023.
- Python/package compatibility may require pinning older dependencies.
- No generic panel ID container is documented for forecasting.
- Exogenous support is model-specific despite the common API accepting `exog_data`.
- Not every model returns uncertainty or supports `return_iqr`.
- Deep-learning models require optional dependencies and enough data.
- Some models require regular sampling and explicit resampling transforms.
