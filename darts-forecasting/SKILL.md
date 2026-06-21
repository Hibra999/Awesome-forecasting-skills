---
name: darts-forecasting
description: Use Unit8 Darts for forecasting with TimeSeries, local/statistical models, global regression and deep learning models, foundation models, covariates, static covariates, probabilistic forecasts, conformal prediction, multiple-series training, backtesting, residual diagnostics, plotting, and leakage-safe temporal validation. Trigger when an agent needs to model prepared time-series data with Darts after applying forecasting-data-prep for frequency, horizon, covariate availability, splits, and anti-leakage checks.
---

# Darts Forecasting

Use this skill after `forecasting-data-prep`. Darts is best when you need a unified `fit()`/`predict()` API over classical, regression, deep learning, foundation, ensemble, and conformal forecasting models using Darts `TimeSeries` objects.

## Minimum Install

```bash
python -m pip install darts
python -m pip install "darts[torch]"
python -m pip install "darts[notorch]"
python -m pip install "darts[all]"
conda install -c conda-forge u8darts
conda install -c conda-forge -c pytorch u8darts-all
```

Use the smallest install that contains the needed model family. Some models require extra packages not included in `all`, such as `neuralforecast` for `NeuralForecastModel` and `tirex-ts` for `TiRexModel`.

## Data Contract

- Start from a `forecasting-data-prep` contract: sorted timestamps, no duplicate keys, fixed or documented frequency, temporal splits, horizon, known-future covariates, and leakage notes.
- Convert target data to `darts.TimeSeries`; every Darts forecasting model consumes and returns `TimeSeries`.
- Use `TimeSeries.from_dataframe`, `from_times_and_values`, `from_values`, `from_series`, `from_group_dataframe`, `from_xarray`, `from_csv`, or `from_json` as appropriate.
- Represent one entity with multiple jointly modeled columns as one multivariate `TimeSeries`; represent multiple entities/panel data as a `Sequence[TimeSeries]`.
- Use `past_covariates` only for variables observed up to prediction time; use `future_covariates` only for variables known or validly forecasted through the horizon.
- Keep static covariates on the `TimeSeries` only when the chosen model documents support for them.
- Keep hierarchical metadata on a multivariate `TimeSeries` only for documented posthoc reconciliation workflows.

## Model Selection

- Always benchmark `NaiveSeasonal`, `NaiveDrift`, `NaiveMean`, or another simple baseline before complex models.
- Use local/statistical models for one series at a time: ARIMA-like, smoothing, Theta, Prophet, FFT, Kalman, TBATS, Croston, and StatsForecast wrappers.
- Use regression/global models when lagged target/covariate features, multiple series training, static covariates, or tabular ML backends are useful.
- Use Torch models for larger datasets, global learning, probabilistic likelihoods, GPUs, validation series, and sequence models.
- Use foundation models only when their documented dependencies, no-training/fine-tuning behavior, and input requirements fit the task.
- Use conformal models to calibrate intervals for pre-trained global models.

Read `references/darts-model-map.md` before choosing among model families or claiming model support.

## Workflow

1. Prepare data with `forecasting-data-prep`; preserve time column, target columns, series IDs, frequency, horizon, gap, cutoffs, and covariate roles.
2. Build target `TimeSeries` objects and split by time with slicing or `split_before`/`split_after`; keep a final test segment untouched.
3. Fit fittable transformers such as `Scaler`, `BoxCox`, encoders, imputers, or static-covariate transformers on train only, then transform validation/test/future data.
4. Build covariate `TimeSeries` with enough span for the selected model, horizon, and lag/window settings.
5. Select a model whose documented capabilities match univariate/multivariate target, multiple series, covariates, static covariates, probabilistic output, and sample weights.
6. Fit with `model.fit(train, past_covariates=..., future_covariates=...)`; for global models, pass a list of target series and matching covariate lists.
7. Predict with `model.predict(n=horizon, series=..., past_covariates=..., future_covariates=..., num_samples=...)` as required by model type.
8. Validate with `historical_forecasts()` or `backtest()` over temporal cutoffs; use test once for final reporting.

## Python Pattern

```python
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, rmse, mase
from darts.models import NaiveSeasonal

series = TimeSeries.from_dataframe(df, time_col=time_col, value_cols=target_cols)
train, val = series.split_before(cutoff)

scaler = Scaler()
train_t = scaler.fit_transform(train)
val_t = scaler.transform(val)

model = NaiveSeasonal(K=seasonal_period)
model.fit(train_t)
forecast_t = model.predict(n=len(val_t))
forecast = scaler.inverse_transform(forecast_t)

mae_score = mae(val, forecast)
rmse_score = rmse(val, forecast)
mase_score = mase(val, forecast, insample=train)

series.plot(label="actual")
forecast.plot(label="forecast")
```

For global models, pass `Sequence[TimeSeries]` to `fit()` and specify `series=` at `predict()` when forecasting one or more target series.

## Validation, Metrics, and Diagnostics

- Use `historical_forecasts()` to simulate historical forecasts and `backtest()` to compute errors over those forecasts.
- Set `forecast_horizon`, `start`, `stride`, `last_points_only`, and retraining behavior to match the real forecast process.
- Recommended point metrics: MAE, RMSE, MASE/RMSSE, wMAPE, bias/OPE where relevant; avoid MAPE/sMAPE when actuals can be zero or near zero.
- Recommended probabilistic metrics: MCRPS/CRPS, MQL/QL, QR, interval coverage, interval width, and Winkler score.
- Plot with `TimeSeries.plot()`. For probabilistic forecasts, use quantile plotting via `low_quantile`/`high_quantile`.
- Use `model.residuals(...)` and `darts.utils.statistics.plot_residuals_analysis()` where applicable. Prefer out-of-sample residuals from historical forecasts for diagnostics.

Read `references/darts-data-validation.md` before building covariates, transformers, historical forecasts, probabilistic forecasts, or residual diagnostics.

## Anti-Leakage Rules

- Never random split forecasting rows or forecast cutoffs.
- Fit all transformers, scalers, imputers, encoders, static-covariate transformers, model selection, and anomaly thresholds on train only or inside each backtest fold.
- Create lag/window features only through past data available at each cutoff; regression model `lags` and `lags_past_covariates` must be negative/past-facing.
- Use `future_covariates` only when values are known at prediction time for every required future timestamp, or when they come from a separately validated forecast.
- Respect horizon `n`, `forecast_horizon`, `output_chunk_length`, `output_chunk_shift`, frequency, gaps, covariate spans, and series boundaries.
- Rebuild covariates, encoders, transformations, and historical forecasts separately per cutoff during validation.

## Common Errors

- Passing pandas/NumPy directly to models instead of converting to `TimeSeries`.
- Confusing multivariate components with multiple series/panel data.
- Fitting `Scaler` or missing-value filling on the full series before splitting.
- Passing `future_covariates` that are not actually known through the prediction horizon.
- Using a local model when multiple-series/global training is required, or omitting `series=` at `predict()` for global models trained on multiple series.
- Installing core `darts` and then using unavailable Torch, Prophet, LightGBM, CatBoost, XGBoost, StatsForecast, NeuralForecast, or TiRex models.
- Treating all probabilistic forecasts as calibrated intervals; use conformal models or validation coverage checks when calibration matters.

## References

- Read `references/darts-model-map.md` for documented model families, classes, dependencies, and capability checks.
- Read `references/darts-data-validation.md` for `TimeSeries` formats, covariates, horizons, transformations, backtesting, metrics, plotting, and diagnostics.
- Read `references/official-sources.md` for official sources consulted.

## Ready Checklist

- `forecasting-data-prep` contract is complete and leakage risks are resolved or documented.
- Target, covariates, static covariates, series IDs, frequency, and horizon are converted to valid `TimeSeries` objects.
- The chosen model supports the required univariate/multivariate target, multiple series, covariates, static covariates, probabilistic output, and sample weights.
- Future covariates cover the full horizon and are genuinely available at prediction time.
- Transformations and model selection are fit inside temporal train folds only.
- Validation uses temporal holdout or Darts historical backtesting and reports baseline comparison, horizon-aware metrics, plots, and residual checks.
