# Darts Data, Validation, and Diagnostics

Use this reference before implementing data conversion, covariates, transformations, historical forecasts, probabilistic forecasts, plotting, or diagnostics in Darts.

## TimeSeries Formats

- `TimeSeries` is Darts' main data class. All Darts forecasting models consume and produce `TimeSeries`.
- A `TimeSeries` has a complete, time-sorted index: `pandas.DatetimeIndex` or `pandas.RangeIndex`.
- Components are columns/dimensions. One component is univariate; multiple components in one `TimeSeries` are multivariate and share the same time axis.
- Multiple series/panel data is a Python sequence such as `list[TimeSeries]`; different series do not need to share the same time index or frequency.
- Probabilistic forecasts are `TimeSeries` with multiple samples.
- Factory methods include `TimeSeries.from_dataframe`, `from_times_and_values`, `from_values`, `from_series`, `from_group_dataframe`, `from_xarray`, `from_csv`, and `from_json`.
- Export methods include `to_dataframe`, `to_series`, `values`, `all_values`, `data_array`, `to_csv`, and `to_json`.
- `TimeSeries` can also carry static covariates and hierarchy metadata for documented model/reconciliation workflows.

## Covariates

- Target series: the series being forecasted.
- Past covariates: values observed only up to prediction time, e.g. measured signals. Pass as `past_covariates`.
- Future covariates: values known into the future up to the forecast horizon, e.g. calendar, holidays, schedules, planned prices, or validated forecasts. Pass as `future_covariates`.
- Static covariates: time-invariant features attached to a target `TimeSeries`.
- When training on multiple targets, provide one covariate `TimeSeries` per target series.
- Stack or concatenate multiple covariates into one multivariate covariate `TimeSeries`.
- Darts slices covariates based on target and covariate time axes, but the covariate span must still satisfy the chosen model, horizon, and lag requirements.

## Horizons, Chunks, and Lags

- `predict(n=...)` defines the number of future time steps to forecast after the end of the input/training series.
- `historical_forecasts(forecast_horizon=...)` defines the distance between forecast origin and predicted time in backtesting.
- Regression/global models use lagged features. Target and past covariate lags must refer to past values. Future covariate lags can include past and future positions when those future values are known.
- `output_chunk_length` is the number of time steps predicted at once by an internal global model; it is not the same as `predict(n)`.
- `output_chunk_shift` creates a gap between input history and prediction start. If set, autoregressive predictions with `n > output_chunk_length` are not allowed for the documented regression model behavior.

## Transformations

- Fit fittable transformers only on train data: `Scaler`, `BoxCox`, missing-value logic with fitted statistics, static covariate transformers, and any custom transformer with learned state.
- Apply fitted transformers forward to validation, test, and future covariates. Inverse-transform forecasts before reporting metrics on the original scale.
- Use Darts data transformers for preprocessing such as scaling, missing value filling, differencing, Box-Cox transforms, static covariates, window transforms, and hierarchical reconciliation.
- In examples, Darts explicitly avoids fitting a `Scaler` on validation data.

## Backtesting

- `historical_forecasts()` simulates forecasts that would have been made historically.
- `backtest()` computes metrics over historical forecasts and can reuse precomputed historical forecasts.
- Key parameters include `series`, `start`, `forecast_horizon`, `stride`, `last_points_only`, `metric`, `reduction`, and retraining behavior.
- Use expanding or rolling retraining only if it matches the intended production process.
- For model selection, use validation/backtesting folds; keep the final test period untouched.

## Metrics

- Deterministic aggregated metrics include MAE, MSE, RMSE, RMSLE, MASE, MSSE, RMSSE, MAPE, wMAPE, sMAPE, OPE, MARRE, R2, CV, and AUTC.
- Deterministic per-time-step metrics include error, absolute error, squared error, squared log error, absolute scaled error, squared scaled error, absolute percentage error, symmetric absolute percentage error, and absolute ranged relative error.
- Probabilistic/quantile aggregated metrics include MCRPS, MQL, QR, MIW, MWS, MIC, and MINCS_QR.
- Probabilistic/quantile per-time-step metrics include CRPS, QL, IW, WS, IC, and INCS_QR.
- Prefer MAE/RMSE for scale-dependent accuracy, MASE/RMSSE for cross-series comparisons, wMAPE for demand planning when appropriate, and bias/OPE for systematic over/underforecasting.
- Avoid MAPE/sMAPE when actuals can be zero or near zero.

## Probabilistic Forecasts and Intervals

- Probabilistic forecasts are usually generated with `predict(..., num_samples > 1)` when supported.
- Local models such as ARIMA, Exponential Smoothing, TBATS, and Kalman forecasters may make normality assumptions where documented.
- Torch models support likelihood models such as parametric likelihoods and `QuantileRegression`; quantile regression trains with pinball loss.
- Monte Carlo dropout is available for documented Torch models with dropout support using `mc_dropout=True` at prediction time.
- Use `low_quantile` and `high_quantile` in `TimeSeries.plot()` for interval visualization.
- Validate coverage with interval metrics; use conformal models when calibrated intervals are required.

## Plotting and Residual Diagnostics

- Plot actuals and forecasts with `TimeSeries.plot(label=...)`.
- Plot probabilistic intervals with `forecast.plot(label="forecast", low_quantile=..., high_quantile=...)`.
- Use `model.residuals(series, ...)` for fitted or historical forecast residuals where available.
- Use `darts.utils.statistics.plot_residuals_analysis()` for residual distribution and autocorrelation inspection.
- Prefer residuals derived from historical forecasts/backtesting over in-sample residual checks for final diagnostics.

## Anti-Leakage Checks

- Do not random split rows for forecasting validation.
- Do not fit transformers or missing-value logic on validation/test data.
- Do not use observed future values as `past_covariates`.
- Do not build lag/rolling features from future values relative to each cutoff.
- Do not use future covariates unless they are known at prediction time or forecasted by a separately validated process.
- Keep target, covariate, and static covariate transformations inside each backtest fold when model selection depends on them.
