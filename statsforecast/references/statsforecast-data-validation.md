# StatsForecast Data, Validation, and Evaluation

Use this reference when turning a prepared dataset into a StatsForecast workflow.

## Accepted Data Formats

StatsForecast expects long format:

- `unique_id`: series identifier, string/int/category.
- `ds`: datestamp or integer time index.
- `y`: numeric target.

You can pass custom column names to core methods with `id_col`, `time_col`, and `target_col`. Use pandas by default. Polars is documented with the `polars` extra. Distributed examples cover Dask, Ray, and Spark.

Validate before modeling:

- unique `(unique_id, ds)` keys;
- sorted `ds` within each series;
- no missing target values unless a chosen preprocessing step handles them before StatsForecast;
- enough observations per series for `h`, `season_length`, and conformal windows;
- consistent frequency per series, or explicit handling for ragged panel ends.

## Forecast Horizon and Frequency

- `h`: forecast horizon in number of future periods.
- `freq`: pandas-style frequency string for datestamps, or integer such as `1` for integer datestamps.
- `season_length`: observations per seasonal period, for example 7 for daily weekly seasonality, 12 for monthly yearly seasonality, 24 for hourly daily seasonality.

Keep `h`, `freq`, `season_length`, and business reporting periods aligned. StatsForecast will generate future `ds` values from `freq`; wrong frequency silently produces wrong timestamps.

## Fit/Predict vs Forecast

Use `StatsForecast.forecast(df, h, ...)` for memory-efficient forecasting. It does not store fitted model objects and is compatible with distributed backends.

Use `StatsForecast.fit(df)` then `predict(h, ...)` when you need to store fitted models, use `save()`/`load()`, call model-specific fitted-value methods, or issue repeated predictions from the fitted state.

Use `fit_predict` when you need the fit/predict interface in one step. Check the core API for parameters before using it.

## Exogenous Regressors and Static Covariates

Official docs state every column after `y` is considered an external regressor and passed to models that support them. For forecasting:

- Training `df` contains `unique_id`, `ds`, `y`, and regressor columns.
- `X_df` contains `unique_id`, `ds`, and future values of the same regressor columns for every horizon row.
- For validation, split train/test first, then build `X_df` from the future portion without `y`.

Only use future regressors that are known at prediction time. Calendar fields, holidays, scheduled promotions, and planned prices can be valid. Realized weather, realized demand drivers, and future target-derived features are leakage unless separately forecast and validated.

Static covariates are supported according to README/features, but their exact handling is guide/model dependent. Verify the relevant official guide/source before relying on them.

## Feature Generation

The official feature-generation guide shows `mstl_decomposition(train, model=MSTL(...), freq=..., h=...)`, which returns a transformed training frame and future `X_df` with trend/seasonal features.

Treat any decomposition, lag, rolling, calendar encoding, scaling, and model selection as train-only. Recompute those features inside each cross-validation fold for leakage-safe backtests.

## Backtesting and Temporal Validation

Use:

```python
cv_df = sf.cross_validation(df=train_df, h=horizon, n_windows=3, step_size=horizon)
```

The result contains `unique_id`, `ds`, `cutoff`, `y`, and one column per model. `n_windows=1` mirrors a temporal train/test split. Larger `n_windows` gives repeated sliding-window evaluation.

Do not random split rows. If panel series end at different times, define cutoffs per series or restrict to comparable windows.

## Metrics

Official examples use `utilsforecast.evaluation.evaluate` and `utilsforecast.losses` such as `mse`, `mae`, `smape`, and `mase`.

Recommended:

- MAE/RMSE/MSE for scale-dependent accuracy.
- MASE/RMSSE for cross-series comparability.
- sMAPE or WAPE for business reporting; avoid MAPE for granular or zero-heavy data.
- Coverage, interval width, and pinball-style losses for interval workflows.
- Bias/mean error for systematic over/under-forecasting.

## Probabilistic Forecasts and Intervals

Pass `level=[80, 90, 95]` to `forecast`, `predict`, or `cross_validation` where supported. Output interval columns follow model names such as `AutoARIMA-lo-95` and `AutoARIMA-hi-95`.

For conformal prediction, use `statsforecast.utils.ConformalIntervals(h=..., n_windows=...)` either on the model constructor or in `forecast(..., prediction_intervals=...)`. Docs state `n_windows * h` must be less than the series length and recommend `n_windows >= 2`.

Validate interval coverage and width on cross-validation windows; do not assume nominal coverage in production.

## Plotting and Residual Diagnostics

Use `StatsForecast.plot(df, forecasts_df, level=[...], engine="matplotlib"|"plotly")`. The docs also use `utilsforecast.plotting.plot_series`.

Residual diagnostics are not one universal high-level API. Use held-out or cross-validation outputs:

1. Join predictions with actual `y`.
2. Compute residuals by model, horizon, cutoff, series, and relevant covariate slices.
3. Inspect bias, autocorrelation, heteroskedasticity, outliers, and interval misses.
4. Tune on validation/backtest only, then report final test once.

## Documented Limitations to Surface

- StatsForecast local models are univariate per `unique_id`; panels scale local modeling, not joint multivariate endogenous dynamics.
- Exogenous support is model-specific. Passing `X_df` to unsupported models will not make them exogenous models.
- `forecast()` does not store fitted models. Use `fit()`/`predict()` when fitted state matters.
- Distributed backends are intended for scalable forecasts and may not support every stateful local method.
- Conformal intervals require enough history and calibration windows.
- Numba JIT means first runs can be slower than subsequent runs.
