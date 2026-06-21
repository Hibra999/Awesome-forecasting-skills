# Statsmodels Validation and Diagnostics

Use this reference when implementing evaluation, backtesting, intervals, plots, and residual checks.

## Forecasting APIs

- `results.forecast(steps=...)`: point forecasts. Use integer `steps` when the index has no fixed frequency.
- `results.get_forecast(steps=...)`: out-of-sample forecasts with a prediction result object and intervals when supported.
- `results.get_prediction(...)`: in-sample predictions and out-of-sample forecasts with prediction results when supported.
- `PredictionResults.summary_frame()`: common way to retrieve means and interval columns.
- `predict(start=..., end=..., dynamic=...)`: available on several results/model classes, but behavior differs by family. Check the selected class docs.

When the model uses `exog`, provide correctly shaped future `exog` to forecast/predict methods for the entire horizon.

## Temporal Validation

Use chronological validation only:

1. Choose initial train cutoff, validation horizon, optional gap, and final test period.
2. Fit all transformations and model parameters on data at or before the cutoff.
3. Build lags, rolling features, and future exog from information available at that cutoff.
4. Forecast exactly the planned horizon.
5. Store errors by cutoff and horizon.
6. Advance the cutoff with rolling-origin or expanding-window logic.

For state space results:

- `append(refit=True)` adds new observations and re-estimates parameters.
- `append(refit=False)` or `extend` updates the results without re-estimating parameters.
- Official examples note `extend` is faster but can differ from refitting because parameters are not re-estimated.

## Metrics

Prefer metrics that match the business loss and the data scale:

- MAE: robust, easy to explain.
- RMSE: penalizes large errors.
- MASE or RMSSE: scale-free comparison against a naive benchmark.
- WAPE: useful for aggregate demand/load contexts.
- Bias or mean error: checks systematic over/under-forecasting.
- Coverage and interval width: evaluate prediction intervals.
- Horizon-by-horizon error: required when accuracy decays across horizon.

Use AIC, BIC, HQIC, and log likelihood only to compare compatible statistical models fitted to the same training response and likelihood setup. They do not replace out-of-sample temporal validation.

Avoid MAPE and sMAPE when actual values can be zero, negative, intermittent, or close to zero.

## Plotting

Minimum diagnostic plots:

- Actuals vs fitted values over train.
- Actuals vs forecasts over validation/test, with interval bands when available.
- Residual time plot.
- Residual ACF/PACF.
- Histogram or Q-Q plot of residuals.

Useful Statsmodels methods include `summary()`, `plot_diagnostics()` for many state-space results, model-specific component plots such as unobserved-components plots, and ACF/PACF plotting utilities.

## Residual Diagnostics

Check whether residuals still contain forecastable structure:

- Autocorrelation: ACF/PACF, Ljung-Box or `q_stat`.
- Stationarity context: ADF/KPSS on series or residuals when relevant.
- Normality: Jarque-Bera or visual Q-Q checks when interval assumptions require it.
- Heteroskedasticity: residual variance over time and documented heteroskedasticity tests where applicable.
- Structural breaks or regime changes when residual patterns shift.

If diagnostics fail, prefer changing the model specification, seasonality, transformations, exog design, or validation assumptions before tuning blindly.

## Leakage Traps

- Fitting Box-Cox/log scaling/imputation/outlier capping on all data before splitting.
- Selecting ARIMA orders using validation/test observations.
- Building rolling statistics centered on the current timestamp or including the target at the prediction timestamp.
- Passing future realized exog that would not have been known at the cutoff.
- Using one multivariate model for independent panel IDs merely because they share a timestamp column.
- Dropping missing values after exog/endog alignment in a way that changes the effective training horizon silently.
