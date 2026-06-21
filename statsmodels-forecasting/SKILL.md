---
name: statsmodels-forecasting
description: Use Statsmodels for classical and econometric forecasting with AR/ARIMA/SARIMAX, exponential smoothing/ETS, state space models, VAR/SVAR/VECM/VARMAX, dynamic factor models, ARDL/UECM, ThetaModel, STLForecast, exogenous regressors, prediction intervals, temporal validation, and residual diagnostics. Trigger when an agent needs to model prepared time-series data with official statsmodels APIs after applying forecasting-data-prep for frequency, horizon, splits, covariates, and anti-leakage checks.
---

# Statsmodels Forecasting

Use this skill after `forecasting-data-prep`. Statsmodels is best for classical statistical forecasting, econometric time-series models, state space modeling, interpretable regression-with-time-series-errors, and residual diagnostics. It is not a native global, hierarchical, or panel forecasting library.

## Minimum Install

```bash
python -m pip install statsmodels
conda install -c conda-forge statsmodels
```

Core ecosystem dependencies are NumPy, SciPy, pandas, and patsy. Use pandas indexes whenever date-aware forecasting matters.

## Data Contract

- Use a pandas `Series` or `DataFrame` with a sorted `DatetimeIndex` or `PeriodIndex` and fixed frequency when possible.
- `endog` is the target: 1-dimensional for univariate models; 2-dimensional for VAR, VECM, VARMAX, and dynamic factor workflows.
- `exog` must align row-for-row with `endog`; if a fitted model uses `exog`, provide future `exog` for every forecast step.
- For date-like forecast endpoints, keep a fixed-frequency index. If the index has no fixed frequency, use integer `steps`.
- Statsmodels does not provide native panel/global multi-series forecasting. For independent series IDs, fit one model per series and aggregate metrics.
- Prefer `missing="raise"` after data prep unless deliberately using a state space model feature that handles missing observations.

Run the bundled checker when converting prepared data:

```bash
python statsmodels-forecasting/scripts/statsmodels_contract_check.py data.csv \
  --time-col ds \
  --target-cols y \
  --exog-cols promo,price \
  --freq D \
  --horizon 14 \
  --future-exog-confirmed
```

## Model Selection

- Autoregression: `AutoReg`, `ar_select_order`.
- ARIMA family: `ARIMA` for AR, MA, ARMA, ARIMA, SARIMA, and regression with ARIMA errors; `SARIMAX` for state-space seasonal ARIMA with exogenous regressors and advanced state options.
- Exponential smoothing: `SimpleExpSmoothing`, `Holt`, `ExponentialSmoothing`, `ETSModel`, and state-space `ExponentialSmoothing`.
- Structural/state space: `UnobservedComponents`, custom `MLEModel`.
- Multivariate: `VAR`, `SVAR`, `VECM`, `VARMAX`, `DynamicFactor`, `DynamicFactorMQ`.
- Regression with lagged variables: `ARDL`, `UECM`.
- Forecasting helpers: `ThetaModel`, `STLForecast`.
- Regime switching: `MarkovRegression`, `MarkovAutoregression`.

Read `references/statsmodels-model-map.md` before choosing among similar APIs.

## Modeling Workflow

1. Prepare data with `forecasting-data-prep`; preserve `freq`, horizon, cutoff timestamps, gap, known-future covariates, and excluded leaky columns.
2. Split train/validation/test by time before fitting transformations. Never random split.
3. Build `endog` and optional `exog` from train only; fit imputers, scalers, encoders, and outlier rules on train only.
4. Select the smallest model family that matches the data contract: univariate, multivariate endogenous system, or per-series loop.
5. Fit with the official model class and `.fit()`.
6. Forecast with `results.forecast(steps=horizon, ...)` for point forecasts or `results.get_forecast(...).summary_frame()` when intervals are supported.
7. Validate with temporal holdout, rolling-origin, or expanding-window backtesting. Use test once for final reporting.

## Python Pattern

```python
import statsmodels.api as sm

y_train = train_df.set_index(time_col)[target_col].asfreq(freq)
X_train = train_df.set_index(time_col)[known_future_exog].asfreq(freq)
X_future = future_df.set_index(time_col)[known_future_exog].asfreq(freq)

mod = sm.tsa.SARIMAX(
    y_train,
    exog=X_train,
    order=(p, d, q),
    seasonal_order=(P, D, Q, s),
    trend="c",
    missing="raise",
)
res = mod.fit()
pred = res.get_forecast(steps=horizon, exog=X_future)
forecast = pred.summary_frame()
```

Use `ARIMA` for regression with ARIMA errors, `SARIMAX` for state-space SARIMA/SARIMAX behavior, `VAR` or `VARMAX` only when targets are jointly endogenous, and one model per ID for panel-like data.

## Validation, Metrics, and Diagnostics

- Backtest by cutoff. For state space results, use `append(refit=True)` when re-estimating parameters or `extend` when only filtering new observations.
- Recompute features and transformations independently inside each fold.
- Recommended metrics: MAE, RMSE, MASE/RMSSE, WAPE, bias, interval coverage, and horizon-by-horizon error. Avoid MAPE/sMAPE when actuals can be zero or near zero.
- Use AIC/BIC/log likelihood only for comparing compatible fitted statistical models; do not treat them as forecast accuracy.
- Plot forecasts against actuals and intervals. For state space results, use `plot_diagnostics()` when available.
- Inspect `summary()`, `resid`, residual ACF/PACF, Ljung-Box/autocorrelation checks, normality, heteroskedasticity, and model-specific component plots.

## Anti-Leakage Rules

- Never use random split for forecasting.
- Fit scalers, imputers, encoders, outlier thresholds, and target transforms on train only.
- Create lag and rolling features using only past observations available at each cutoff.
- Use future `exog` only when it is genuinely known at prediction time for every horizon step.
- Respect forecast horizon, gap, timestamp cutoff, frequency, and valid timestamp windows.
- During backtesting, rebuild future covariates and features per cutoff.

## Common Errors

- Using date forecast endpoints with an index that lacks fixed frequency.
- Omitting future `exog` after fitting with `exog`, or misaligning `endog` and `exog`.
- Treating multiple independent IDs as one multivariate endogenous system.
- Fitting preprocessing on the full dataset before temporal splitting.
- Assuming every smoothing model provides intervals: Holt-Winters `ExponentialSmoothing` does not provide confidence intervals like the state-space alternatives.
- Assuming Statsmodels has a documented `auto_arima` API. Use documented selectors such as `ar_select_order`, `arma_order_select_ic`, `ardl_select_order`, or VAR `select_order`, or choose orders through temporal validation.

## References

- Read `references/statsmodels-model-map.md` for documented forecasting model families, data shapes, exog support, intervals, and limitations.
- Read `references/statsmodels-validation-diagnostics.md` for backtesting, forecasting methods, intervals, plotting, diagnostics, and metrics.
- Read `references/official-sources.md` for official sources consulted.

## Ready Checklist

- `forecasting-data-prep` contract is complete and leakage risks are resolved or documented.
- Time index is sorted, duplicate-free per modeled series, fixed-frequency or forecasted with integer steps.
- Target and exog arrays are numeric, aligned, finite, and split by temporal cutoff.
- Future exogenous values are available for the full horizon or excluded.
- Model family matches the actual data shape: univariate, multivariate endogenous, or per-ID loop.
- Validation uses temporal cutoffs/backtesting and reports horizon-aware metrics plus residual diagnostics.
