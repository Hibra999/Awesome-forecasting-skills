# skforecast Data, Validation, and Evaluation

Use this reference when preparing data for skforecast forecasters.

## Accepted Data Formats

Single series:

- `y`: pandas `Series`.
- Index: `DatetimeIndex` with `freq`, or `RangeIndex` with regular step.
- `exog`: optional pandas `Series` or `DataFrame` aligned to `y`.

Independent multi-series with `ForecasterRecursiveMultiSeries`:

- Wide pandas `DataFrame`: each column is one series.
- Long MultiIndex pandas `DataFrame`: first index level is series ID, second is `DatetimeIndex`.
- `dict[str, pandas.Series]`: keys are series IDs and values are named series.

Docs recommend dictionaries for large or uneven multi-series workflows, especially with different lengths or different exogenous variables.

Dependent multivariate:

- Use `ForecasterDirectMultiVariate` for related columns where one target level is predicted using other series.

## Frequency, Missingness, and Gaps

Set or validate index frequency before fitting. If timestamps are irregular, resample or regularize during `forecasting-data-prep`.

Handle missing targets before fitting. skforecast has guidance for missing values and can work with delayed historical data patterns, but do not let interpolation or imputation use validation/test target values.

## Exogenous Variables

Training:

- Pass `exog` aligned to `y` or to each series.
- Use only variables available at forecast creation time.

Prediction:

- If a forecaster was fit with `exog`, provide future `exog` for exactly the requested `steps`.
- For multi-series with different exogenous variables, prefer dictionary inputs as documented.

Valid future exog examples: calendar features, holidays, scheduled promotions, planned prices, capacity. Observed future weather, demand drivers, or target-derived signals are leakage unless separately forecast and validated.

## Lags, Window Features, and Transformations

Use forecaster `lags`, `RollingFeatures`, and custom feature mechanisms instead of manually building features on full data. If building manually:

- sort by time and series;
- use `shift` before rolling/expanding;
- compute features separately inside each temporal fold.

Use `transformer_y`, `transformer_exog`, differentiation, categorical encoders, and feature selection only on train data or inside skforecast backtesting/tuning folds.

## Forecast Horizon and Backtesting

Key validation utilities:

- `TimeSeriesFold`
- `OneStepAheadFold`
- `backtesting_forecaster`
- `backtesting_forecaster_multiseries`
- `backtesting_stats`
- `grid_search_forecaster`
- `random_search_forecaster`
- `bayesian_search_forecaster`
- multiseries/stat-specific search variants

Key parameters:

- `steps`: forecast horizon per fold.
- `initial_train_size`: first training window length.
- `refit`: whether and how often to refit.
- `fixed_train_size`: rolling versus expanding training window.
- `gap`: holdout gap between train end and forecast start when supported/needed.

Use temporal cutoffs only. Keep final test untouched until final reporting.

## Metrics

Official examples use sklearn metrics such as mean squared error and mean absolute error plus skforecast metric utilities.

Recommended:

- MAE/RMSE/MSE for scale-dependent errors.
- MASE/RMSSE for cross-series comparison.
- WAPE/sMAPE for reporting; avoid MAPE when actuals can be zero or near zero.
- Bias/mean error for systematic over/under-forecasting.
- Pinball loss, interval coverage, interval width, and CRPS for probabilistic forecasts.

## Plotting, Explainability, and Residual Diagnostics

Use:

- pandas/matplotlib plots for train/test/prediction lines.
- `plot_prediction_intervals` for intervals.
- skforecast plotting utilities and `set_dark_theme`.
- `get_feature_importances()` where supported by the estimator/forecaster.
- SHAP or estimator-specific explainability where documented.
- drift detection utilities for production monitoring.

Residual diagnostics are manual: join backtest predictions to actuals, compute residuals by horizon, timestamp, series, and feature slices. Do not tune against final test residuals.

## Deployment Notes and Limitations

- Save/load forecasters with documented deployment utilities.
- Pin skforecast version; 0.22.0 is the stable docs version checked.
- Basic install does not include stats, plotting, or deep learning extras.
- First runs of Numba-accelerated statistical models may be slower due to JIT compilation.
- Foundation and RNN workflows require additional dependencies, hardware/API checks, and explicit validation against simpler baselines.
- Not all estimators expose feature importances, quantiles, intervals, or probabilistic outputs.
