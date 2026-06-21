# Greykite Data, Validation, and Evaluation

Use this reference when turning a prepared forecasting dataset into a Greykite workflow.

## Accepted Data Format

The mainstream workflow uses a pandas `DataFrame` with:

- one timestamp column, configured by `MetadataParam.time_col`;
- one numeric target column, configured by `MetadataParam.value_col`;
- optional regressor columns;
- optional future rows for prediction when regressors are needed.

`MetadataParam.freq` should be a pandas frequency string. Provide it explicitly when there are missing timepoints. `date_format` can parse nonstandard timestamps. `train_end_date` defines the last timestamp used for fitting before forecasts begin.

`result.timeseries` loads the input into Greykite's `UnivariateTimeSeries` object. This confirms the main forecasting path is one target series at a time.

## Univariate, Multivariate, and Panel Use

- Univariate: first-class path through `Forecaster`.
- Multivariate endogenous targets: not documented as a main `Forecaster` capability.
- Multiple independent series/panel: not documented as one container. Loop over IDs externally and build separate configs/results per series.
- Hierarchical/additive multiple forecasts: use `ReconcileAdditiveForecasts` only after producing base forecasts.

When looping over IDs, split and feature-engineer per ID. Do not compute global target-derived features using future rows from any series.

## Forecast Horizon, Frequency, and Evaluation Periods

- `forecast_horizon`: number of periods ahead; must be positive. If omitted, Greykite chooses a default from frequency.
- `coverage`: requested prediction interval coverage between 0 and 1, or `None` for no intervals.
- `test_horizon`: periods held out at the end of `df` for backtest.
- `periods_between_train_test`: gap between train and test/forecast.
- `cv_horizon`: periods in each CV test set.
- `cv_min_train_periods`: minimum training length per CV fold.
- `cv_expanding_window`: expanding versus sliding training window.
- `cv_periods_between_splits`: how far to slide between CV splits.
- `cv_periods_between_train_test`: gap inside each CV split.
- `cv_max_splits`: maximum CV splits.

Greykite defaults are designed to align `forecast_horizon`, `test_horizon`, and `cv_horizon`, but verify them against the data-prep contract.

## Regressors and Events

Use `model_components.regressors` for high-level Silverkite and Prophet:

- Silverkite: `regressors={"regressor_cols": [...]}`.
- Prophet: `regressors={"add_regressor_dict": {...}}`.
- Low-level `SK`: use `custom={"extra_pred_cols": [...]}`.

Regressors can be numeric or categorical. Historical values are needed for training and future values are needed for prediction. A regressor is valid only if its future value is known at the time the forecast is created, or if it comes from a separately validated upstream forecast.

Use `model_components.lagged_regressors` for Silverkite lagged regressors. Prophet does not support lagged regressors according to docs.

Events/holidays should be configured through `model_components.events`; custom events must be known in advance for the forecast horizon.

## Autoregression and Leakage

Silverkite supports autoregression through `model_components.autoregression`.

The `"auto"` setting activates autoregression when the forecast horizon is less than or equal to 30 days. Official docs state automatic lags are greater than or equal to the forecast horizon to avoid simulation for forecasts and prediction intervals.

Custom lags smaller than `forecast_horizon` require simulation. Ensure any manual lag or aggregated-lag feature uses only observations available before the forecast creation timestamp.

## Training, Prediction, and Outputs

Main call:

```python
result = Forecaster().run_forecast_config(df=df, config=config)
```

`ForecastResult` contains:

- `timeseries`: processed `UnivariateTimeSeries`.
- `grid_search`: time-series CV/model selection result.
- `backtest`: holdout `UnivariateForecast`.
- `forecast`: final forecast `UnivariateForecast`.
- `model`: fitted sklearn `Pipeline`.

`backtest.df` and `forecast.df` contain forecasted values and prediction intervals when requested.

## Metrics, Plotting, and Diagnostics

Use:

- `summarize_grid_search_results(...)` for CV summaries and correct ranks.
- `result.backtest.train_evaluation` and `result.backtest.test_evaluation` for holdout metrics.
- `result.forecast.train_evaluation` only as in-sample fit diagnostics.
- `backtest.plot()`, `forecast.plot()`, `plot_components()`, and `plot_grouping_evaluation(...)`.

`EvaluationMetricEnum` provides the metric functions used by the framework. Common documented examples include mean absolute percent error and root mean squared error; coverage and interval metrics are available for interval workflows. Add WAPE, bias, MASE/RMSSE, and business metrics externally when needed. Avoid percent metrics when actuals can be zero or near zero.

Residual diagnostics are available through component plots and manual residual calculations from `backtest.df`/CV output. Use held-out residuals for model decisions.

## Intervals and Probabilistic Forecasting

Set `coverage` to produce prediction intervals. Silverkite and Prophet support intervals.

Silverkite uncertainty can be configured with `model_components.uncertainty`; the implemented method documented is `simple_conditional_residuals`, with parameters such as `conditional_cols`, `quantile_estimation_method`, and small-sample handling.

Prophet interval behavior is configured with `mcmc_samples` and `uncertainty_samples`. Setting `uncertainty_samples` to 0 or False disables uncertainty estimation.

Quantile loss is a Silverkite fitting option, but do not present Greykite as a full probabilistic distribution forecasting library. Validate empirical coverage and interval width on backtests.

## Documented Limitations to Surface

- Main forecasting workflow is univariate; no generic panel container is documented.
- Future regressor rows must be supplied when regressors are used.
- Prophet does not support lagged regressors in Greykite docs.
- Low-level `SK` regressor syntax differs from high-level Silverkite syntax.
- Hyperparameter search and CV can be computationally expensive; use `ComputationParam`.
- Current setup metadata pins older pandas/scikit-learn versions, so dependency compatibility matters.
- `AUTO_ARIMA` depends on pmdarima behavior and is mainly a classic baseline.
