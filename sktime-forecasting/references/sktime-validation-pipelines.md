# sktime Validation, Pipelines, and Diagnostics

Use this reference before building sktime pipelines, tuning, exogenous-variable workflows, probabilistic forecasts, or panel/hierarchical validation.

## Data Formats

- `Series` scitype: one time series, univariate or multivariate. Common mtypes include `pd.Series`, `pd.DataFrame`, `np.ndarray`, xarray, dask, polars, and GluonTS representations.
- `Panel` scitype: collection of multiple time series. The recommended pandas form is `pd.DataFrame` with a two-level `MultiIndex`, where the last level is the time-like index.
- `Hierarchical` scitype: hierarchical collection of time series. The recommended pandas form is `pd.DataFrame` with a three-or-more-level `MultiIndex`, where the last level is the time-like index and previous levels identify hierarchy nodes.
- Time-like index can be integer/range-like, `DatetimeIndex`, or `PeriodIndex`; keep it monotonic and consistent with the prepared frequency.
- Use `sktime.datatypes.MTYPE_REGISTER` when exact mtype names matter.

## Exogenous Variables

- `X` is optional and passed to `fit`, `predict`, `update`, and probabilistic prediction methods where supported.
- If `X` is passed to `fit`, future `X` must be passed to `predict` unless using a documented composition that forecasts `X`, such as `ForecastX`.
- `X` in `fit` should align with `y`; future `X` should cover every requested horizon timestamp.
- Treat calendar, holidays, scheduled promotions, planned capacity, and contractual prices as known-future only when they are known at forecast creation time.
- Treat observed weather, realized demand drivers, and operations measurements as observed-past unless a valid future source is available.

## Forecasting Horizon

- Use relative horizons such as `[1, 2, 3]` for regular next-step forecasts.
- Use `ForecastingHorizon(index, is_relative=False)` for absolute timestamp forecasts against a prepared test index.
- Some forecasters require `fh` in `fit`; inspect `requires-fh-in-fit`.
- For backtesting, keep horizon, cutoff, gap, and frequency consistent across folds. Convert absolute horizons to relative horizons when constructing splitters from a training cutoff.

## Pipelines and Transformations

- Use `TransformedTargetForecaster` for transformations of `y` and inverse transformation of forecasts.
- Use `ForecastingPipeline` for transformations of `X` before passing it to the forecaster.
- Use nested pipelines when both `y` and `X` transformations are required.
- Use sktime pipeline composition (`*` for target pipelines, `**` for exogenous pipelines) only when it stays readable.
- Fit transformations inside the temporal training window only. For grid search or backtesting, wrap transformations inside the estimator being evaluated so each fold refits correctly.

## Backtesting and Tuning

- `temporal_train_test_split` is a quick single holdout utility. Prefer full backtesting for model selection.
- Use temporal splitters such as `SingleWindowSplitter`, `SlidingWindowSplitter`, `ExpandingWindowSplitter`, `CutoffSplitter`, `CutoffFhSplitter`, and related splitters for evaluation and tuning.
- Use `ForecastingGridSearchCV`, `ForecastingRandomizedSearchCV`, `ForecastingSkoptSearchCV`, or `ForecastingOptunaSearchCV` with a temporal splitter. Avoid sklearn random CV splitters for forecasting rows.
- Use `MultiplexForecaster` or forecaster-valued grids when selecting among model classes under the same splitter and metric.
- Use `evaluate(forecaster, cv, y, X=..., strategy=...)` for repeated temporal evaluation.

## Metrics

- Point metrics available in sktime include function and class forms such as `mean_absolute_error`, `mean_squared_error`, `mean_absolute_percentage_error`, `mean_absolute_scaled_error`, `MeanAbsoluteScaledError`, relative loss utilities, and asymmetric/linex losses.
- Prefer MAE/RMSE for scale-dependent errors; MASE/RMSSE or relative loss for scale-free comparison; WAPE externally when needed for demand planning.
- Use MAPE/sMAPE only when zeros and near-zeros are not a meaningful risk.
- Quantile/interval metrics include `PinballLoss`, `EmpiricalCoverage`, `ConstraintViolation`, and `IntervalWidth`.
- Distribution metrics include `AUCalibration`, `CRPS`, `LogLoss`, and `SquaredDistrLoss`.

## Probabilistic Forecasts

- Probabilistic methods are `predict_interval(fh=None, X=None, coverage=0.90)`, `predict_quantiles(fh=None, X=None, alpha=[0.05, 0.95])`, `predict_var(fh=None, X=None, cov=False)`, and `predict_proba(fh=None, X=None, marginal=True)`.
- Verify support with `capability:pred_int` and `capability:pred_var`; not every forecaster supports every probabilistic method.
- Interval outputs use a column MultiIndex with variable, coverage, and lower/upper levels.
- Quantile outputs use a column MultiIndex with variable and alpha levels.
- For panel/hierarchical data, sktime can vectorize probabilistic forecasts when supported by the component forecaster.

## Panel, Global, and Hierarchical Forecasting

- sktime provides a unified interface for panel and hierarchical forecasts.
- All sktime forecasters can be applied to panel/hierarchical data; forecasters that are not genuinely panel or hierarchical are applied by instance.
- Inspect `forecasters_` after fit when vectorization creates separate fitted forecasters by instance, variable, or hierarchy node.
- For hierarchical reconciliation, use documented tools such as `Aggregator`, reconciliation transformers, and `ReconcilerForecaster`; preserve the special `__total` convention for aggregate nodes.
- Do not confuse multivariate columns with panel instances. Columns are variables; MultiIndex non-time levels are instances or hierarchy nodes.

## Plotting and Diagnostics

- Plot point forecasts with `sktime.utils.plotting.plot_series(y_train, y_test, y_pred, labels=[...])`.
- Plot interval forecasts by passing `pred_interval` to `plot_series`.
- Plot temporal split windows with `sktime.utils.plotting.plot_windows(cv=cv, y=y)`.
- Inspect `get_fitted_params()` for fitted model parameters exposed by the forecaster.
- For residual diagnostics, compute out-of-sample residuals from temporal validation and use model-specific wrapped estimator diagnostics only when documented. There is no single residual-diagnostic API that applies uniformly to every sktime forecaster.

## Common Failure Modes

- Future `X` missing, misaligned, or not available at prediction time.
- Wrong `fh` type or absolute horizon dates outside the expected frequency.
- Transformations fit on full data before temporal validation.
- Treating panel IDs as columns instead of MultiIndex instances.
- Expecting one global model when automatic vectorization is actually fitting one model per instance.
- Installing `sktime` core only and then using forecasters that require uninstalled packages such as pmdarima, statsforecast, prophet, neuralforecast, pytorch-forecasting, transformers, or other wrappers.
