---
name: sktime-forecasting
description: Use sktime for forecasting with its unified forecaster API, ForecastingHorizon, temporal splitters, pipelines, tuning, reductions to regression, statistical/deep/foundation/wrapped forecasters, exogenous variables, probabilistic forecasts, panel/global/hierarchical data, and leakage-safe backtesting. Trigger when an agent needs to model prepared time-series data with sktime after applying forecasting-data-prep for frequency, horizon, splits, covariates, and anti-leakage checks.
---

# sktime Forecasting

Use this skill after `forecasting-data-prep`. `sktime` is best when you need a scikit-learn-like forecasting interface across many model families, pipelines, hyperparameter search, temporal backtesting, probabilistic forecasts, or panel/hierarchical workflows on single-machine pandas/NumPy-scale data.

## Minimum Install

```bash
python -m pip install sktime
python -m pip install "sktime[forecasting]"
python -m pip install "sktime[all_extras]"
conda install -c conda-forge sktime
```

Install estimator-specific soft dependencies only when needed. `all_extras` is not required for most workflows and may fail on some platforms.

## Data Contract

- Start from a `forecasting-data-prep` contract: sorted time index, no duplicate time keys per series, documented frequency, horizon, cutoffs, known-future covariates, and leakage notes.
- Use `y` as the endogenous target: `pd.Series` for one univariate series, `pd.DataFrame` for uni/multivariate series, or a `pd.DataFrame` with `MultiIndex` for panel/hierarchical data.
- For panel/hierarchical data, make the last index level the time-like index; earlier levels identify instances or hierarchy nodes.
- Use `X` only for exogenous variables. `X` is a `pd.DataFrame`; if passed to `fit`, pass aligned future `X` to `predict` unless using a documented composition such as `ForecastX`.
- Do not assume every forecaster supports multivariate `y`, panel/hierarchical data, probabilistic output, or `X`. Inspect tags with `sktime.registry.all_estimators` and `all_tags`.
- Use `ForecastingHorizon` for absolute horizons; use relative integer horizons for regular next-step forecasts.

## Model Selection

- Always benchmark with `NaiveForecaster` or another simple baseline before using complex models.
- Use classical/statistical forecasters for short-to-medium univariate series with trend/seasonality: `ThetaForecaster`, `ExponentialSmoothing`, `AutoETS`, `ARIMA`, `AutoARIMA`, `SARIMAX`, `BATS`, `TBATS`.
- Use `VAR`, `VARMAX`, `VECM`, or `DynamicFactor` only when multiple targets are jointly endogenous and the chosen class supports that shape.
- Use reduction forecasters or `make_reduction` when you want scikit-learn regressors with lag windows, direct/recursive/dirrec strategies, and optional `X`.
- Use panel/global/hierarchical workflows when the data has multiple related series; expect automatic vectorization for forecasters that are not genuinely global/hierarchical.
- Use deep, foundation, AutoML, and third-party adapters only after checking soft dependencies, documented input shape, and installed estimator tags.

Read `references/sktime-model-map.md` before selecting among model families or claiming model support.

## Workflow

1. Prepare data with `forecasting-data-prep`; preserve frequency, horizon, cutoffs, gap, panel keys, and known-future covariate classification.
2. Split by time with `temporal_train_test_split` for a holdout, or use `ExpandingWindowSplitter`, `SlidingWindowSplitter`, or related splitters for backtesting.
3. Build `y_train`, `y_test`, optional `X_train`, `X_test`; never fit transformations before the temporal split.
4. Select a forecaster by data shape and tags. For transformations, use `TransformedTargetForecaster` for `y` and `ForecastingPipeline` for `X`.
5. Fit with `forecaster.fit(y_train, X=X_train, fh=fh)` if the forecaster requires or benefits from `fh` at fit; otherwise pass `fh` at `predict`.
6. Predict point forecasts with `predict(fh=fh, X=X_future)`.
7. For probabilistic forecasts, call documented methods only when supported: `predict_interval`, `predict_quantiles`, `predict_var`, or `predict_proba`.
8. Evaluate with temporal cutoffs/backtesting and keep the test period for final reporting.

## Python Pattern

```python
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
from sktime.split import ExpandingWindowSplitter, temporal_train_test_split
from sktime.utils.plotting import plot_series

y_train, y_test, X_train, X_test = temporal_train_test_split(
    y, X=X, test_size=horizon
)
fh = ForecastingHorizon(y_test.index, is_relative=False)

forecaster = NaiveForecaster(strategy="last", sp=seasonal_period)
forecaster.fit(y_train, X=X_train)
y_pred = forecaster.predict(fh=fh, X=X_test)

score = mean_absolute_scaled_error(y_test, y_pred, y_train=y_train)
plot_series(y_train, y_test, y_pred, labels=["train", "test", "forecast"])

cv = ExpandingWindowSplitter(
    fh=fh.to_relative(cutoff=y_train.index[-1]),
    initial_window=initial_window,
    step_length=step_length,
)
```

Use `evaluate(forecaster, cv, y, X=...)` or `ForecastingGridSearchCV` with a temporal splitter for backtesting and tuning.

## Validation, Metrics, and Plotting

- Prefer rolling-origin or expanding-window backtesting over one split for model selection.
- Recommended point metrics: MAE, RMSE, MASE/RMSSE, WAPE or relative loss against a baseline; avoid MAPE/sMAPE when actuals can be zero or near zero.
- Recommended probabilistic metrics: pinball loss for quantiles, empirical coverage and interval width for intervals, CRPS or log loss for distribution forecasts.
- Plot `y_train`, `y_test`, `y_pred` with `sktime.utils.plotting.plot_series`; pass `pred_interval` for interval plots.
- Inspect `get_fitted_params()` and documented wrapped-model diagnostics where available. `sktime` does not provide one universal residual-diagnostics interface for every forecaster.

Read `references/sktime-validation-pipelines.md` before building pipelines, tuning, probabilistic evaluation, or panel/hierarchical validation.

## Anti-Leakage Rules

- Never random split forecasting rows for validation.
- Fit scalers, imputers, encoders, detrenders, deseasonalizers, feature selectors, and target transforms on train only; use sktime pipelines or refit inside each backtest fold.
- Create lag, rolling, expanding, and window features using only data available before each forecast cutoff.
- Use future `X` only when the covariate is genuinely known at prediction time for every horizon step; otherwise forecast it separately and account for that uncertainty.
- Respect forecast horizon, gap, frequency, cutoff timestamps, panel instance boundaries, and valid timestamp windows.
- During backtesting, rebuild features and future covariates separately for every cutoff.

## Common Errors

- Installing only core `sktime` and then using a forecaster that needs an uninstalled soft dependency.
- Passing `X` to `fit` but omitting future `X` at `predict`.
- Treating unrelated independent series as multivariate endogenous variables instead of panel instances.
- Assuming all forecasters support intervals, multivariate `y`, panel/global fitting, or exogenous variables.
- Passing an absolute horizon whose dates do not match the prepared frequency.
- Tuning on transformed full data instead of wrapping transformations inside `TransformedTargetForecaster` or `ForecastingPipeline`.

## References

- Read `references/sktime-model-map.md` for official forecaster categories, supported model names, tags, and selection guidance.
- Read `references/sktime-validation-pipelines.md` for data formats, `X`, `ForecastingHorizon`, temporal validation, metrics, probabilistic forecasts, plotting, and diagnostics.
- Read `references/official-sources.md` for official sources consulted.

## Ready Checklist

- `forecasting-data-prep` contract is complete and leakage risks are resolved or documented.
- `y`, optional `X`, frequency, horizon, panel/hierarchy index levels, and future covariate availability are explicit.
- Chosen forecaster supports the required `y` scitype, `X`, probabilistic output, and panel/global behavior according to tags or API docs.
- Train/validation/test or backtest folds use temporal cutoffs, not random splits.
- Transformations and model selection are fit inside the training data of each fold.
- Forecasts are compared with a simple baseline using horizon-aware metrics and plots.
