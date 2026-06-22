---
name: skforecast
description: Use skforecast 0.22+ for time-series forecasting after forecasting-data-prep, including pandas Series/DataFrame inputs, ForecasterRecursive, ForecasterDirect, ForecasterRecursiveMultiSeries, ForecasterDirectMultiVariate, ForecasterRecursiveClassifier, ForecasterRnn, ForecasterStats, ForecasterFoundation, ForecasterEquivalentDate, scikit-learn-compatible estimators, ARIMA/SARIMAX/AutoARIMA/ETS/AutoETS/ARAR, exogenous variables, window features, backtesting, hyperparameter search, prediction intervals, conformal/quantile forecasts, plotting, explainability, drift detection, and anti-leakage validation.
---

# skforecast

Use this skill after `forecasting-data-prep`. skforecast is best when you want forecasting with scikit-learn-style regressors, tree/boosting models, statistical models, deep learning RNNs, or zero-shot foundation models while keeping a consistent API for fit, predict, backtesting, tuning, intervals, plotting, and deployment.

Use version-specific docs. The official README and docs identify 0.22.0 as stable; older class names such as `ForecasterAutoreg` are obsolete.

## Minimum Install

```bash
pip install skforecast
pip install skforecast[full]
```

Basic install requires Python 3.10+ and core dependencies. Use extras only when needed: `skforecast[stats]`, `skforecast[plotting]`, or `skforecast[deeplearning]`.

## Data Contract

- Start from a `forecasting-data-prep` contract: sorted time index, target, frequency/step size, train/validation/test cutoffs, horizon, known-future covariates, and leakage notes.
- Single-series forecasters use a pandas `Series` `y` with `DatetimeIndex` or `RangeIndex`; set/infer frequency before modeling.
- Exogenous variables use pandas `Series`/`DataFrame` `exog` aligned to `y`; future `exog` must be provided for every predicted step.
- Independent multi-series forecasting uses `ForecasterRecursiveMultiSeries` with a wide `DataFrame`, MultiIndex long `DataFrame`, or `dict[str, Series]`.
- Dependent multivariate forecasting uses `ForecasterDirectMultiVariate` when one target series is forecast using other series as predictors.
- Transformations, differentiation, encoders, scalers, feature selection, and window features belong inside train-only forecaster/backtesting workflows, not prefit on full data.

Read `references/skforecast-data-validation.md` before using multiple series, exogenous variables, probabilistic forecasts, backtesting, transforms, or deployment.

## Forecaster Selection

- `ForecasterRecursive`: single-series recursive multi-step forecasting with scikit-learn-compatible regressors.
- `ForecasterDirect`: single-series direct multi-step forecasting, one model per horizon step.
- `ForecasterRecursiveMultiSeries`: global independent multi-series forecasting.
- `ForecasterDirectMultiVariate`: dependent multivariate forecasting for one target level.
- `ForecasterRecursiveClassifier`: autoregressive classification.
- `ForecasterRnn`: Keras RNN/LSTM/GRU deep learning forecasting.
- `ForecasterStats`: statistical models through skforecast stats, statsmodels, sktime, or aeon-style estimators.
- `ForecasterFoundation`: zero-shot foundation models through `FoundationModel`.
- `ForecasterEquivalentDate`: equivalent-date baseline.

Read `references/skforecast-model-map.md` before claiming support for a forecaster, statistical estimator, foundation backend, deep learning architecture, or probabilistic method.

## Forecasting Pattern

```python
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster

forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(random_state=123, verbose=-1),
    lags=24,
)
forecaster.fit(y=y_train, exog=exog_train)

pred = forecaster.predict(steps=horizon, exog=exog_future)

cv = TimeSeriesFold(
    steps=horizon,
    initial_train_size=len(y_train),
    refit=True,
    fixed_train_size=False,
)
metric, backtest_pred = backtesting_forecaster(
    forecaster=forecaster,
    y=y_all,
    exog=exog_all,
    cv=cv,
    metric="mean_absolute_error",
)
```

For tuning, use `grid_search_forecaster`, `random_search_forecaster`, or `bayesian_search_forecaster` with `TimeSeriesFold` or `OneStepAheadFold`; use the multiseries/stat variants for those forecasters.

## Validation, Metrics, and Diagnostics

- Never random split. Use temporal holdouts and skforecast backtesting utilities.
- Use `TimeSeriesFold` for multi-step evaluation and `OneStepAheadFold` for one-step-ahead workflows.
- Recommended metrics: MAE/RMSE/MSE, MASE/RMSSE for scale-free comparison, WAPE/sMAPE for reporting, bias/mean error, pinball loss/coverage/CRPS for probabilistic forecasts. Avoid MAPE near zero.
- Use `plot_prediction_intervals`, skforecast plotting utilities, and pandas/matplotlib plots for target, predictions, and intervals.
- Use feature importance, SHAP where supported by the estimator, and `get_feature_importances()` where documented.
- Residual diagnostics are manual: use held-out/backtest residuals, not final test residuals after tuning.

## Probabilistic Forecasting

skforecast documents bootstrapped residual intervals, conformal prediction, conformal calibration, quantile regression, probabilistic global models, CRPS, and interval calibration. Use `predict_interval`, `predict_quantiles`, or forecaster-specific probabilistic methods only where supported.

Always validate empirical coverage and interval width on temporal validation/backtest windows.

## Anti-Leakage Rules

- Do not random split time rows or windows.
- Fit transformations, imputers, scalers, encoders, differentiation, feature selection, hyperparameter search, conformal calibration, and residual bootstrapping on train only or inside each backtest fold.
- Build lag/window/custom features only from past values; use skforecast `lags` and `RollingFeatures` or manual `shift` before rolling.
- Use future `exog` only when values are known for every forecast timestamp.
- Keep `steps`, `initial_train_size`, `gap`, `fixed_train_size`, `lags`, `window_features`, frequency, and cutoffs aligned with the data-prep contract.
- For multiple series, split and transform per series ID without leaking future target values across IDs.

## Common Errors

- Using deprecated class names from old docs.
- Passing a `DataFrame` where a single-series forecaster expects `Series`.
- Missing or inconsistent index frequency.
- Forgetting future `exog` for `predict`.
- Creating lag/rolling features from full data before splitting.
- Using `ForecasterRecursiveMultiSeries` for dependent multivariate forecasting instead of `ForecasterDirectMultiVariate`.
- Expecting all scikit-learn estimators to support quantile intervals or feature importances.
- Treating foundation models or RNNs as drop-in replacements without installing required extras/dependencies.

## References

- Read `references/skforecast-model-map.md` for forecasters, supported estimators/backends, and capability caveats.
- Read `references/skforecast-data-validation.md` for data formats, exogenous variables, validation, intervals, plotting, diagnostics, and deployment.
- Read `references/official-sources.md` for official sources consulted.

## Ready Checklist

- `forecasting-data-prep` contract is complete and leakage risks are documented.
- Index type/frequency, target, horizon, lags/window features, and cutoffs are explicit.
- Future exogenous values exist for every forecast step when `exog` is used.
- Forecaster choice matches single-series, independent multi-series, dependent multivariate, stats, deep learning, or foundation-model needs.
- Validation uses `TimeSeriesFold`/backtesting or temporal holdout, not random split.
- Metrics, plots, residuals, and interval coverage are computed on held-out periods only.
