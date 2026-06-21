# StatsForecast Model Map

Use this reference before choosing models or promising a capability. StatsForecast models are local statistical forecasters applied independently to each `unique_id`.

## Models Listed in Official README and Docs

Automatic forecasting:

- `AutoARIMA`
- `AutoETS`
- `AutoCES`
- `AutoTheta`
- `AutoMFLES`
- `AutoTBATS`

ARIMA family:

- `ARIMA`
- `AutoRegressive`

Theta family:

- `Theta`
- `OptimizedTheta`
- `DynamicTheta`
- `DynamicOptimizedTheta`

Multiple seasonalities:

- `MSTL`
- `MFLES`
- `TBATS`

GARCH and ARCH:

- `GARCH`
- `ARCH`

Baseline models:

- `HistoricAverage`
- `Naive`
- `RandomWalkWithDrift`
- `SeasonalNaive`
- `WindowAverage`
- `SeasonalWindowAverage`

Exponential smoothing:

- `SimpleExponentialSmoothing`
- `SimpleExponentialSmoothingOptimized`
- `SeasonalExponentialSmoothing`
- `SeasonalExponentialSmoothingOptimized`
- `Holt`
- `HoltWinters`

Sparse or intermittent demand:

- `ADIDA`
- `CrostonClassic`
- `CrostonOptimized`
- `CrostonSBA`
- `IMAPA`
- `TSB`

## Source-Exported Names to Verify

The official `python/statsforecast/models.py` `__all__` export also includes:

- `SklearnModel`
- `ConstantModel`
- `ZeroModel`
- `NaNModel`
- `UCM`

Use these only after checking the installed source/API because not all appeared in the browsed high-level model reference index.

## Capability Notes

- Exogenous variables are documented for `AutoARIMA`, `ARIMA`, `AutoRegressive`, `Theta`, `AutoMFLES`, and `MFLES`; `MSTL` can use exogenous variables if the trend forecaster supports them. Verify `uses_exog` or the model reference before passing `X_df`.
- Prediction intervals are requested with `level=[...]`. Native interval availability is model-specific; conformal prediction can add distribution-free intervals through `ConformalIntervals`.
- In-sample fitted values and fitted-value intervals are model-specific in the docs table. Use `fitted=True`, `predict_in_sample`, or `fit()`/`predict()` paths only when the selected model supports them.
- `GARCH` and `ARCH` are for volatility-style forecasting, not ordinary mean demand forecasts.
- Intermittent models are for sparse non-zero demand. Do not use them as generic seasonal models.
- `AutoTBATS`, `MSTL`, `MFLES`, and `TBATS` are useful when multiple seasonalities matter.

## Selection Heuristics

- Always include a simple baseline such as `Naive`, `SeasonalNaive`, or `HistoricAverage`.
- Start with `AutoARIMA`, `AutoETS`, `AutoTheta`, and a seasonal naive baseline for broad statistical benchmarking.
- Use `AutoARIMA` or `ARIMA` when exogenous regressors are central.
- Use `MSTL`, `MFLES`, `TBATS`, or `AutoTBATS` for multiple seasonalities.
- Use `ADIDA`, `CrostonClassic`, `CrostonOptimized`, `CrostonSBA`, `IMAPA`, or `TSB` for intermittent/sparse demand.
- Use `fallback_model` for large panels so one model failure does not break the entire run.
