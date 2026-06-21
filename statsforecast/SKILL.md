---
name: statsforecast
description: Use Nixtla StatsForecast for fast statistical and econometric forecasting after forecasting-data-prep, including long-format pandas/polars data with unique_id/ds/y, local models for many series, AutoARIMA/AutoETS/AutoCES/AutoTheta/AutoMFLES/AutoTBATS, ARIMA, AutoRegressive, Theta, MSTL, MFLES, TBATS, GARCH/ARCH, exponential smoothing, naive/intermittent baselines, exogenous regressors, static covariates, conformal or native prediction intervals, cross_validation, distributed Dask/Ray/Spark workflows, plotting, fitted values, and leakage-safe temporal validation.
---

# StatsForecast

Use this skill after `forecasting-data-prep`. StatsForecast is best for fast local statistical forecasting across one or many univariate series, production baselines, automatic ARIMA/ETS/CES/Theta model selection, intermittent demand, multiple seasonality, prediction intervals, exogenous regressors, and distributed execution.

Do not treat StatsForecast as a multivariate endogenous VAR-style library. It models each `unique_id` series locally; multiple series are panels for parallel local modeling, not one joint multivariate model.

## Minimum Install

```bash
pip install statsforecast
conda install -c conda-forge statsforecast
```

Pin versions for production. Use extras only when needed: `statsforecast[polars]`, `statsforecast[plotly]`, `statsforecast[dask]`, `statsforecast[ray]`, or `statsforecast[spark]`.

## Data Contract

- Start from a `forecasting-data-prep` contract: sorted timestamps, unique series keys, frequency, horizon, temporal cutoffs, covariate availability, and leakage notes.
- Use long format with `unique_id`, `ds`, and `y` unless passing custom names via `id_col`, `time_col`, and `target_col`.
- `unique_id` identifies each series; `ds` is a datestamp or integer time index; `y` is numeric.
- Use pandas by default; official docs also show Polars with the `polars` extra. Distributed docs cover Dask, Ray, and Spark.
- Extra columns after `y` are external regressors for models that support them. Future regressor values must be passed in `X_df` for the full horizon.
- Static covariates are documented as supported, but model-specific behavior must be verified in the relevant guide/source.
- Use `freq` as a pandas-style frequency string for datetimes or an integer such as `1` for integer datestamps.
- Set each model's `season_length` to the true seasonal period; wrong season length is a common source of bad forecasts.

Read `references/statsforecast-data-validation.md` before using exogenous variables, panels, Polars/distributed backends, intervals, or custom column names.

## Model Selection

Use model names exactly from `statsforecast.models`. Main documented model families:

- Automatic: `AutoARIMA`, `AutoETS`, `AutoCES`, `AutoTheta`, `AutoMFLES`, `AutoTBATS`.
- ARIMA family: `ARIMA`, `AutoRegressive`.
- Theta: `Theta`, `OptimizedTheta`, `DynamicTheta`, `DynamicOptimizedTheta`.
- Multiple seasonality: `MSTL`, `MFLES`, `TBATS`.
- GARCH/ARCH: `GARCH`, `ARCH`.
- Baselines: `HistoricAverage`, `Naive`, `RandomWalkWithDrift`, `SeasonalNaive`, `WindowAverage`, `SeasonalWindowAverage`.
- Exponential smoothing: `SimpleExponentialSmoothing`, `SimpleExponentialSmoothingOptimized`, `SeasonalExponentialSmoothing`, `SeasonalExponentialSmoothingOptimized`, `Holt`, `HoltWinters`.
- Intermittent/sparse: `ADIDA`, `CrostonClassic`, `CrostonOptimized`, `CrostonSBA`, `IMAPA`, `TSB`.
- Source-exported utilities/models: `SklearnModel`, `ConstantModel`, `ZeroModel`, `NaNModel`, `UCM`. `UCM` was exported in the official source but not listed in the browsed model reference index; verify local docs/source before using.

Read `references/statsforecast-model-map.md` before claiming interval, exogenous, fitted-value, or source-exported support.

## Forecasting Pattern

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive

sf = StatsForecast(
    models=[
        AutoARIMA(season_length=12),
        SeasonalNaive(season_length=12),
    ],
    freq="MS",
    n_jobs=-1,
    fallback_model=SeasonalNaive(season_length=12),
)

forecasts = sf.forecast(df=train_df, h=12, level=[80, 95])
```

Use `forecast(df=..., h=...)` for memory-efficient production forecasts that do not store fitted model objects. Use `fit(df)` then `predict(h)` when you need stored models, in-sample fitted values, save/load, or repeated predictions from the fitted state.

For exogenous regressors:

```python
fcst = sf.forecast(df=train_df, h=horizon, X_df=future_x_df, level=[95])
```

`future_x_df` must contain `unique_id`, `ds`, and the same future regressor columns for every horizon row.

## Validation, Metrics, and Plotting

- Validate with `cross_validation(df=..., h=..., n_windows=..., step_size=...)`; `n_windows=1` mirrors a temporal train/test split.
- Cross-validation output contains `unique_id`, `ds`, `cutoff`, `y`, and one column per model.
- Use `utilsforecast.evaluation.evaluate` with losses from `utilsforecast.losses`, as in official docs.
- Recommended metrics: MAE/RMSE/MSE for scale-dependent error, MASE or RMSSE for cross-series comparison, sMAPE/WAPE for reporting, and pinball/coverage/interval width for intervals. Avoid MAPE when actuals can be zero or very small.
- Plot with `StatsForecast.plot(df, forecasts_df, level=[...], engine="matplotlib"|"plotly")`.
- Use `fitted=True`, `predict_in_sample`, or stored fitted models when you need fitted values. Residual diagnostics are manual: compute residuals on validation/test or cross-validation results, not on final test after tuning.

## Probabilistic Forecasting

Pass `level=[80, 95]` to request prediction intervals. Some models have native intervals; `ConformalIntervals(h=..., n_windows=...)` provides distribution-free conformal intervals for models, with `n_windows * h` less than series length and `n_windows >= 2` recommended by docs.

Evaluate interval coverage and width on temporal validation windows before using intervals operationally.

## Anti-Leakage Rules

- Never random split time-series rows. Split by cutoff and keep training windows contiguous.
- Fit transformations, calendar encoders, scaling, decomposition features, model selection, and conformal calibration on train only or inside each cross-validation fold.
- Build lag/rolling features only from past data; use `groupby("unique_id").shift(...)` before rolling.
- Use `X_df` only for covariates that are known for every future `ds` at prediction time.
- Keep `h`, `freq`, `season_length`, `step_size`, `n_windows`, and panel cutoffs aligned with the data-prep contract.
- Recreate features and refit models per cutoff during backtesting; do not tune on final test residuals.

## Common Errors

- Using wide data instead of long `unique_id`/`ds`/`y` format.
- Forgetting `X_df` when training data includes future-known exogenous regressors.
- Passing observed-only future variables as exogenous regressors.
- Treating panels as one multivariate model instead of many local univariate models.
- Choosing `season_length` by habit instead of the actual data frequency.
- Using `forecast()` when you need stored fitted models; use `fit()`/`predict()` instead.
- Expecting distributed backends to support every stateful method exactly like local pandas.

## References

- Read `references/statsforecast-model-map.md` for supported models and capability notes.
- Read `references/statsforecast-data-validation.md` for formats, exogenous variables, validation, intervals, plotting, fitted values, and diagnostics.
- Read `references/official-sources.md` for official sources consulted.

## Ready Checklist

- `forecasting-data-prep` contract is complete and leakage risks are resolved or documented.
- Data is long format, sorted, unique by `(unique_id, ds)`, and has valid `freq`.
- `h`, temporal cutoffs, and `season_length` match the business horizon and data frequency.
- Exogenous regressors are known future variables and `X_df` covers every horizon row.
- Selected models officially support the needed intervals, exogenous variables, fitted values, or intermittent/multiple-seasonal behavior.
- Validation uses `cross_validation` or temporal holdouts, not random splits.
- Metrics, plots, residuals, and interval coverage are computed on held-out periods only.
