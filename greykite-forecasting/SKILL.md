---
name: greykite-forecasting
description: Use LinkedIn Greykite for interpretable univariate forecasting after forecasting-data-prep, including pandas DataFrame inputs with time/value/regressor columns, Forecaster.run_forecast_config, ForecastConfig, MetadataParam, Silverkite, Prophet, Auto-ARIMA, lag-based and multistage templates, AUTO/SILVERKITE model templates, holidays/events, changepoints, regressors, lagged regressors, autoregression, prediction intervals via coverage, rolling time-series CV/backtest, benchmarking, plotting, component diagnostics, and anti-leakage safeguards.
---

# Greykite Forecasting

Use this skill after `forecasting-data-prep`. Greykite is best for interpretable univariate business forecasts with trend, seasonality, holidays/events, changepoints, autoregression, external regressors, prediction intervals, time-series CV, backtests, benchmarking, and component plots.

Do not treat Greykite as a joint multivariate or generic panel forecasting library. Its main `Forecaster` workflow forecasts one `value_col` at a time; loop over series IDs externally and use reconciliation only as documented post-processing.

## Minimum Install

```bash
pip install greykite
```

The GitHub release page shows 1.0.0 as latest release on January 18, 2024, while current `master` setup metadata declares 1.1.0. Pin and verify the installed version in production. Current setup metadata requires Python `>=3.10` and pins key dependencies such as pandas `<2.0.0` and scikit-learn `1.3.1`.

## Data Contract

- Start from a `forecasting-data-prep` contract: sorted timestamp, one target column, frequency, horizon, temporal cutoffs, known-future regressors, events, anomaly decisions, and leakage notes.
- Use a pandas `DataFrame`. Main columns are `time_col` and `value_col`, configured through `MetadataParam(time_col=..., value_col=..., freq=...)`.
- Provide `freq` explicitly when timestamps have missing timepoints; otherwise Greykite may infer it.
- Include future rows for the forecast horizon when using regressors; future regressor values must be known for prediction.
- Use `train_end_date` or `EvaluationPeriodParam` to enforce temporal holdouts, gaps, and CV windows.
- Use `anomaly_info` only for known anomalies; do not infer adjustments from validation/test target values.

Read `references/greykite-data-validation.md` before using regressors, autoregression, uncertainty intervals, cross-validation, benchmarking, or multiple series.

## Model Selection

Greykite supports these documented forecasting model families:

- `Silverkite`: flagship fast/interpretable model via `SILVERKITE`, `AUTO`, and many `SILVERKITE_*` templates.
- `Prophet`: `PROPHET` template via `ProphetEstimator`.
- `Auto-ARIMA`: `AUTO_ARIMA` template via `AutoArimaEstimator`.
- Lag-based: `LAG_BASED` template for simple lag/aggregated-lag baselines.
- Multistage: `SILVERKITE_TWO_STAGE`, `SILVERKITE_WOW`, and `MULTISTAGE_EMPTY`.
- Low-level Silverkite: `SK` template, `SimpleSilverkiteForecast`, and `SilverkiteForecast` for advanced users.
- Reconciliation: `ReconcileAdditiveForecasts` reconciles multiple forecasts post hoc; it is not a base forecasting model.

Start with `AUTO` unless the task requires a specific template. Read `references/greykite-model-map.md` before claiming model/template support.

## Forecasting Pattern

```python
from greykite.framework.templates.autogen.forecast_config import ForecastConfig, MetadataParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum

config = ForecastConfig(
    model_template=ModelTemplateEnum.AUTO.name,
    forecast_horizon=24,
    coverage=0.95,
    metadata_param=MetadataParam(time_col="ts", value_col="y", freq="H"),
)

result = Forecaster().run_forecast_config(df=train_and_future_df, config=config)
forecast_df = result.forecast.df
backtest_df = result.backtest.df
```

For regressors, include columns in `df` and configure:

```python
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam

model_components = ModelComponentsParam(
    regressors={"regressor_cols": ["promo", "planned_price"]},
)
```

For low-level `SK`, use `custom={"extra_pred_cols": [...]}` instead of `regressors.regressor_cols`.

## Validation, Metrics, and Plotting

- Use `EvaluationPeriodParam` for temporal backtest and rolling CV: `test_horizon`, `periods_between_train_test`, `cv_horizon`, `cv_min_train_periods`, `cv_periods_between_splits`, `cv_periods_between_train_test`, and `cv_max_splits`.
- Use `result.grid_search` and `summarize_grid_search_results(...)` for CV results. Ignore raw sklearn rank columns for metrics where lower is better; use the Greykite summary helper.
- Use `result.backtest` for holdout test metrics and plots. Use `result.forecast` for the final fitted forecast.
- Built-in evaluation uses `EvaluationMetricEnum`; common choices include MAPE, RMSE, MAE-like metrics, correlation, quantile loss, coverage, and interval metrics depending on configuration.
- Plot with `result.backtest.plot()`, `result.forecast.plot()`, `plot_components()`, and `plot_grouping_evaluation(...)`.
- For residual diagnostics, use `backtest.df` or CV outputs; component plots can visualize residuals and changepoints. Do not tune on final test residuals.

## Intervals and Quantiles

Set `ForecastConfig.coverage` to request prediction bands. Silverkite and Prophet support prediction intervals. Silverkite uncertainty uses residual-based methods such as `simple_conditional_residuals`; Prophet uses its own uncertainty configuration. Silverkite also supports quantile loss as a fitting objective.

Validate empirical coverage and interval width on temporal backtests before relying on intervals.

## Anti-Leakage Rules

- Never random split time-series rows. Use Greykite's rolling CV/backtest parameters or explicit time cutoffs.
- Fit anomaly adjustments, preprocessing, regressors, autoregression choices, changepoints, grid search, and model selection on train/CV folds only.
- Use `periods_between_train_test` or `cv_periods_between_train_test` when a gap is needed between training and prediction.
- Use future regressors only if they are known for every forecast timestamp. Lagged regressors must be built from past values only.
- Keep `forecast_horizon`, `freq`, `train_end_date`, `test_horizon`, CV horizons, and gaps aligned with the data-prep contract.
- For multiple IDs, loop by series and create splits/features independently per ID; do not let aggregate future target information leak into per-series features.

## Common Errors

- Passing wide/panel data to `Forecaster` as if it supported `series_id`.
- Omitting `freq` when timestamps have missing timepoints.
- Including future regressor columns without known future values.
- Using `SK` low-level template but configuring regressors with the high-level Silverkite syntax.
- Using autoregressive lags smaller than `forecast_horizon` without understanding simulation and interval implications.
- Treating `forecast.train_evaluation` as honest validation; use `backtest` or CV.
- Trusting raw sklearn `rank_test_*` columns instead of `summarize_grid_search_results`.

## References

- Read `references/greykite-model-map.md` for templates, estimators, and capability caveats.
- Read `references/greykite-data-validation.md` for data format, regressors, CV/backtesting, intervals, plotting, diagnostics, and multiple series.
- Read `references/official-sources.md` for official sources consulted.

## Ready Checklist

- `forecasting-data-prep` contract is complete and leakage risks are resolved or documented.
- Data is one-target pandas format with explicit `time_col`, `value_col`, and `freq`.
- Future regressor/event rows are available and known at prediction time.
- Template choice supports required regressors, autoregression, intervals, and horizon.
- Validation uses temporal holdout/CV with any required gap, not random split.
- Metrics, plots, residuals, and interval coverage are computed on held-out periods only.
