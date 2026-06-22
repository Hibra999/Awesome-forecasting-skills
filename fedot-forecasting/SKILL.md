---
name: fedot-forecasting
description: Use FEDOT 0.7.5 for AutoML time-series forecasting after forecasting-data-prep, including Fedot(problem='ts_forecasting'), TaskTypesEnum.ts_forecasting, TsForecastingParams, InputData time-series loaders, univariate forecasting, multi-time-series and multimodal exogenous workflows, lagged/sparse_lagged/exog_ts transformations, native TS and regression operations, forecast_length/horizon handling, temporal train_test_data_setup, cv_folds validation, metrics, plotting, residual checks, and anti-leakage safeguards.
---

# FEDOT Forecasting

Use this skill after `forecasting-data-prep`. FEDOT is best when the user wants AutoML or manually composed graph pipelines for time-series forecasting, including evolutionary pipeline design, lagged feature extraction, regression-on-lags models, native time-series operations, multimodal exogenous series, pipeline tuning, and pipeline export.

Do not treat FEDOT as a generic probabilistic forecasting or panel-forecasting library. The browsed official 0.7.5 docs cover point forecasts; prediction intervals/probabilistic TS forecasts are not documented.

## Minimum Install

```bash
pip install fedot
pip install "fedot[extra]"
```

Use extras only when optional image/text/DNN dependencies are needed. Pin `fedot==0.7.5` if reproducing the consulted stable docs.

## Data Contract

- Start from a `forecasting-data-prep` contract: ordered time index, target, frequency/step, forecast horizon, temporal cutoffs, lag/window plan, exogenous availability, and leakage notes.
- Define `Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=h))`; `forecast_length` is required.
- Use `InputData.from_csv_time_series(...)` for one target series from CSV, or pass arrays/DataFrames/InputData to `Fedot.fit`.
- Use `InputData.from_csv_multi_time_series(...)` when official multi-time-series format applies: several columns of the same variable and one `target_column`.
- Use dict or `MultiModalData` for multimodal/exogenous workflows; keep target history and exogenous series aligned by the same cutoffs.
- FEDOT forecasts one target series per TS task. For many unrelated IDs, run separate workflows unless an official grouped/panel workflow is provided.

Read `references/fedot-data-validation.md` before using multi-time-series, multimodal exogenous data, iterative horizons, or manual pipelines.

## Model And Operation Selection

Use official operation IDs exactly. Time-series-specific model operations include `ar`, `arima`, `cgru`, `clstm`, `ets`, `glm`, `locf`, `polyfit`, `stl_arima`, and `ts_naive_average`.

Regression operations officially applicable to `ts_forecasting` after `lagged`/`sparse_lagged` conversion include `adareg`, `dtreg`, `gbr`, `lasso`, `linear`, `rfr`, `ridge`, `sgdr`, `svr`, `treg`, and `custom`.

Time-series data operations include `lagged`, `sparse_lagged`, `smoothing`, `gaussian_filter`, `diff_filter`, `cut`, `exog_ts`, and `data_source_ts`. Read `references/fedot-model-map.md` before claiming support for a model, operation, preset, DNN model, or exogenous pipeline.

## Forecasting Pattern

```python
from fedot import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

forecast_length = 24
task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=forecast_length))

data = InputData.from_csv_time_series(
    file_path="series.csv",
    task=task,
    target_column="target",
)
train_data, test_data = train_test_data_setup(data)

model = Fedot(
    problem="ts_forecasting",
    task_params=task.task_params,
    timeout=10,
    cv_folds=3,
    preset="ts",
    metric=["rmse", "mae"],
)
pipeline = model.fit(train_data)
forecast = model.forecast(test_data)
metrics = model.get_metrics(metric_names=["rmse", "mae", "mape"], target=test_data.target)
model.plot_prediction()
```

For a manual pipeline, pass `predefined_model=` to `fit`, usually built with `PipelineBuilder`, such as `PipelineBuilder().add_sequence("lagged", "ridge").build()`.

## Validation, Metrics, And Diagnostics

- Never random split. Use `train_test_data_setup(data)`; for TS it uses `forecast_length` and ignores `shuffle`, `stratify`, and random split behavior.
- Use `validation_blocks` for in-sample validation windows and `cv_folds` for FEDOT TS validation during model design.
- Use `Fedot.predict(..., in_sample=True, validation_blocks=...)` for in-sample validation and `Fedot.forecast(..., horizon=...)` for out-of-sample future forecasts.
- Recommended metrics: `rmse`, `mae`, `mape`, `smape`, `mase`, `mse`, `r2`, and `rmse_pen`; avoid MAPE when actuals can be zero or near zero.
- Plot with `pipeline.show()`, `pipeline.print_structure()`, and `model.plot_prediction()`.
- FEDOT docs do not document a dedicated residual diagnostic API. Compute residuals manually from held-out/in-sample forecasts and inspect bias, autocorrelation, and segment/exogenous failures outside final test tuning.

## Forecast Horizon

`forecast_length` is the model design depth. `Fedot.forecast(pre_history, horizon=...)` can forecast a custom horizon: if `horizon > forecast_length`, FEDOT iterates using previous forecasts; if `horizon < forecast_length`, output is cut to the requested horizon. Official API says out-of-sample horizon greater than fitted forecast length is not supported for multimodal data.

## Exogenous And Multivariate Use

- Multi-time-series: use `InputData.from_csv_multi_time_series(...)` for several variants of the same variable and one target column.
- Multimodal exogenous: build aligned `InputData` objects and combine them with `MultiModalData({"lagged": target_data, "exog_ts": exog_data})`, then use a pipeline with `lagged` and `exog_ts` branches joined by a regression model such as `ridge`.
- Future exogenous values must be known at prediction time for the full validation or forecast window. Do not use observed-only future covariates.

## Anti-Leakage Rules

- Do not random split rows; split by time using `train_test_data_setup` or explicit cutoffs.
- Fit lagged transforms, smoothing, filters, decomposition, scaling, imputation, feature selection, hyperparameter tuning, and AutoML composition only on train folds/windows.
- Build lag/window features only from past values; never construct full-series lag tables before splitting.
- Use exogenous variables only when they are available for each forecast timestamp at prediction time.
- Keep `forecast_length`, `horizon`, `validation_blocks`, `cv_folds`, and pre-history windows aligned with the data-prep contract.
- Tune on validation/in-sample folds, then reserve a final temporal test window for one final estimate.

## Common Errors

- Forgetting `TsForecastingParams(forecast_length=...)`.
- Calling `predict` for future forecasting instead of `forecast`, or ignoring `in_sample=True` semantics.
- Passing multi-ID panel data as if FEDOT automatically fits one global grouped model.
- Using `horizon > forecast_length` with multimodal data, which the official API says is unsupported for out-of-sample forecast.
- Including future target-derived features or unavailable exogenous values in `exog_ts`.
- Expecting built-in prediction intervals; they were not documented in the official 0.7.5 forecasting docs.
- Claiming operation IDs not present in FEDOT's official repositories.

## References

- Read `references/fedot-model-map.md` for official model/data operation IDs and caveats.
- Read `references/fedot-data-validation.md` for data formats, exogenous workflows, validation, horizons, metrics, plotting, and diagnostics.
- Read `references/official-sources.md` for official sources consulted.

## Ready Checklist

- `forecasting-data-prep` contract is complete and leakage risks are documented.
- `forecast_length`, horizon, validation blocks, and temporal cutoffs are explicit.
- Input is `InputData`, supported CSV loader output, array/DataFrame, dict, or `MultiModalData` with aligned target/exog history.
- Operation IDs and presets are official for FEDOT 0.7.5.
- Validation uses temporal holdout/in-sample folds, not random split.
- Metrics, plots, residuals, and final test evaluation are computed on held-out time windows only.
