# FEDOT Data, Validation, And Leakage Notes

Use this file when converting prepared time-series data to FEDOT, validating temporal folds, adding exogenous series, plotting, or diagnosing errors.

## Accepted Inputs

High-level `Fedot.fit` accepts feature values in supported formats documented by the API:
- file path string or path-like object
- `numpy.ndarray`
- pandas `DataFrame`
- `InputData`
- `MultiModalData`
- `dict`
- `tuple`

For time series, prefer explicit FEDOT structures:

```python
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=horizon))
data = InputData(idx=idx, features=series, target=series, task=task, data_type=DataTypesEnum.ts)
```

CSV helpers:
- `InputData.from_csv_time_series(file_path=..., task=..., target_column=...)`
- `InputData.from_csv_multi_time_series(file_path=..., task=..., target_column=..., columns_to_use=...)`

`InputData.from_csv_time_series` forms `DataTypesEnum.ts`; `from_csv_multi_time_series` forms `DataTypesEnum.multi_ts`.

## Univariate, Multivariate, Multiple Series

- Univariate: one numeric target series with `DataTypesEnum.ts`; for CSV use `from_csv_time_series`.
- Multi-time-series: official docs describe using several columns of the same variable to forecast one target column. Use `from_csv_multi_time_series`.
- Multimodal/exogenous: use `MultiModalData` or a dict of named time series. The docs show keys such as `ws`/`ssh`, or operation-oriented keys such as `lagged` and `exog_ts`.
- Multiple unrelated series/panel: not documented as a first-class grouped forecasting mode. Run one workflow per series ID or require an official grouped/panel source before claiming support.

## Exogenous Variables

Official exogenous example pattern:

```python
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

train_dataset = MultiModalData({
    "lagged": train_lagged,
    "exog_ts": train_exog,
})
predict_dataset = MultiModalData({
    "lagged": predict_lagged,
    "exog_ts": predict_exog,
})
pipeline = PipelineBuilder().add_node("lagged", 0).add_node("exog_ts", 1).join_branches("ridge").build()
```

Rules:
- Split target and exogenous series with the same `Task`, `forecast_length`, and `validation_blocks`.
- Future exogenous values must be known for the whole validation or forecast period.
- Do not pass exogenous values derived from future target observations.
- Use `plot_prediction(target=...)` with the relevant multimodal target name.

## Temporal Splitting And Validation

Use `train_test_data_setup(data)`.

Documented behavior:
- For time series, `split_ratio`, `shuffle`, and `stratify` are ignored.
- The method uses `forecast_length` from `data.task`.
- Default split holds out the last `forecast_length` target values.
- `validation_blocks=N` holds out `forecast_length * N` values for in-sample validation.

Example:

```python
train_data, test_data = train_test_data_setup(data)
train_data, validation_data = train_test_data_setup(data, validation_blocks=3)
```

FEDOT TS validation:
- Set `cv_folds` in `Fedot(...)` to use time-series validation during model design.
- Use `Fedot.predict(..., in_sample=True, validation_blocks=N)` for in-sample forecasts.
- Use `Fedot.forecast(pre_history, horizon=...)` for future out-of-sample forecasts.

Do not use random train/test split utilities from pandas, numpy, or sklearn for forecasting rows.

## Horizon Semantics

- `forecast_length` in `TsForecastingParams` is the required training/design depth.
- `Fedot.forecast(pre_history=None, horizon=None)` uses `forecast_length` by default.
- If `horizon > forecast_length`, FEDOT iteratively forecasts additional blocks using previously forecasted values.
- If `horizon < forecast_length`, FEDOT cuts the forecast to the requested horizon.
- Official API says extended out-of-sample forecast with horizon greater than fitted forecast length is not supported for multimodal data.

## Metrics And Plotting

Use:
- `model.get_metrics(metric_names=["rmse", "mae", "mape"], target=test_data.target)`
- `pipeline.print_structure()`
- `pipeline.show()`
- `model.plot_prediction()`

Recommended metrics:
- `mae`, `rmse`, `mse` for scale-dependent accuracy.
- `smape` or `mase` for reporting across scales.
- `mape` only when actuals are not zero/near-zero.
- `r2` cautiously; it is often less interpretable for nonstationary series.

## Residuals And Diagnostics

FEDOT docs do not document a dedicated residual diagnostics API for forecasting. Use manual diagnostics:
- residuals = held-out target - forecast
- plot residuals over time
- inspect residual bias and large errors by horizon step
- check residual autocorrelation outside FEDOT if needed
- compare in-sample validation residuals with final holdout residuals

Never tune final models on final test residuals.

## Anti-Leakage Checklist

- Create temporal cutoffs before any target-aware feature engineering.
- Use `train_test_data_setup` or explicit time slicing.
- Fit AutoML composition, tuning, `lagged`, `sparse_lagged`, filters, smoothing, decomposition, scaling, and regressors only on training windows/folds.
- Keep target lags causal; no full-series rolling/lags before split.
- Ensure exogenous values are known future values for every forecast timestamp.
- Keep `forecast_length`, `horizon`, `validation_blocks`, and `cv_folds` consistent.
- Reserve an untouched final test window after selecting operations and hyperparameters.
