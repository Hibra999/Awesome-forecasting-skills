# FEDOT Model And Operation Map

Use this file before selecting or claiming FEDOT forecasting support. Names below come from FEDOT 0.7.5 docs and official repository JSON/source consulted on 2026-06-22.

## Forecasting Model Operations

Official operations applicable to `TaskTypesEnum.ts_forecasting` from `model_repository.json`:

Native time-series operations:
- `ar`: auto regression.
- `arima`: ARIMA.
- `cgru`: convolutional gated recurrent unit model.
- `clstm`: convolutional long short-term memory model.
- `ets`: exponential smoothing.
- `glm`: generalized linear model.
- `locf`: last observation carried forward.
- `polyfit`: polynomial interpolation.
- `stl_arima`: STL + ARIMA.
- `ts_naive_average`: simple forecast using the known-part mean.

Regression-on-lagged-table operations:
- `adareg`
- `dtreg`
- `gbr`
- `lasso`
- `linear`
- `rfr`
- `ridge`
- `sgdr`
- `svr`
- `treg`
- `custom`

The official docs explain that FEDOT uses lagged transformation to convert a time series to tabular features so regression methods can forecast. Use regression operations only after a time-series-to-table operation such as `lagged` or `sparse_lagged`.

## Time-Series Data Operations

Core TS operations from docs/examples/repository:
- `lagged`: windowing method to represent a series as a trajectory matrix/table.
- `sparse_lagged`: sparse lagged transformation.
- `smoothing`: rolling mean smoothing.
- `gaussian_filter`: Gaussian smoothing.
- `diff_filter`: differential filter.
- `cut`: cut part of a dataset.
- `exog_ts`: exogenous time-series branch for multimodal workflows.
- `data_source_ts`: source node for multimodal time-series pipelines.

General table preprocessing operations can appear after conversion to table, but do not apply raw-table preprocessing to the full series before temporal splitting.

## Example Pipeline Patterns

Documented example/manual patterns include:
- `lagged -> ridge`
- `glm` branch joined with `lagged -> ridge`, then final `ridge`
- `polyfit`
- `polyfit -> ridge`
- `smoothing -> lagged -> ridge`
- `lagged -> dtreg` or tree/regression branches joined with `rfr`
- `ets`, multiple `ets`, `ar`, `arima`, `stl_arima`
- `locf` with lagged features
- `ts_naive_average` with lagged features
- `cgru` and `clstm` DNN pipeline examples
- `lagged` + `exog_ts` branches joined by `ridge` for exogenous TS examples

Use `PipelineBuilder` for manual pipelines, for example:

```python
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

pipeline = PipelineBuilder().add_sequence("lagged", "ridge").build()
```

## Presets And Available Operations

Use `preset="ts"` for forecasting-oriented AutoML search. To constrain search, pass `available_operations=[...]` with official operation IDs. Avoid including operations that are not in FEDOT's official operation repositories for the installed version.

The exogenous official example constrains operations with names including `lagged`, `ridge`, `exog_ts`, `arima`, `knnreg`, `rfr`, and `svr`; however, `knnreg` was not returned by a repository-task filter against `TaskTypesEnum.ts_forecasting` in the consulted source. Prefer operation IDs in the official task-filtered list unless reproducing that exact example and verifying local FEDOT behavior.

## Metrics

`TimeSeriesForecastingMetricsEnum` values from official source:
- `mase`
- `rmse`
- `mse`
- `neg_mean_squared_log_error`
- `mape`
- `smape`
- `mae`
- `r2`
- `rmse_pen`

Use `model.get_metrics(metric_names=[...], target=...)` after `predict` or `forecast`. Prefer MAE/RMSE/SMAPE/MASE for forecasting reports; avoid MAPE around zero actuals.

## Probabilistic Forecasting

The browsed official FEDOT 0.7.5 forecasting docs and API cover point forecasts. `predict_proba` is documented for classification, not time-series forecasting. Do not claim native prediction intervals, quantile forecasts, conformal forecasts, CRPS, or probabilistic TS forecasts unless a specific official source for the installed version documents them.

## Documented Limitations And Caveats

- `forecast_length` is mandatory through `TsForecastingParams`.
- `Fedot.forecast(..., horizon=...)` iterates when `horizon > forecast_length`; official API says this extended out-of-sample mode is not supported for multimodal data.
- FEDOT multi-time-series docs forecast one target using several time series of the same variable; this is not the same as fitting a global panel model for many unrelated IDs.
- DNN operations `cgru` and `clstm` may require optional dependencies and more data/time.
- FEDOT is AutoML; set `timeout`, `n_jobs`, `metric`, `available_operations`, and seeds for reproducibility and bounded runtime.
