# tsflex API Map

Use exact documented names.

## Core Feature Extraction APIs

| API | Purpose | Notes |
| --- | --- | --- |
| `tsflex.features.FeatureCollection` | Registry for feature descriptors and feature calculation. | Main extraction object. |
| `FeatureCollection.calculate(data, ...)` | Calculate features on pandas data. | Returns DataFrame or list of DataFrames. |
| `FeatureCollection.add(features)` | Add descriptors after initialization. | Accepts descriptors/collections as documented. |
| `FeatureCollection.reduce(feature_names)` | Reduce a collection after selecting feature columns. | Useful for faster inference after train-only feature selection. |
| `FeatureCollection.serialize(file_path)` | Serialize a feature collection. | Uses documented serialization support. |
| `tsflex.features.FeatureDescriptor(function, series_name, window=None, stride=None)` | Define one feature. | `series_name` can be a string or tuple for multi-series functions. |
| `tsflex.features.MultipleFeatureDescriptors` | Expand feature/function/series/window/stride combinations. | Use for repetitive descriptor grids. |
| `tsflex.features.FuncWrapper` | Configure callable features. | Use for output names, kwargs, or `input_type=pd.Series`. |

## `calculate()` Parameters to Know

- `data`: Series, DataFrame, DataFrameGroupBy, or list of Series/DataFrames.
- `stride`: override descriptor stride when passed.
- `segment_start_idxs`, `segment_end_idxs`, `window_idx`: manual segment controls.
- `group_by_all`, `group_by_consecutive`: grouping modes documented in API.
- `bound_method`: `"inner"`, `"inner-outer"`, or `"outer"` for multi-series bounds.
- `approve_sparsity`: acknowledge irregular/sparse windows.
- `show_progress`: progress bar.
- `logging_file_path`: execution-time log path.
- `n_jobs`: process count; `0` or `1` runs sequentially.

## Feature Function Patterns

Feature callables follow:

```python
function(*series, **kwargs) -> value_or_list
```

Supported patterns:

- one-to-one: one input series, one output.
- one-to-many: one input series, multiple outputs; wrap with `FuncWrapper(output_names=[...])`.
- many-to-one: multiple input series, one output; pass `series_name=("a", "b")`.
- many-to-many: multiple input series, multiple outputs; use `FuncWrapper`.

Use `input_type=pd.Series` when a feature function needs the sequence index.

## Official Integration Wrappers

The `tsflex.features.integrations` module documents wrappers for:

- `seglearn_wrapper`
- `seglearn_feature_dict_wrapper`
- `tsfel_feature_dict_wrapper`
- `tsfresh_combiner_wrapper`
- `tsfresh_settings_wrapper`
- `catch22_wrapper`

These wrappers depend on the corresponding external packages. Do not claim those packages are installed unless project dependencies include them.

## Processing and Chunking APIs

- `tsflex.processing.SeriesProcessor`: define processing operations on one or more series.
- `tsflex.processing.SeriesPipeline`: apply processors sequentially and serialize pipelines.
- `tsflex.chunking.chunk_data()`: split data into continuous chunks based on gaps, min/max chunk duration, and optional overlap.

Processing can leak if it learns global parameters. Fit or define train-only processing before applying to validation/test.

## Logging and Diagnostics

- Pass `logging_file_path` to `.calculate()`.
- Use `get_feature_logs(logging_file_path)`, `get_function_stats(logging_file_path)`, or `get_series_names_stats(logging_file_path)` to inspect execution bottlenecks.

## Documented Gaps

- tsflex is not a classifier, regressor, forecaster, clustering model, or supervised selector.
- README lists sklearn integration as future work, not stable core API.
- README lists time-series segmentation support as future work, while lower-level strided rolling is used internally.
- Multi-indexed DataFrames are not supported.
- Multiprocessing is documented as unsupported on Windows.
- tsflex does not document a dedicated plotting API.
