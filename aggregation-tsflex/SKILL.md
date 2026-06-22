---
name: aggregation-tsflex
description: Use tsflex for time-series aggregation and feature extraction after validating data, including pandas Series/DataFrame/list inputs, wide or series-list layouts, asynchronous multivariate signals, irregular sampling, strided rolling windows, FeatureCollection/FeatureDescriptor/MultipleFeatureDescriptors, FuncWrapper custom features, external feature integrations, chunking gaps, execution logging, serialization, and leakage-safe tabular ML feature generation.
---

# tsflex Aggregation

Use this skill after the time-series data-prep step. tsflex is best when feature extraction needs flexible windows/strides, irregular or asynchronous modalities, custom feature functions, or integrations with numpy/scipy/tsfresh/TSFEL/seglearn/catch22.

Do not treat tsflex as a model. It extracts or processes time-series data; downstream scaling, feature selection, modeling, metrics, and validation are separate steps.

## Minimum Install

```bash
python -m pip install tsflex
```

PyPI identifies `tsflex 0.4.1` as the latest release, published September 6, 2024. The official README also documents `conda install -c conda-forge tsflex`.

## Data Contract

- Require a prepared contract: entity id, sequence index/time column, series names, value columns, window/stride policy, split strategy, target alignment, and leakage notes.
- `FeatureCollection.calculate(data=...)` accepts a pandas `Series`, pandas `DataFrame`, pandas `DataFrameGroupBy`, or a list of pandas `Series`/`DataFrame`.
- tsflex is built for wide DataFrames and series-list data. Convert long data to a list of named Series rather than blindly pivoting to wide with many `NaN`s.
- Every time series must have a unique name, sortable monotonic index, and compatible index dtype. Multi-indexes and multi-columns are documented as unsupported.
- Output is a sequence-indexed pandas `DataFrame` or list of DataFrames. Feature names follow `<SERIES>__<FEATURE>__w=<WINDOW>__s=<STRIDE>`.

Read `references/tsflex-data-formats.md` before adapting long, wide, asynchronous, grouped, or chunked data.

## Extraction Pattern

```python
import numpy as np
import scipy.stats as ss
from tsflex.features import FeatureCollection, MultipleFeatureDescriptors

fc = FeatureCollection(
    MultipleFeatureDescriptors(
        functions=[np.min, np.mean, np.std, ss.skew],
        series_names=["sensor_a", "sensor_b"],
        windows=["15min", "30min"],
        strides="15min",
    )
)

X = fc.calculate(data=[sensor_a_df, sensor_b_df], approve_sparsity=True, n_jobs=1)
```

Use `n_jobs=1` while debugging. Official docs note multiprocessing is not supported on Windows and may not help for very short extraction jobs.

## Feature Functions

- Define features with `FeatureDescriptor(function, series_name, window, stride)` or expand combinations with `MultipleFeatureDescriptors`.
- Feature functions are callables with signature like `function(*series, **kwargs)` and may return one or multiple values.
- Use `FuncWrapper` when output names, keyword arguments, or `input_type=pd.Series` are required.
- tsflex does not document a fixed built-in feature catalog like tsfresh or TSFEL. It uses user-supplied callables and documented wrappers for external packages.
- Documented wrappers include `seglearn_feature_dict_wrapper`, `tsfel_feature_dict_wrapper`, `tsfresh_settings_wrapper`, and `catch22_wrapper`.

Read `references/tsflex-api-map.md` before changing descriptors, wrappers, chunking, logging, or serialization.

## Irregular and Multivariate Data

- tsflex supports unevenly sampled and asynchronous multivariate data, but feature functions must tolerate variable-length and possibly empty windows.
- When sparsity is expected, pass `approve_sparsity=True` to `.calculate()` and choose robust functions.
- Use `bound_method="inner"`, `"inner-outer"`, or `"outer"` intentionally when extracting over multiple series/columns.
- Use `tsflex.chunking.chunk_data()` to split continuous chunks around gaps before extraction when gaps matter.

## Validation and Metrics

- Validate the final extraction plus downstream model with temporal, group-aware, or stratified folds matching the task.
- For classification, report F1-macro, balanced accuracy, ROC-AUC/PR-AUC where applicable, and confusion matrices.
- For regression, report MAE/RMSE plus scale-aware metrics if useful.
- For clustering, inspect scaling, feature distributions, stability, silhouette/task metrics, and representative raw windows.
- tsflex does not document a dedicated plotting API. Plot raw series/windows, feature distributions, missingness, timing logs, and downstream validation results with pandas/matplotlib or the selected ML stack.

## Anti-Leakage Rules

- Split ids, groups, chunks, windows, or temporal periods before fitting processing, imputation, scaling, PCA, selectors, or downstream models.
- Generate windows/chunks so each feature row uses only observations available at the decision timestamp.
- Do not allow overlapping windows or `sub_chunk_overlap` to cross train/validation/test boundaries unless the task explicitly permits that information.
- Fit any `SeriesPipeline`, interpolation, normalization, robust feature wrapper decisions, `FeatureCollection.reduce()`, selectors, and downstream models only on train folds.
- Keep feature functions, `window`, `stride`, `bound_method`, chunking, sparsity approval, and feature names fixed before final test evaluation.
- Future covariates may be included only if known at prediction time and excluded from the target horizon.

## Common Errors

- Passing numpy arrays directly; tsflex expects pandas objects with named series/columns.
- Duplicate series names after combining DataFrames/Series.
- Mixing incompatible index dtypes, such as a `DatetimeIndex` series with a `RangeIndex` series.
- Using time-string windows like `"15min"` on non-time-indexed data.
- Assuming irregular/asynchronous windows have equal sample counts.
- Pivoting long data to wide and unintentionally introducing many `NaN`s.
- Expecting built-in supervised feature selection or sklearn transformers; README lists sklearn integration as future work.
- Using multiprocessing on Windows despite the documented limitation.

## References

- Read `references/tsflex-data-formats.md` for accepted inputs, long/wide conversion, output shape, and chunking assumptions.
- Read `references/tsflex-api-map.md` for official classes, methods, wrappers, logging, serialization, and limitations.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_tsflex_frame.py` to sanity-check CSV wide or long data before converting to pandas objects.

## Ready Checklist

- Data-prep contract defines entity, index/time, series names, value columns, split strategy, window/stride, and leakage risks.
- Input is pandas Series/DataFrame/list/GroupBy with unique series names and sortable compatible indexes.
- Long data is converted to a series list or reviewed before any pivot.
- Feature functions are documented callables or official wrappers, with output names for multi-output functions.
- Windows/chunks do not cross split or target-horizon boundaries.
- Downstream transformations and models are fit only on train folds.
