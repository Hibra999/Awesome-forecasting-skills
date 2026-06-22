---
name: aggregation-tsfel
description: Use TSFEL for time-series aggregation and feature extraction after validating data, including ndarray/Series/DataFrame inputs, univariate and multivariate signals, window_size and overlap extraction, statistical/temporal/spectral/fractal domains, JSON feature configs, dataset_features_extractor file workflows, custom features, sampling-frequency checks, tabular ML matrices, and leakage-safe train/fold feature generation.
---

# TSFEL Aggregation

Use this skill after the time-series data-prep step. TSFEL is best when you need interpretable feature vectors from signals for downstream classification, regression, clustering, anomaly workflows, or exploratory profiling.

Do not treat TSFEL as a model. It extracts feature matrices; downstream estimators, scaling, PCA, feature selection, metrics, and validation are separate steps.

## Minimum Install

```bash
python -m pip install tsfel
```

PyPI and GitHub identify `tsfel 0.2.0` as the latest release, published August 20, 2025.

## Data Contract

- Require a prepared contract: signal/entity id, value columns, sampling frequency `fs`, window policy, split strategy, downstream target alignment, and leakage notes.
- `time_series_features_extractor(config, timeseries, fs=None, window_size=None, overlap=0, ...)` accepts list, `numpy.ndarray`, pandas `Series`, or pandas `DataFrame`.
- TSFEL supports univariate and multivariate series. Multivariate dimensions must be separate columns and share the same sampling frequency.
- Output is always a pandas `DataFrame`: rows are the full time series or generated windows; columns are extracted feature names.
- If input columns are named, TSFEL uses those names as feature prefixes; otherwise it uses numeric prefixes such as `0_`.
- TSFEL does not handle unevenly sampled data by default. Resample/interpolate before extraction or use documented dataset extraction resampling intentionally.

Read `references/tsfel-data-formats.md` before adapting in-memory arrays, multivariate data, windows, or dataset files.

## Extraction Pattern

```python
import tsfel

cfg = tsfel.get_features_by_domain()
X = tsfel.time_series_features_extractor(
    cfg,
    data,
    fs=sampling_frequency,
    window_size=window_size,
    overlap=0.0,
    n_jobs=None,
)
```

Use `get_features_by_domain("statistical")`, `"temporal"`, `"spectral"`, or `"fractal"` for individual domains. The default `get_features_by_domain()` is documented for temporal, statistical, and spectral features; fractal features are documented as disabled in default config files and usually suited to longer signals.

## Feature Configuration

- TSFEL extracts more than 65 distinct features across statistical, temporal, spectral, and fractal domains.
- Feature configuration is a dictionary/JSON structure. Edit `use: yes/no` and feature `parameters` to control extraction.
- Spectral features require a correct `fs`; the FAQ says a default sampling rate exists, but recommends passing the real rate and avoiding spectral features if it is unknown.
- Use `features_path` for custom feature modules and follow the documented personalised-features format.
- Use `n_jobs` for parallel extraction; TSFEL disables multiprocessing by default on Windows unless explicitly configured.

Read `references/tsfel-api-map.md` before changing domains, configs, custom features, dataset extraction, or parallelism.

## Multiple Series and Dataset Files

- For many independent series already in one table, loop over ids and call `time_series_features_extractor` per id/window; add the id to each output row.
- For files on disk, use `dataset_features_extractor(main_directory, feat_dict, search_criteria=..., time_unit=..., resample_rate=..., window_size=..., output_directory=...)`.
- Dataset extraction assumes delimited files where the first column is timestamp and following columns are values; it can interpolate/resample unsynchronized files at the user-specified rate.

## Validation and Metrics

- Validate the downstream pipeline with temporal, group-aware, or stratified folds depending on labels and leakage risk.
- For classification, report F1-macro, balanced accuracy, ROC-AUC/PR-AUC where applicable, and confusion matrices.
- For regression, report MAE/RMSE plus scale-aware metrics if useful.
- For clustering, inspect feature scaling, feature distributions, cluster stability, silhouette/task-specific metrics, and representative raw windows.
- TSFEL does not document a dedicated plotting API for feature extraction; plot raw signals/windows and downstream feature/model diagnostics with pandas/matplotlib or the chosen ML stack.

## Anti-Leakage Rules

- Split ids, groups, windows, or temporal periods before imputation, resampling decisions, scaling, PCA, feature selection, or downstream model fitting.
- Create windows after splitting when labels depend on time. Do not split already-overlapping windows across train/test boundaries.
- Rolling/windowed feature rows must end at or before the decision timestamp and exclude the target horizon.
- Fit resampling/interpolation parameters, `StandardScaler`, PCA, selectors, classifiers, regressors, and clustering postprocessing only on train folds.
- Use future samples/covariates only if they are known at prediction time and are not part of the target horizon.
- Keep `fs`, `window_size`, `overlap`, feature domain, custom config, and feature parameters fixed from train/validation design before final test evaluation.

## Common Errors

- Calling `time_series_feature_extractor` instead of the documented/source `time_series_features_extractor`.
- Omitting `fs` while extracting spectral features.
- Feeding unevenly sampled data without resampling.
- Generating overlapping windows before the train/test split.
- Treating one multivariate signal as multiple independent entities, or merging independent ids into one multivariate signal.
- Forgetting that output rows correspond to windows when `window_size` is set.
- Enabling fractal features on short windows despite documentation that they are typically for longer signals.
- Expecting TSFEL to provide supervised feature selection or sklearn transformers; these are not documented as core TSFEL APIs.

## References

- Read `references/tsfel-data-formats.md` for accepted inputs, output shape, windows, and dataset extraction assumptions.
- Read `references/tsfel-api-map.md` for extractor functions, configs, feature domains, custom features, and documented gaps.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_tsfel_frame.py` to sanity-check CSV signals or per-id panels before extraction.

## Ready Checklist

- Data-prep contract defines id, time/order, value columns, `fs`, split strategy, window policy, and leakage risks.
- Input is a list, ndarray, Series, DataFrame, or documented dataset-file layout.
- Sampling is equally spaced or intentionally resampled before extraction.
- Feature config/domain is explicit and saved for reproducibility.
- Windows are generated leakage-safely and do not cross split or target-horizon boundaries.
- Scaling, PCA, selection, and downstream modeling are fit only on train folds.
