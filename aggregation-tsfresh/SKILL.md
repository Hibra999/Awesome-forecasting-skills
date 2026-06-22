---
name: aggregation-tsfresh
description: Use tsfresh for time-series aggregation and feature extraction after validating data, including flat, stacked, and dict input formats, feature calculator settings, supervised relevance filtering, sklearn transformers, Dask/Spark scaling, rolling windows, and leakage-safe tabular ML feature generation.
---

# tsfresh Aggregation

Use this skill after the time-series data-prep step. tsfresh is best when you need to aggregate ordered time-series observations into a classical tabular feature matrix, where each row represents an entity/window/id and columns are statistical time-series characteristics for downstream classification, regression, clustering, anomaly detection, or ranking.

Do not treat tsfresh as a forecaster, classifier, or clustering model. It extracts and optionally selects features; downstream modeling happens separately.

## Minimum Install

```bash
pip install tsfresh
pip install "tsfresh[dask]"
```

Use the Dask extra only for large data workflows. The official README documents `tsfresh[matrixprofile]` only for backwards-compatible matrixprofile features on Python 3.8.

## Data Contract

- Require a prepared contract: entity/window id, sorted time column or guaranteed row order, value columns/kinds, target labels if supervised selection is used, train/validation/test split, and leakage notes.
- Supported inputs to `extract_features()` are pandas flat/wide DataFrames, stacked/long DataFrames, or a dictionary of flat DataFrames keyed by kind.
- Core columns are `column_id`, optional `column_sort`, optional `column_kind`, and optional `column_value`.
- Special columns `column_id`, `column_sort`, `column_kind`, and `column_value` must not contain `NaN`, `Inf`, or `-Inf`.
- Output is a pandas `DataFrame` with one row per id and one column per extracted feature.
- Equidistant timestamps are not generally required, but some calculators only make sense on equidistant series.

Read `references/tsfresh-data-formats.md` before adapting flat, stacked, dict, rolling, or large-data inputs.

## Extraction Pattern

```python
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

X_features = extract_features(
    timeseries,
    column_id="id",
    column_sort="time",
    default_fc_parameters=EfficientFCParameters(),
)
```

Feature settings documented by tsfresh:

- `ComprehensiveFCParameters`: broad default feature set.
- `EfficientFCParameters`: excludes high-cost calculators.
- `MinimalFCParameters`: small set for smoke tests and fast baselines.
- `default_fc_parameters`, `kind_to_fc_parameters`, and `from_columns()` for custom or reproduced extraction.

Feature families include statistics, change/energy measures, autocorrelation, trend, entropy, AR/ADF tests, binned/frequency-style summaries, and many other official calculators. Use the official feature list instead of inventing calculator names.

## Supervised Selection

Use relevance filtering only inside train data or cross-validation folds. `select_features()`, `calculate_relevance_table()`, `extract_relevant_features()`, `FeatureSelector`, and `RelevantFeatureAugmenter` use the target `y` and can leak labels if run before splitting.

```python
from tsfresh import extract_features, select_features
from tsfresh.transformers import FeatureSelector, PerColumnImputer

X_train_features = extract_features(train_ts, column_id="id", column_sort="time")

imputer = PerColumnImputer()
X_train_features = imputer.fit_transform(X_train_features)

selector = FeatureSelector()
X_train_selected = selector.fit_transform(X_train_features, y_train)

X_valid_features = extract_features(valid_ts, column_id="id", column_sort="time")
X_valid_selected = selector.transform(imputer.transform(X_valid_features))
```

Use `extract_relevant_features()` for concise train-only workflows, not for pre-split global feature creation.

## sklearn and Scaling

- `FeatureAugmenter` wraps feature extraction and appends extracted features to tabular `X`; set its `timeseries_container`.
- `RelevantFeatureAugmenter` extracts and selects relevant features during `fit`.
- `FeatureSelector` selects already-extracted relevant columns.
- `PerColumnImputer` learns per-column replacement values on `fit`.
- For large data, use `n_jobs`, `chunksize`, Dask DataFrames with `tsfresh[dask]`, `pivot=False`, `dask_feature_extraction_on_chunk`, or `spark_feature_extraction_on_chunk` where documented.

## Rolling Windows

For forecasting-style feature tables, use `roll_time_series()` or `make_forecasting_frame()` only after defining the prediction time and horizon. Official docs state `make_forecasting_frame()` is limited to one-dimensional time series of one id and kind.

Every feature window must end at or before the prediction timestamp. Do not create rolling windows that include the target horizon unless the task explicitly permits future information.

## Validation and Metrics

- Validate the feature extraction pipeline with temporal, stratified, or group-aware folds that match the downstream task.
- For classification, prefer F1-macro, balanced accuracy, ROC-AUC/PR-AUC where probabilities exist, and confusion matrices for imbalance.
- For regression, prefer MAE/RMSE plus scale-aware metrics if relevant.
- For clustering or pattern discovery, evaluate downstream clustering stability, silhouette or task-specific scores, and inspect feature distributions.
- Plot original series/window examples, feature distributions, missing/imputed feature counts, relevance tables, and model validation results. tsfresh itself does not document a dedicated plotting API.

## Anti-Leakage Rules

- Split ids, entities, windows, or temporal periods before imputation, scaling, feature selection, target relevance tests, or model fitting.
- Fit `PerColumnImputer`, scalers, selectors, downstream models, and `RelevantFeatureAugmenter` only on train folds.
- If using rolling windows, every row's extracted features must use only observations available at that row's decision time.
- Align `y` by id after extraction; never let labels from validation/test influence selected feature columns.
- Future covariates or future time rows may be used only if they are known at prediction time and are not target leakage.
- Keep frequency, sorting, cutoffs, window length, and horizon explicit in the data-prep contract.

## Common Errors

- Passing `NaN`, `Inf`, or `-Inf` in id, sort, kind, or value columns.
- Using stacked data without both `column_kind` and `column_value`.
- Omitting `column_sort` when rows are not already sorted.
- Running `extract_relevant_features()` on the full dataset before train/test split.
- Misaligning `y.index` with extracted feature row ids.
- Pivoting huge Dask outputs unintentionally; use `pivot=False` for very large intermediate results.
- Expecting tsfresh to classify, forecast, or cluster without a downstream estimator.

## References

- Read `references/tsfresh-data-formats.md` for accepted input shapes and output contracts.
- Read `references/tsfresh-api-map.md` for official functions, transformers, settings, rolling utilities, and scaling hooks.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_tsfresh_frame.py` to sanity-check CSV flat or stacked inputs before extraction.

## Ready Checklist

- Data-prep contract defines id, time order, value columns/kinds, target alignment, split strategy, and leakage risks.
- Input format is flat, stacked, or dict-of-flat and matches the tsfresh column arguments.
- Train/validation/test split happens before imputation, scaling, relevance filtering, or downstream modeling.
- Feature settings are `MinimalFCParameters`, `EfficientFCParameters`, `ComprehensiveFCParameters`, or documented custom settings.
- Rolling/forecasting windows use only data available at the prediction time.
- Downstream metrics match classification, regression, clustering, or anomaly-use requirements.
