# tsfresh API Map

Use exact documented names. Do not invent feature calculators, transformers, or parameters.

## Extraction

| API | Purpose | Notes |
| --- | --- | --- |
| `tsfresh.extract_features` | Extracts feature columns from a time-series container. | Accepts flat, stacked, or dict inputs; returns a pandas `DataFrame` by default. |
| `tsfresh.feature_extraction.extract_features` | Same core extraction API. | Parameters include `column_id`, `column_sort`, `column_kind`, `column_value`, `default_fc_parameters`, `kind_to_fc_parameters`, `chunksize`, `n_jobs`, `impute_function`, `distributor`, and `pivot`. |
| `tsfresh.convenience.extract_relevant_features` | Extracts features and selects those relevant to `y`. | Supervised; run only on train/fold data. |

## Feature Settings

| API | Purpose |
| --- | --- |
| `ComprehensiveFCParameters` | Broad default set of feature calculators and parameters. |
| `EfficientFCParameters` | Similar scope but excludes high computational cost features. |
| `MinimalFCParameters` | Small feature set for smoke tests and fast baselines. |
| `from_columns()` | Reconstructs extraction settings from selected/existing feature column names. |
| `default_fc_parameters` | Applies one feature setting map to all kinds. |
| `kind_to_fc_parameters` | Applies kind-specific feature settings. |

## Feature Filtering and Imputation

| API | Purpose | Leakage rule |
| --- | --- | --- |
| `tsfresh.select_features` | Selects relevant already-extracted features using target `y`. | Fit/select on train only. |
| `tsfresh.feature_selection.relevance.calculate_relevance_table` | Calculates feature relevance/significance table. | Use only inside train/folds. |
| `tsfresh.utilities.dataframe_functions.impute` | Replaces non-finite feature values in a feature matrix. | For rigorous validation, prefer train-fitted `PerColumnImputer`. |
| `tsfresh.transformers.PerColumnImputer` | sklearn-style imputer that learns replacement values on `fit`. | Fit on train only. |

tsfresh documents the FRESH relevance process as feature extraction, significance testing, then multiple-test correction using the Benjamini-Yekutieli procedure.

## sklearn Transformers

| Transformer | Purpose |
| --- | --- |
| `FeatureAugmenter` | Extracts features from a `timeseries_container` and augments a tabular `X`; `fit` does not learn extraction parameters. |
| `RelevantFeatureAugmenter` | Extracts and selects relevant features during `fit`, then extracts selected features during `transform`. |
| `FeatureSelector` | Selects columns from an already-extracted feature matrix. |
| `PerColumnImputer` | Imputes per-column non-finite values with statistics learned on train. |

When using these in sklearn pipelines, set the time-series container for the train fold only, then update it for validation/test transform calls without refitting selectors or imputers.

## Rolling Utilities

| API | Purpose | Limitation |
| --- | --- | --- |
| `roll_time_series` | Builds rolled/shifted windows for feature extraction. | Enforce past-only windows for forecasting or decision-time tasks. |
| `make_forecasting_frame` | Convenience helper for one-dimensional forecasting-style feature/target creation. | Official docs limit it to one-dimensional time series of one id and one kind. |

## Large Data Hooks

| API | Purpose |
| --- | --- |
| `n_jobs` | Parallel jobs for local extraction. |
| `chunksize` | Controls chunk sizes for extraction/distribution. |
| `pivot=False` | Keeps extraction output in a less-wide form for large workflows. |
| `dask_feature_extraction_on_chunk` | Lower-level Dask extraction helper for grouped chunks. |
| `spark_feature_extraction_on_chunk` | Lower-level Spark extraction helper for grouped chunks. |

## Feature Calculator Families

The official feature list is broad and version-specific. Examples documented in the official list include `abs_energy`, `absolute_maximum`, `absolute_sum_of_changes`, `agg_autocorrelation`, `agg_linear_trend`, `approximate_entropy`, `ar_coefficient`, `augmented_dickey_fuller`, `autocorrelation`, `benford_correlation`, and `binned_entropy`.

For complete coverage, inspect the official "Feature calculators" page or `tsfresh.feature_extraction.feature_calculators` for the installed version.
