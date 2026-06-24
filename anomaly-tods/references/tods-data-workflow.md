# TODS Data Workflow

Use this reference before converting raw time-series tables into TODS datasets or D3M pipeline inputs.

## Input preparation

Expected practical input:

- pandas `DataFrame` from CSV or equivalent table;
- ordered timestamp column;
- one or more numeric time-series feature columns;
- optional entity/series identifier for panels;
- label/target column separated by `target_index` when labels exist.

`generate_dataset(df, target_index=...)` is the README path for turning the table into a TODS dataset. Keep `target_index` stable and verify that the label column is not treated as an `Attribute`.

## Detection unit

Choose and document one unit:

- point-wise: each timestamp/entity-timestamp is a candidate outlier;
- pattern-wise: a subsequence/window is a candidate outlier;
- system-wise: a set of time series is a candidate outlier.

For pattern-wise outputs, keep window size, step size, left/right indices, and mapping to timestamps explicit before evaluation.

## Safe validation workflow

1. Sort and deduplicate by entity/time.
2. Split chronologically before transformations, feature analysis, AutoML search, or detector fitting.
3. Build any pipeline description using only train/validation design decisions.
4. Fit data-processing primitives, time-series transforms, feature analysis, and detectors on train only.
5. Search pipelines with `BruteForceSearch` only inside train/validation budget.
6. Freeze the selected pipeline and evaluate once on test.
7. Map output labels/scores back to original timestamps/entities.

Do not use TODS `KFoldSplit` for time-indexed anomaly reports unless the task is non-temporal or the fold construction is explicitly time-safe. Prefer `FixedSplit`, `TrainScoreSplit`, the TODS time-series split entry point after verifying its exact installed name, or external chronological splits.

## Metrics

Official examples use `F1_MACRO`. For rare anomaly detection, also report:

- anomaly-class precision, recall, and F1;
- PR-AUC or average precision;
- ROC-AUC only when both classes are present and ranking is meaningful;
- precision@N/top-K hit rate;
- false positives per time period;
- alert volume and operator budget;
- delay or overlap for subsequence/event labels.

## Leakage risks

Avoid:

- imputing/validating continuity with future timestamps;
- fitting scalers or decomposition on the full series;
- creating moving averages or subsequence windows with future values for online detection;
- searching pipelines on the final test period;
- choosing `target_index`, metric, window length, contamination, or primitive family after inspecting test labels;
- mixing panel entities without entity-aware splits.

## When to avoid TODS

Use another library when:

- you need a lightweight scikit-learn-style detector API;
- modern Python/TensorFlow compatibility is required and cannot be pinned;
- the task is forecasting, changepoint segmentation, or root-cause analysis rather than outlier detection;
- you cannot install D3M/TODS legacy dependencies.
