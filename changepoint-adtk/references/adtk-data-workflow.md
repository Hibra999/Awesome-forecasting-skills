# ADTK Data Workflow

Use this reference when adapting prepared time-series data into ADTK changepoint-candidate workflows.

## Required input shape

ADTK data utilities and detectors expect:

- `pandas.Series` for one numeric target.
- `pandas.DataFrame` for multiple aligned numeric signals.
- `pandas.DatetimeIndex` on all time series.

`validate_series(ts, check_freq=True, check_categorical=False)` validates the index, sorts timestamps, removes duplicated timestamps by keeping the first occurrence, and can infer frequency when possible. Record every correction before modeling.

## From prepared data

Starting from the base data-prep skill output:

1. Keep only one entity/metric for univariate detection, or one aligned wide `DataFrame` for intrinsic multivariate detection.
2. Parse timestamps with pandas and sort by time.
3. Deduplicate timestamps according to a documented policy before or during `validate_series`.
4. Split chronologically into train, validation, and test before fitting trainable detectors or transforms.
5. Run `validate_series` independently inside each fold to avoid learning frequency or cleanup behavior from future rows.
6. Fit the detector on train and call `detect` on validation/test.

For panel data, loop per entity and keep entity-specific windows, thresholds, frequency, labels, and detected events. Do not concatenate unrelated entities into one time index unless that is a documented multivariate design.

## Labels and events

- `to_events(labels, freq_as_period=True, merge_consecutive=None)` converts binary anomaly labels to events.
- `to_labels(lists, time_index, freq_as_period=True)` converts event lists back to binary labels.
- `validate_events(event_list, point_as_interval=False)` validates and merges overlapping event intervals.
- `expand_events` can expand labeled windows for tolerance rules.

For changepoint evaluation, decide the conversion before scoring:

- event start as the change onset;
- event end as recovery/end of a collective anomaly;
- event midpoint as a rough breakpoint;
- any-overlap within a tolerance window as event-level detection.

## Temporal validation

ADTK includes `split_train_test(ts, mode=1, n_splits=1, train_ratio=0.7)` for time-series cross-validation. You may also use explicit calendar cutoffs. The rule is the same: no random splits and no fitting on validation/test.

Use validation folds for:

- detector family choice;
- window length and `min_periods`;
- `c`, quantiles, thresholds, `alpha`, seasonal frequency, regression/PCA/clustering model parameters;
- event-to-changepoint mapping and tolerance.

Use final test only once for reporting.

## Retrospective vs online use

ADTK can be used operationally only if every feature and window is available at detection time. Be explicit:

- `PersistAD` compares current behavior against near past and can be closer to online use depending on parameters.
- `LevelShiftAD` and `VolatilityShiftAD` compare adjacent windows around a candidate boundary; these are retrospective unless the right-hand window has already elapsed.
- Seasonal and autoregression detectors require enough historical context and trained baselines.
- Multivariate detectors require all signal columns at detection time.

## Common preparation failures

- Non-`DatetimeIndex` input.
- Duplicate timestamps with undocumented keep/drop behavior.
- Irregular sampling treated as regular because a frequency was inferred on the full dataset.
- Wide `DataFrame` passed to a univariate detector without realizing ADTK runs per-column independent detectors.
- Missing values left without a documented imputation/drop policy.
- Labels in point format scored against intervals without a tolerance rule.
