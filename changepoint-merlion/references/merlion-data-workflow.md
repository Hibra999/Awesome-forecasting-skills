# Merlion Changepoint Data Workflow

Use this reference when converting prepared time-series data into Merlion's `TimeSeries` format for BOCPD.

## Accepted Data

- `merlion.utils.TimeSeries` is the modeling container.
- `TimeSeries.from_pd(df)` accepts pandas `Series`, `DataFrame`, `numpy.ndarray`, or compatible tabular objects after conversion through pandas.
- A pandas `DatetimeIndex` is the safest input. Object indexes are converted with `pd.to_datetime`.
- DataFrame columns become named variables in a multivariate `TimeSeries`; a Series becomes a single-variable `TimeSeries`.
- `TimeSeries.to_pd()` returns a time-indexed pandas `DataFrame`.

## Preparation Contract

Before Merlion:

1. Sort by timestamp and remove or document duplicate timestamps.
2. Keep only numeric finite variables used for detection.
3. Split into train/validation/test by time.
4. Decide gap handling, interpolation, resampling, and alignment inside each split/history window.
5. For labels, create a single-variable `TimeSeries` aligned with the target evaluation period where nonzero means change/anomaly event.

`TimeSeries.from_pd(..., check_times=True, drop_nan=True)` can sort and deduplicate indexes and drop NaNs. Do not rely on that as a substitute for an explicit data audit.

## Alignment

Merlion `TimeSeries` can contain variables sampled at different times. `TimeSeries.align()` supports outer join, inner join, fixed reference, and fixed granularity policies with aggregation and missing-value policies.

For leakage-safe evaluation:

- Fit or choose resampling/interpolation policy on train/validation only.
- Align multivariate variables inside each chronological fold.
- Avoid using a reference index created from future/test timestamps when scoring earlier periods.
- Document whether timestamps are irregular. BOCPD does not require even sampling, but irregular sampling changes the interpretation of delay and windows.

## Univariate, Multivariate, and Panels

- Univariate: one value column.
- Multivariate: multiple columns describing one process at each time.
- Panels/multiple independent series: loop by entity and fit one BOCPD model per entity. Do not mix independent assets, stores, patients, or devices as multivariate channels unless the desired change is joint process change.

For multivariate BOCPD, the source sets `target_seq_index=0` with a warning if `forecast()` is needed and no target was provided. Changepoint scoring can still use multivariate values.

## Online vs Retrospective Use

BOCPD is an online model. Retrospective runs are acceptable for analysis, but feed data in chronological order and report detection delay:

- Initial history: `model.train(train_data=history)`.
- Streaming batches: `model.update(batch)` or `get_anomaly_score(time_series=batch, time_series_prev=context)`.
- Test simulation: use `TSADEvaluator` when retraining or post-processing should mimic production.

Do not score all data, inspect future labels, then move alarms backward to the true changepoint.

## Labels and Changepoints

Merlion's anomaly metrics evaluate nonzero alert labels against anomalous windows. If a project has exact changepoint timestamps:

- Convert timestamps to a binary label window with an allowed tolerance/delay for Merlion TSAD metrics, or
- Extract nonzero alarm times from `get_anomaly_label()` and evaluate exact timestamp tolerance outside Merlion.

Always state which interpretation is used.
