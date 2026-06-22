# Kats Changepoint Data Workflow

Use a prepared time-series contract before Kats detection: entity id if applicable, time column, numeric value columns, frequency/gaps, train/history/current/test windows, candidate detector family, labels if available, and leakage notes.

## TimeSeriesData Inputs

`TimeSeriesData` is Kats' core structure. Official docs say it can be initialized from:

- `pandas.DataFrame`
- `pandas.Series`
- `pandas.DatetimeIndex`
- explicit `time` and `value`

DataFrame input defaults to `time_col_name="time"`. The value side is a pandas `Series` for univariate data or a pandas `DataFrame` for multivariate data.

Recommended setup:

```python
from kats.consts import TimeSeriesData

ts = TimeSeriesData(df, time_col_name="time")
```

## Frequency and Missing Data

Kats exposes:

- `infer_freq_robust()` for frequency inference with missing data.
- `validate_data(validate_frequency=..., validate_dimension=...)`.
- `interpolate(freq=..., method=...)` with linear, backward fill, or forward fill.
- `is_data_missing()` and `to_dataframe()`.

For detection, decide frequency and interpolation before running detectors. If interpolation is used in validation/backtests, fit/choose that policy inside the train/history fold only.

## Multiple Entities and Panels

For multiple independent series:

1. Split by entity and time.
2. Build one `TimeSeriesData` per entity.
3. Run detectors per entity unless the detector explicitly documents multivariate/vectorized support.
4. Store outputs with entity id, detector name, timestamp, index, confidence/score, and parameters.

Do not concatenate independent entities into one time axis.

## Offline vs Online

- `CUSUMDetector`, `RobustStatDetector`, and `MKDetector` operate over the supplied series and are best treated as retrospective/windowed detectors unless explicitly embedded in a rolling workflow.
- `BOCPDetector` is documented as Bayesian Online Changepoint Detection and reports after a `lag`.
- `CUSUMDetectorModel` uses `historical_window`, `scan_window`, and `step_window` to run rolling CUSUM and stores detected Unix-time changepoints in `cps`.
- `StatSigDetectorModel` and `MultiStatSigDetectorModel` compare current data to historical data through rolling windows.

For operational alerts, simulate the same history/current split that production will use.

## Changepoint Outputs

Common output objects derive from `TimeSeriesChangePoint`, which has:

- `start_time`
- `end_time`
- `confidence`

Detector-specific objects add fields such as CUSUM direction, index, deltas, p-values, or BOCPD model metadata. Convert outputs to a stable table for review.
