# Luminol API Map for Changepoint-Like Workflows

Use exact documented/current source names. Luminol is an anomaly detection and correlation library; do not claim native changepoint segmentation.

## Main Classes

`luminol.anomaly_detector.AnomalyDetector`

```python
AnomalyDetector(
    time_series,
    baseline_time_series=None,
    score_only=False,
    score_threshold=None,
    score_percent_threshold=None,
    algorithm_name=None,
    algorithm_params=None,
    refine_algorithm_name=None,
    refine_algorithm_params=None,
    algorithm_class=None,
)
```

- `time_series`: CSV path, `dict`, or `TimeSeries`.
- `baseline_time_series`: optional CSV path, `dict`, or `TimeSeries`.
- `score_only=True`: computes scores without anomaly periods.
- `algorithm_name`: key in `anomaly_detector_algorithms`.
- `algorithm_params`: algorithm-specific parameters.
- `refine_algorithm_name`: algorithm used to locate the maximum severity timestamp inside each interval. Default is `exp_avg_detector`.
- Public methods: `get_all_scores()`, `get_anomalies()`.

`luminol.modules.anomaly.Anomaly`

- Attributes: `start_timestamp`, `end_timestamp`, `anomaly_score`, `exact_timestamp`.
- Public method: `get_time_window()`.

`luminol.modules.time_series.TimeSeries`

- Construct with `TimeSeries(series)` where `series` is `timestamp -> value`.
- Sorts timestamps, casts timestamps to `int`, casts values to `float`, drops `None`.
- Useful methods include `items`, `iteritems`, `crop`, `smooth`, `normalize`, `average`, `median`, `max`, `min`, `percentile`, `stdev`, and `sum`.

`luminol.correlator.Correlator`

```python
Correlator(
    time_series_a,
    time_series_b,
    time_period=None,
    use_anomaly_score=False,
    algorithm_name=None,
    algorithm_params=None,
)
```

- Public methods: `get_correlation_result()`, `is_correlated(threshold=0)`.
- `CorrelationResult` has `coefficient`, `shift`, and `shifted_coefficient`.
- Default algorithm is `cross_correlator`.

## Anomaly Algorithms

`bitmap_detector`

- Default detector.
- Parameters: `precision`, `lag_window_size`, `future_window_size`, `chunk_size`.
- Uses SAX chunk-frequency differences between lag and future windows.
- Source raises `NotEnoughDataPoints` when lag/future windows are invalid or too small.

`derivative_detector`

- Parameter: `smoothing_factor`.
- Scores deviations of absolute derivatives from exponential moving-average derivatives.
- Best fit when abrupt value changes are the target event.

`exp_avg_detector`

- Parameters: `smoothing_factor`, `use_lag_window`, `lag_window_size`.
- Scores deviation from exponential moving average.
- Default refine algorithm for locating `exact_timestamp`.

`default_detector`

- Combines exponential-average and derivative scores.
- README says it is used when other algorithms fail and is not meant to be explicitly used.

`absolute_threshold`

- Parameters: `absolute_threshold_value_upper`, `absolute_threshold_value_lower`.
- Requires at least one of the threshold values.
- Does not use `baseline_time_series`.

`diff_percent_threshold`

- Parameters: `percent_threshold_upper`, `percent_threshold_lower`.
- Requires aligned `baseline_time_series`.
- Requires at least one percent threshold.

`sign_test`

- Parameters: `percent_threshold_upper`, `percent_threshold_lower`, `offset`, `scan_window`, `confidence`.
- Requires aligned `baseline_time_series`.
- Requires exactly one of upper/lower percent thresholds and a `scan_window`.
- The lower threshold should be negative when detecting drops below baseline.

## Correlation Algorithm

`cross_correlator`

- Parameters: `max_shift_seconds`, `shift_impact`.
- Allows a shift room so related peaks that are slightly apart in time can still correlate.
- Default constants include `DEFAULT_ALLOWED_SHIFT_SECONDS = 60` and `DEFAULT_SHIFT_IMPACT = 0.05`.

## Thresholds

- `score_threshold` overrides threshold selection when supplied.
- Otherwise Luminol uses a detector default threshold where available or `max_anom_score * score_percent_threshold`.
- Default `score_percent_threshold` constant is `0.2`.
- Defaults in `ANOMALY_THRESHOLD` are documented in code for `exp_avg_detector` and `default_detector` as `3`.

## Limitations

- No native changepoint object, penalty, breakpoint count, or optimal segmentation API is documented.
- No documented online/streaming API.
- No documented multivariate detector; anomaly detection is per series.
- Core docs do not document plotting helpers.
- Luminol is old: latest PyPI release is `0.4` from 2017.
- README names only a subset of algorithms; current source also registers `absolute_threshold`, `diff_percent_threshold`, and `sign_test`.
