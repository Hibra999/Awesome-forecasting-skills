---
name: changepoint-luminol
description: Use LinkedIn Luminol for changepoint-like anomaly event detection after validating prepared time-series data, including AnomalyDetector anomaly windows, exact anomaly timestamps, bitmap/derivative/EMA/threshold/sign-test algorithms, baseline comparisons, Correlator root-cause ranking, temporal evaluation, plotting outside Luminol, and anti-leakage safeguards.
---

# Luminol Changepoints

Use this skill after time-series data preparation when the task can be framed as detecting anomalous event windows or abrupt changes in one metric, then optionally correlating the event period with other metrics.

Important limitation: official Luminol docs describe anomaly detection and correlation, not native changepoint segmentation or online minimal-delay changepoint detection. Report anomaly windows and `exact_timestamp` as changepoint candidates, not optimal breakpoints.

## Minimum Install

```bash
pip install luminol
```

PyPI and the README list `luminol 0.4` as latest, released December 11, 2017. The repo requirements are `numpy`, `scipy`, and `future==0.16.0`; no explicit `python_requires` is documented.

## Data Contract

- Use one numeric target series at a time.
- `AnomalyDetector` accepts a CSV path, `dict` of `timestamp -> value`, or `luminol.modules.time_series.TimeSeries`.
- CSV input is read from the first two columns. Timestamps may be numeric or a documented timestamp string format.
- `TimeSeries` sorts timestamps, casts timestamps to `int`, casts values to `float`, and drops `None` values.
- For panels or multiple metrics, run one detector per metric/entity. Use `Correlator` for two-series root-cause analysis after a candidate event window exists.
- Baseline algorithms require an aligned `baseline_time_series`.

Read `references/luminol-data-workflow.md` before adapting CSVs, baseline comparisons, panels, or labeled changepoints.

## Core Pattern

```python
from luminol.anomaly_detector import AnomalyDetector

ts = {
    1704067200: 10.0,
    1704153600: 10.3,
    1704240000: 28.0,
    1704326400: 10.1,
}

detector = AnomalyDetector(
    ts,
    algorithm_name="derivative_detector",
    algorithm_params={"smoothing_factor": 0.2},
    score_threshold=1.5,
)

for anomaly in detector.get_anomalies():
    start, end = anomaly.get_time_window()
    candidate_time = anomaly.exact_timestamp
    severity = anomaly.anomaly_score
```

Correlate other metrics inside an anomaly window:

```python
from luminol.correlator import Correlator

period = detector.get_anomalies()[0].get_time_window()
correlator = Correlator(ts, other_ts, time_period=period)
result = correlator.get_correlation_result()
```

## Algorithm Choice

- `bitmap_detector`: default; compares SAX chunk-frequency changes across lag/future windows and is documented as good for huge datasets.
- `derivative_detector`: use when abrupt value changes are the main interest.
- `exp_avg_detector`: use when values are in a roughly stationary range; also default refine algorithm.
- `default_detector`: fallback when other algorithms fail due to insufficient data; not meant to be explicitly used.
- `absolute_threshold`: use known upper/lower business thresholds.
- `diff_percent_threshold`: compare current series against an aligned baseline by percent difference.
- `sign_test`: compare current series against an aligned baseline with a sliding sign test.

Read `references/luminol-api-map.md` before selecting parameters or claiming detector support.

## Detection and Outputs

- `get_all_scores()` returns a `TimeSeries` of anomaly scores.
- `get_anomalies()` returns `Anomaly` objects.
- `Anomaly.get_time_window()` returns `(start_timestamp, end_timestamp)`.
- `Anomaly.exact_timestamp` is the timestamp in the anomaly period where severity is highest.
- Use `score_threshold` for explicit score cutoffs, or `score_percent_threshold` for a fraction of max score. The code uses `score_percent_threshold`; the README text contains older spelling variants.

## Evaluation and Plotting

- With labeled changepoints, score `exact_timestamp` or window start/end against labels with a documented tolerance.
- Report precision, recall, F1, false positives, false negatives, detection delay, and anomaly severity distribution.
- For root-cause ranking, report `CorrelationResult.coefficient`, `shift`, and `shifted_coefficient`.
- Luminol core docs do not document a plotting API. Plot the source series, anomaly score series, anomaly windows, and exact timestamps with matplotlib or Plotly outside Luminol.

## Anti-Leakage Rules

- Never random split time series. Use chronological validation/test periods.
- Tune `score_threshold`, `score_percent_threshold`, detector choice, windows, smoothing factors, thresholds, baseline offsets, and correlation thresholds on training/validation periods only.
- Do not compute baselines from future data unless that baseline is known at detection time.
- For bitmap-like lag/future window methods, label output as retrospective unless the future window would be available in the intended workflow.
- If converting anomaly windows to changepoints, define whether the changepoint is window start, window end, or `exact_timestamp` before seeing test labels.
- For panels, detect per entity and tune per training fold; do not leak aggregate future behavior into entity-level thresholds.

## Common Errors

- Calling Luminol a native changepoint segmentation library.
- Passing wide multivariate data into `AnomalyDetector`; use one target series at a time.
- Forgetting that CSV parsing uses only the first two columns.
- Using `diff_percent_threshold` or `sign_test` without an aligned baseline.
- Assuming date string timestamps and numeric epoch timestamps use the same unit without checking conversion.
- Treating correlation as causation; `Correlator` ranks related metrics, not confirmed root causes.
- Tuning thresholds on final test anomalies.

## References

- Read `references/luminol-api-map.md` for exact APIs, algorithms, parameters, and outputs.
- Read `references/luminol-data-workflow.md` for data formats, timestamp units, baselines, panels, labels, and leakage.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_luminol_changepoints.py` to sanity-check CSV inputs and common detector settings.

## Ready Checklist

- Task is framed as anomaly event/changepoint-candidate detection, not optimal segmentation.
- Input is one sorted numeric time series with documented timestamp units.
- Detector and parameters match documented Luminol APIs.
- Baseline requirements are satisfied where needed.
- Evaluation policy maps anomaly windows to changepoint candidates before test scoring.
- Temporal validation and anti-leakage rules are documented.
