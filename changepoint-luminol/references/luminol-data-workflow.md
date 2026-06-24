# Luminol Data Workflow

Use this reference when preparing data for Luminol anomaly windows used as changepoint candidates.

## Accepted Inputs

`AnomalyDetector` and `Correlator` accept:

- CSV path string
- `dict` mapping timestamp to value
- `luminol.modules.time_series.TimeSeries`

For CSVs, `luminol.utils.read_csv` reads the first two comma-delimited columns as timestamp and value. Rows that cannot parse are skipped.

## Timestamp Handling

Official README describes anomaly timestamps as epoch seconds. Source `utils.to_epoch` accepts numeric timestamps directly and converts supported date strings into epoch-millisecond-like values. The safest rule is to keep timestamp units consistent across:

- primary series
- baseline series
- correlated secondary series
- labels
- tolerance windows

If using date strings, map output timestamps back through the same conversion policy before scoring.

## Baselines

Baseline algorithms need aligned primary and baseline series:

- `diff_percent_threshold`
- `sign_test`

Validate that both series use the same timestamp keys and row count before detection. Do not build a baseline from future target information unless that is what production will know at the detection time.

## Panels and Multiple Metrics

Run `AnomalyDetector` independently per entity/metric:

```python
for metric_name, series in metric_series.items():
    detector = AnomalyDetector(series, algorithm_name="derivative_detector")
```

After detection, use `Correlator(primary, candidate, time_period=anomaly.get_time_window())` to rank candidate root-cause metrics in the event window.

## Changepoint Candidate Policy

Luminol returns anomaly windows, not breakpoints. Choose one policy before evaluation:

- `start_timestamp`: process begins changing at anomaly start.
- `end_timestamp`: process change is confirmed when anomaly window ends.
- `exact_timestamp`: most severe timestamp inside the anomaly interval.

Use the same policy for all validation/test folds.

## Evaluation

For labeled changepoints, compare chosen candidate timestamps to labels using a fixed tolerance. For labeled anomalous intervals, count true positives when windows overlap labels under a documented overlap rule.

Report:

- precision, recall, F1
- false positives and false negatives
- detection delay
- anomaly score/severity
- correlated metrics and coefficients for root-cause review

## Leakage Controls

Tune thresholds and algorithms on train/validation windows only. Bitmap uses future windows relative to a timestamp; if the deployment setting is online, either avoid this detector or report that the output is retrospective.
