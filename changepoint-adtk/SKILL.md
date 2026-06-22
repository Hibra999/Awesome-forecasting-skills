---
name: changepoint-adtk
description: Use Arundo ADTK for changepoint-like anomaly event detection after validating prepared pandas time series, including LevelShiftAD, PersistAD, VolatilityShiftAD, SeasonalAD, AutoregressionAD, threshold/outlier detectors, multivariate detectors, event conversion, metrics, plotting, temporal validation, and anti-leakage safeguards.
---

# ADTK Changepoints

Use this skill after time-series data preparation when abrupt level shifts, temporary persistence/spikes, volatility shifts, seasonal violations, autoregressive changes, or multivariate anomaly events should be treated as changepoint candidates.

Important limitation: official ADTK docs describe unsupervised/rule-based anomaly detection, not native optimal changepoint segmentation or online minimal-delay changepoint detection. Report detected anomaly points/events as changepoint candidates and document the mapping.

## Minimum Install

```bash
pip install adtk
```

The README and docs list Python 3.5+ and recommend PyPI. GitHub and PyPI list `v0.6.2` as latest, released April 17, 2020.

## Data Contract

- Use a `pandas.Series` or `pandas.DataFrame` indexed by `pandas.DatetimeIndex`.
- Call `adtk.data.validate_series` after sorting, deduplicating, frequency checks, and train/test splitting policy are defined.
- Univariate detectors accept `Series`; when given a `DataFrame`, ADTK applies them independently to each column.
- For intrinsic multivariate changes, use documented multivariate detectors such as `MinClusterDetector`, `OutlierDetector`, `RegressionAD`, or `PcaAD`.
- ADTK does not document forecasting-style future covariates. Treat extra columns as simultaneous multivariate signals, not known-future regressors.
- Output is binary anomaly labels (`Series`/`DataFrame`) or event lists when `return_list=True`; `to_events` converts labels to timestamp or interval events.

Read `references/adtk-data-workflow.md` before adapting panel data, labels, event intervals, or validation windows.

## Core Pattern

```python
import pandas as pd
from adtk.data import to_events, validate_series
from adtk.detector import LevelShiftAD
from adtk.visualization import plot

df = pd.read_csv("metric.csv", parse_dates=["ts"])
s = df.sort_values("ts").set_index("ts")["value"]
s = validate_series(s)

s_train = s.loc[: "2024-06-30"]
s_test = s.loc["2024-07-01" :]

detector = LevelShiftAD(window=24, c=6.0, side="both")
detector.fit(s_train)
labels = detector.detect(s_test)
events = to_events(labels)

plot(s_test, anomaly=events, anomaly_tag="span", anomaly_color="red")
```

For exploratory training-period diagnostics only, `fit_detect(train)` is acceptable. For honest validation, fit on train and call `detect` or `score` on validation/test.

## Detector Choice

- `LevelShiftAD`: permanent abrupt level changes; strongest native fit for changepoint candidates.
- `PersistAD`: temporary abrupt changes/spikes relative to recent past.
- `VolatilityShiftAD`: changes in variability using `std`, `iqr`, or `idr`.
- `SeasonalAD`: violations of a fixed seasonal pattern.
- `AutoregressionAD`: changes in autoregressive behavior, including some cyclic non-seasonal cases.
- `ThresholdAD`, `QuantileAD`, `InterQuartileRangeAD`, `GeneralizedESDTestAD`: point outliers; use only when point outliers are valid process-change signals.
- `RollingAggregate`, `DoubleRollingAggregate`, `Pipeline`, and `Pipenet`: custom change-feature pipelines.
- `OrAggregator` and `AndAggregator`: combine detector outputs when the logic is documented before validation.

Read `references/adtk-api-map.md` before claiming exact detector support or selecting parameters.

## Evaluation and Plotting

- Use temporal train/validation/test cuts or `adtk.data.split_train_test`; never random splits.
- With labels, use `detect` plus `adtk.metrics.recall`, `precision`, `f1_score`, and `iou`, or detector `.score(..., scoring="recall"|"precision"|"f1"|"iou")`.
- For changepoint labels, predefine event-to-point mapping: event start, event end, midpoint, or any-overlap within a tolerance window.
- Report false positives, false negatives, F1, IoU for intervals, and detection delay when labels are point changes.
- Plot with `adtk.visualization.plot(ts, anomaly=..., anomaly_tag="span"|"marker")`.

## Anti-Leakage Rules

- Split chronologically before fitting detectors, thresholds, transforms, scalers, PCA, clustering, regressors, or aggregators.
- Do not run `fit_detect` on final validation/test data for performance claims.
- Tune `window`, `c`, `side`, `agg`, frequency, thresholds, model choices, and event mapping on train/validation only.
- Rolling features must use only information available at the prediction time. `LevelShiftAD` and `VolatilityShiftAD` compare adjacent windows and are retrospective around a boundary unless the right window is already observed.
- If future or contextual columns are used in multivariate detection, they must be available at detection time.
- Preserve timestamp order, frequency semantics, panel boundaries, and validation windows across `validate_series`, `to_events`, and plotting.

## Common Errors

- Calling ADTK an optimal segmentation library.
- Passing non-`DatetimeIndex` data or assuming numeric timestamps are accepted by `validate_series`.
- Letting `validate_series` silently sort or drop duplicate timestamps without recording the data fix.
- Applying a univariate detector to a `DataFrame` and assuming intrinsic multivariate modeling.
- Using centered/right-hand windows for online detection without documenting the delay.
- Mixing event intervals and point changepoints without a scoring convention.
- Tuning thresholds or detector families after inspecting final test labels.

## References

- Read `references/adtk-api-map.md` for detectors, methods, parameters, outputs, metrics, and plotting.
- Read `references/adtk-data-workflow.md` for data formats, panels, validation, event conversion, and leakage.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_adtk_changepoints.py` to sanity-check CSV inputs and common detector settings.

## Ready Checklist

- Task is anomaly/changepoint-candidate detection, not optimal segmentation.
- Input is a sorted pandas time series with `DatetimeIndex`, numeric values, and documented frequency/gaps.
- Detector family and parameters match official ADTK APIs.
- Trainable components are fit on train only.
- Event-to-changepoint mapping and metric tolerance are fixed before test scoring.
- Plots, metrics, and diagnostics use temporal validation without leakage.
