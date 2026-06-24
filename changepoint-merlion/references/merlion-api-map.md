# Merlion Changepoint API Map

Use exact documented names from the official repository. Do not rename general anomaly detectors as native changepoint segmenters.

## Native Changepoint Module

- Package: `merlion.models.anomaly.change_point`
- Native implementation: `merlion.models.anomaly.change_point.bocpd`
- Classes:
  - `BOCPDConfig`
  - `BOCPD`
  - `ChangeKind`

`merlion.models.anomaly.change_point.__init__` states that change point models implement the anomaly detector interface but are specialized for detecting change points in time series.

## BOCPD

`BOCPD` implements Bayesian Online Change Point Detection from Adams and MacKay (2007).

Documented properties and behavior:

- `_online_model` returns `True`.
- `require_even_sampling` returns `False`.
- `require_univariate` returns `False`.
- `get_anomaly_score()` returns a `TimeSeries` of z-scores corresponding to probability that each point is a changepoint.
- `get_anomaly_label()` returns post-processed anomaly/changepoint alarm scores through the shared detector interface.
- `update(time_series)` updates internal BOCPD state and returns scores.
- `forecast(time_stamps=...)` returns predictions and standard errors from the underlying piecewise model.
- If `change_kind=ChangeKind.Auto`, training tries level-shift and trend-change candidates and selects by AICc.

## BOCPDConfig Parameters

| Parameter | Meaning |
| --- | --- |
| `change_kind` | One of `ChangeKind.Auto`, `ChangeKind.LevelShift`, `ChangeKind.TrendChange`, or matching string names. |
| `cp_prior` | Prior probability of changepoints. Source default is `1e-2`. |
| `lag` | Maximum allowed delay/lookback in number of steps; `None` considers full history. Source says `lag=0` is not recommended. |
| `min_likelihood` | Discards changepoint hypotheses below this likelihood. Lower values improve accuracy at more time/space cost. |
| `max_forecast_steps` | Accepted by config but ignored by BOCPD changepoint logic. |

Default BOCPD threshold is `AggregateAlarms(alm_threshold=norm.ppf((1 + 0.5) / 2), min_alm_in_window=1)`, corresponding to at least 50% changepoint probability after z-score conversion.

## ChangeKind

- `ChangeKind.Auto`: choose the conjugate prior automatically.
- `ChangeKind.LevelShift`: uses `MVNormInvWishart`; model points with a normal distribution to detect level shifts.
- `ChangeKind.TrendChange`: uses `BayesianMVLinReg`; model points as a linear function of time to detect trend changes.

## Shared Detector Interface

Merlion anomaly detectors support:

- `model = DetectorClass(config)`
- `model.train(train_data, anomaly_labels=None, train_config=None, post_rule_train_config=None)`
- `model.get_anomaly_score(time_series, time_series_prev=None)`
- `model.get_anomaly_label(time_series, time_series_prev=None)`
- `model.save(...)` and `DetectorClass.load(...)` through the base model interface.

Configs may contain trainable transforms, post-processing rules, `enable_calibrator`, and `enable_threshold`.

## Post-Processing

Relevant thresholding classes in `merlion.post_process.threshold`:

- `Threshold`
- `AggregateAlarms`
- `AdaptiveThreshold`
- `AdaptiveAggregateAlarms`

Use post-processing to convert scores into operational alarms. Tune thresholds and alarm suppression on train/validation only.

## Evaluation

`merlion.evaluate.anomaly` provides:

- `TSADMetric.MeanTimeToDetect`
- `TSADMetric.F1`, `Precision`, `Recall`
- `TSADMetric.PointwiseF1`, `PointwisePrecision`, `PointwiseRecall`
- `TSADMetric.PointAdjustedF1`, `PointAdjustedPrecision`, `PointAdjustedRecall`
- `TSADMetric.NABScore`, `NABScoreLowFN`, `NABScoreLowFP`
- `TSADMetric.F2`, `F5`
- `TSADEvaluatorConfig(max_early_sec=None, max_delay_sec=None, ...)`
- `TSADEvaluator`

The default anomaly metrics use revised point-adjusted scoring. For exact changepoint timestamps, add a project-specific tolerance metric around predicted nonzero alarm times.

## Plotting

Official README demonstrates:

```python
from merlion.plot import plot_anoms

fig, ax = model.plot_anomaly(time_series=test_data)
plot_anoms(ax=ax, anomaly_labels=test_labels)
```

`BOCPD.get_figure(time_series=...)` updates on provided data before delegating to the common plotting interface.

## Documented Limitations

- The upstream GitHub repository is archived and read-only as of March 11, 2026.
- PyPI latest release is `salesforce-merlion 2.0.4` from June 20, 2024.
- Merlion's TSAD metrics evaluate anomaly/alert windows, not offline optimal segmentation.
- BOCPD is online; retrospective use must report possible `lag` delay.
- Exogenous data is not used by BOCPD changepoint scoring.
- Related anomaly detectors can be useful context but are not documented native changepoint algorithms.
