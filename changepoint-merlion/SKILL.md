---
name: changepoint-merlion
description: Use Salesforce Merlion for native Bayesian online changepoint detection after validating prepared TimeSeries data, including BOCPD, LevelShift/TrendChange/Auto change kinds, univariate or multivariate inputs, online updates, z-score changepoint scores, thresholded alarms, TSAD metrics, plotting, and leakage-safe evaluation.
---

# Merlion Changepoints

Use this skill after time-series data preparation when the task is online or retrospective changepoint detection with Salesforce Merlion.

Important limitation: the official GitHub repo is archived and read-only as of March 11, 2026. PyPI lists `salesforce-merlion 2.0.4` as the latest release, published June 20, 2024. Prefer Merlion when a project already uses its `TimeSeries`, anomaly detection interface, post-processing rules, evaluator, or dashboard.

## Minimum Install

```bash
pip install salesforce-merlion
```

Optional extras documented by PyPI/README include `salesforce-merlion[all]`, `[dashboard]`, `[spark]`, and `[deep-learning]`. Some anomaly models require JDK, but BOCPD itself is pure Python/scientific stack in the official source.

## Data Contract

- Convert validated pandas `Series` or `DataFrame` to `merlion.utils.TimeSeries` with `TimeSeries.from_pd(...)`.
- Use a time index convertible to `DatetimeIndex`; columns become variables for multivariate detection.
- Require sorted, unique timestamps, numeric finite values, explicit gap/alignment policy, and a temporal train/validation/test split.
- `BOCPD.require_even_sampling` and `BOCPD.require_univariate` are both `False` in the official source, so irregular and multivariate `TimeSeries` inputs are supported by the class.
- For multiple independent entities/panels, fit one BOCPD model per entity unless the variables are one multivariate process.
- Exogenous variables are not part of BOCPD changepoint scoring; the method signature accepts `exog_data` only through the shared detector interface.

Read `references/merlion-data-workflow.md` before adapting panels, alignment, labels, or online evaluation.

## Core Pattern

```python
from merlion.models.anomaly.change_point.bocpd import BOCPD, BOCPDConfig, ChangeKind
from merlion.utils import TimeSeries

train_data = TimeSeries.from_pd(train_df)
test_data = TimeSeries.from_pd(test_df)

config = BOCPDConfig(
    change_kind=ChangeKind.Auto,
    cp_prior=1e-2,
    lag=10,
    min_likelihood=1e-16,
)
model = BOCPD(config)
train_scores = model.train(train_data=train_data)
test_scores = model.get_anomaly_score(time_series=test_data)
alarms = model.get_anomaly_label(time_series=test_data)
```

`get_anomaly_score()` returns a `TimeSeries` of z-scores corresponding to changepoint probability. `get_anomaly_label()` applies Merlion post-processing and thresholding, with BOCPD's default threshold corresponding to at least 50% changepoint probability.

## Detector Choice

- `BOCPD`: Merlion's native changepoint detector, implemented under `merlion.models.anomaly.change_point.bocpd`.
- `ChangeKind.Auto`: trains candidate level-shift and trend-change models and chooses by AICc.
- `ChangeKind.LevelShift`: normal-distribution model for level shifts.
- `ChangeKind.TrendChange`: Bayesian linear-regression model for trend changes.
- Related anomaly detectors such as `DefaultDetector`, isolation forest, spectral residual, LOF, VAE, or forecast-based detectors can flag anomalous regions, but do not document native changepoint segmentation.

Read `references/merlion-api-map.md` before choosing `cp_prior`, `lag`, `min_likelihood`, post-processing, or metrics.

## Fit, Update, Prediction

- Fit initial history with `model.train(train_data=...)`.
- Score new data with `model.get_anomaly_score(time_series=..., time_series_prev=...)`.
- Get thresholded alarms with `model.get_anomaly_label(...)`.
- For streaming workflows, feed chronological batches through `model.update(time_series)` or the evaluator; never revisit future data.
- `lag` controls detection delay/lookback. `lag=None` considers full history; source docs say `lag=0` is not recommended.
- `forecast()` exists because BOCPD inherits a forecasting-detector interface, but changepoint workflows should report scores/alarms unless piecewise forecasts are explicitly needed.

## Evaluation and Plotting

- Use labeled change/anomaly windows as a single-variable Merlion `TimeSeries` of labels.
- Recommended metrics: `TSADMetric.Precision`, `Recall`, `F1`, `PointwisePrecision`, `PointwiseRecall`, `PointwiseF1`, `MeanTimeToDetect`, and NAB scores when alert timing matters.
- Use `TSADEvaluator` to simulate live deployment, retraining, and post-processing on chronological data.
- For pure changepoint timestamps, convert nonzero alarm labels to times and evaluate with a tolerance/delay window outside Merlion if exact segmentation metrics are required.
- Plot with `model.plot_anomaly(time_series=...)`; overlay labels with `merlion.plot.plot_anoms(...)`.

## Anti-Leakage Rules

- Split by time before transforms, imputation policy tuning, threshold tuning, `cp_prior`, `lag`, `min_likelihood`, or `change_kind` selection.
- Fit Merlion transforms and post-processing only on train/validation history.
- Online evaluation must pass batches in timestamp order and update state only with past/current observations.
- Do not tune thresholds or alarm suppression on the final test period.
- If labels describe anomaly intervals, use them only for evaluation or validation-threshold tuning, never to adjust production alarms.
- Align multivariate variables inside each train/history window; avoid global interpolation that sees future data.

## Common Errors

- Calling every Merlion anomaly detector a changepoint detector; only the `change_point` module documents changepoint algorithms.
- Confusing BOCPD z-score outputs with raw probabilities.
- Ignoring `lag` when reporting detection time.
- Passing unsorted or duplicate timestamps and relying on implicit reindexing.
- Treating independent panel entities as variables in one multivariate process.
- Using test labels to train Merlion post-processing or choose thresholds.
- Expecting active upstream maintenance after the repository was archived.

## References

- Read `references/merlion-data-workflow.md` for TimeSeries conversion, alignment, labels, panels, and leakage controls.
- Read `references/merlion-api-map.md` for documented changepoint APIs, metrics, post-processing, and limitations.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_merlion_changepoints.py` to sanity-check CSV inputs and BOCPD parameters.

## Ready Checklist

- Data is sorted, numeric, finite, temporally split, and converted to `TimeSeries`.
- BOCPD `change_kind`, `cp_prior`, `lag`, and `min_likelihood` match the change type and detection-delay budget.
- Threshold/post-processing is trained or chosen only on train/validation history.
- Metrics distinguish pointwise alerts, event-adjusted anomaly metrics, detection delay, and exact changepoint tolerance.
- Plots and reports label Merlion output as changepoint scores/alarms, not offline optimal segmentation.
