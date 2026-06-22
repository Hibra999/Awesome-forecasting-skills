---
name: changepoint-kats
description: Use Kats for changepoint, level-shift, online Bayesian changepoint, robust statistical changepoint, rolling CUSUM, trend, and statistical-change detection after validating prepared time-series data, including TimeSeriesData inputs, univariate/multivariate constraints, thresholds, windows, priors, evaluation, plotting, and leakage-safe offline or online workflows.
---

# Kats Changepoints

Use this skill after time-series data preparation when the task is changepoint or significant process-change detection with Kats.

Kats supports multiple detection families. Some are explicit changepoint detectors; others are trend/anomaly/statistical-change detectors. Keep that distinction in outputs.

## Minimum Install

```bash
pip install --upgrade pip
pip install kats
```

Kats also documents `MINIMAL_KATS=1 pip install kats`, but warns that minimal installation disables many functions and logs warnings. PyPI lists `kats 0.2.0` as the latest release, published March 15, 2022.

## Data Contract

- Use `kats.consts.TimeSeriesData`.
- Initialize from a pandas `DataFrame`, `Series`, `DatetimeIndex`, or explicit `time` and `value`.
- DataFrame input defaults to a time column named `time`; use `time_col_name` when needed.
- `value` can be a pandas `Series` for univariate or `DataFrame` for multivariate.
- Require sorted time, numeric finite values, documented frequency/gaps, train/current split for online-like detectors, and entity id if looping over panels.
- For multiple independent series, fit detectors per entity unless using a documented multivariate/vectorized path.

Read `references/kats-data-workflow.md` before adapting panels, online detection, multivariate inputs, or production windows.

## Core Patterns

Single level-shift changepoint with CUSUM:

```python
from kats.consts import TimeSeriesData
from kats.detectors.cusum_detection import CUSUMDetector

ts = TimeSeriesData(df, time_col_name="time")
detector = CUSUMDetector(ts)
change_points = detector.detector(change_directions=["increase", "decrease"])
detector.plot(change_points)
```

Online Bayesian changepoints:

```python
from kats.detectors.bocpd import BOCPDetector, BOCPDModelType

detector = BOCPDetector(ts)
change_points = detector.detector(
    model=BOCPDModelType.NORMAL_KNOWN_MODEL,
    lag=10,
    threshold=0.5,
)
detector.plot(change_points)
```

Rolling CUSUM model for multiple level shifts:

```python
from kats.detectors.cusum_model import CUSUMDetectorModel

model = CUSUMDetectorModel(
    scan_window=43200,
    historical_window=604800,
    threshold=0.01,
    change_directions=["increase"],
)
response = model.fit_predict(ts)
change_points_unix = model.cps
```

## Detector Choice

- `CUSUMDetector`: explicit level-shift changepoint detector; assumes one increase and/or one decrease changepoint and Gaussian mean-change testing.
- `BOCPDetector`: Bayesian Online Changepoint Detection; use for online-style detection where `lag` controls delay/certainty.
- `RobustStatDetector`: robust univariate statistical changepoint detector from official source; uses smoothed differences, z-scores, and p-value cutoff.
- `CUSUMDetectorModel`: `DetectorModel` wrapper that runs CUSUM repeatedly over `historical_window` and `scan_window` to detect multiple level-shift points.
- `MKDetector`: Mann-Kendall trend detector; use for persistent monotonic trend alerts, not generic distributional changepoints.
- `StatSigDetectorModel` / `MultiStatSigDetectorModel`: rolling test-vs-control statistical-change detectors; use when comparing current windows to historical windows.
- `ProphetDetectorModel`: documented as Prophet-based anomaly detection, not a changepoint detector.

Read `references/kats-api-map.md` before choosing parameters or reporting detector capabilities.

## Fit, Detection, and Outputs

- Classic detectors use `Detector(data).detector(...)` and return changepoint objects, usually with `start_time`, `end_time`, and `confidence`.
- Detector models use `fit`, `predict`, or `fit_predict` and return `AnomalyResponse` or store changepoints in model attributes such as `CUSUMDetectorModel.cps`.
- Plot with detector-specific `.plot(change_points)` where documented. CUSUM multivariate plotting is explicitly not supported in the API docs.
- For BOCPD, use `get_change_prob()` and `get_run_length_matrix()` after detection when probability diagnostics are needed.

## Evaluation

- With labeled changepoints, use `kats.detectors.changepoint_evaluator.get_cp_index`, `f_measure`, and `true_positives`.
- `f_measure` expects 0-based changepoint locations and a `margin` tolerance.
- The Turing benchmark evaluator is documented for changepoint benchmark evaluation, but custom datasets must match the documented columns.
- Without labels, report window/threshold sensitivity, detection delay for online workflows, false alert review, and domain plausibility.

## Anti-Leakage Rules

- Split train/history/current/test periods before interpolation, smoothing, seasonality removal, threshold tuning, priors, or detector selection.
- For online detectors, pass only past data as `historical_data` and current/future windows as `data`; do not let future points influence earlier alerts.
- Tune `threshold`, `lag`, `changepoint_prior`, `scan_window`, `historical_window`, `step_window`, `p_value_cutoff`, `smoothing_window_size`, `window_size`, `n_control`, and `n_test` on validation periods only.
- If removing seasonality or interpolating gaps, fit the policy inside each train/history window.
- Evaluate with temporal cuts, labeled retrospective periods, or rolling/expanding backtests; never random split.

## Common Errors

- Treating all Kats detectors as changepoint detectors; some are anomaly or trend detectors.
- Using `CUSUMDetector` for many changepoints in one pass; use `CUSUMDetectorModel` or windowed workflows for multiple level shifts.
- Passing multivariate data to a univariate-only detector such as `RobustStatDetector`.
- Forgetting that `BOCPDetector` reports after a `lag`, so detection time and true change time can differ.
- Setting windows in seconds for `CUSUMDetectorModel` without matching the data frequency.
- Tuning thresholds on the final test window.
- Using minimal Kats installation and expecting all detection dependencies to work.

## References

- Read `references/kats-data-workflow.md` for TimeSeriesData shapes, panels, and online/history splits.
- Read `references/kats-api-map.md` for detector capabilities, parameters, outputs, plotting, and limitations.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_kats_changepoints.py` to sanity-check CSV input, labels, and window sizes.

## Ready Checklist

- Data is sorted, numeric, finite, and converted to `TimeSeriesData`.
- Detector choice matches the documented change type and univariate/multivariate support.
- Offline vs online semantics, window sizes, `lag`, and threshold policy are explicit.
- Evaluation uses labeled changepoints, temporal validation, or rolling/expanding windows.
- No preprocessing, priors, thresholds, or window settings leak validation/test future data.
