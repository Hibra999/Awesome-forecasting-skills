---
name: changepoint-alibi-detect
description: Use Seldon Alibi Detect for changepoint-like distribution-change monitoring after validating prepared time-series arrays, including online MMD/LSDD/CVM/FET drift detectors, rolling offline drift tests, reference windows, ERT, window sizes, preprocessing/backends, state handling, evaluation delay, plotting, and anti-leakage safeguards.
---

# Alibi Detect Changepoints

Use this skill after time-series data preparation when a changepoint is defined as a distribution shift between stable reference history and later observations.

Important limitation: official Alibi Detect docs describe outlier, adversarial, and drift detection, not native offline changepoint segmentation. Use drift alarms as changepoint candidates and say so explicitly.

## Minimum Install

```bash
pip install alibi-detect
```

Extras documented by the README include `alibi-detect[tensorflow]`, `[torch]`, `[keops]`, and `[prophet]`. PyPI lists `alibi-detect 0.13.0` as latest, released December 11, 2025, requiring Python `>=3.9`.

## Data Contract

- Convert prepared time series to ordered arrays: `x_ref` is stable reference history; stream/test rows arrive after it.
- Use `numpy.ndarray` or list inputs. A typical tabular time-series frame becomes `df[value_cols].to_numpy(dtype=float)`.
- For online detectors, call `predict(x_t)` on one observation at a time, without a batch dimension.
- Keep timestamps outside Alibi Detect for plotting and evaluation; predictions return observation counts in `data["time"]`.
- Univariate and multivariate numeric features are supported depending on detector. `FETDriftOnline` requires binary Bernoulli streams.
- For panels, fit one detector per entity unless the columns are a single multivariate process.

Read `references/alibi-detect-data-workflow.md` before adapting panels, context windows, preprocessing functions, or labels.

## Core Pattern

```python
import numpy as np
from alibi_detect.cd import MMDDriftOnline

x_ref = train_df[value_cols].to_numpy(dtype=np.float32)
stream = test_df[value_cols].to_numpy(dtype=np.float32)

cd = MMDDriftOnline(
    x_ref=x_ref,
    ert=500,
    window_size=50,
    backend="pytorch",
    data_type="time-series",
)

alarms = []
for i, x_t in enumerate(stream):
    pred = cd.predict(x_t, return_test_stat=True)
    if pred["data"]["is_drift"] == 1:
        alarms.append((test_df.index[i], pred["data"]))
```

`pred["data"]` includes `is_drift`, `time`, `ert`, and, when requested, `test_stat` plus `threshold`.

## Detector Choice

- `MMDDriftOnline`: multivariate kernel drift; use for general numeric distribution changes.
- `LSDDDriftOnline`: multivariate least-squares density difference; use for density-difference drift alarms.
- `CVMDriftOnline`: continuous univariate-per-feature test, with multivariate correction; use for continuous feature streams.
- `FETDriftOnline`: binary Bernoulli streams; use for accuracy/error-rate or indicator streams.
- Rolling offline candidates: `MMDDrift`, `LSDDDrift`, `KSDrift`, `CVMDrift`, `FETDrift`, `ChiSquareDrift`, `TabularDrift`, `ClassifierDrift`, `LearnedKernelDrift`, `ContextMMDDrift`, `SpotTheDiffDrift`, `ClassifierUncertaintyDrift`, `RegressorUncertaintyDrift`.
- Time-series outlier proxies such as `OutlierProphet`, `SpectralResidual`, and `OutlierSeq2Seq` can flag anomalous intervals, but they are not documented changepoint segmenters.

Read `references/alibi-detect-api-map.md` before selecting detector parameters or claiming capabilities.

## Fit, State, Prediction

- There is no `.fit()` for online drift detectors in the core pattern; construction configures reference data and thresholds.
- Set `ert` as the expected runtime before false detection under no drift.
- Set `window_size` or `window_sizes` to balance delay and sensitivity.
- Use `preprocess_fn` only if fitted on training/reference data. Set `x_ref_preprocessed=True` only when the stored reference was already transformed.
- Save/load online state with `save_state(...)` and `load_state(...)`; reset with `reset_state()`.

## Evaluation and Plotting

- Evaluate alarm timestamps against labeled changepoints with a tolerance or max-delay window.
- Report precision, recall, F1, mean/median detection delay, first-alarm delay, false alarms per time, and realized no-drift runtime versus configured `ert`.
- For rolling offline tests, map each alarm to the end timestamp of the tested window unless a project-specific convention says otherwise.
- Alibi Detect does not document built-in changepoint plots. Plot source series, `test_stat`, `threshold`, and vertical alarm lines with matplotlib.

## Anti-Leakage Rules

- Split reference/train, validation, and test by time before scaling, dimensionality reduction, encoders, thresholds, or detector choice.
- Build `x_ref` only from pre-change or training history available at deployment time.
- Do not tune `ert`, `window_size(s)`, `p_val`, kernels, preprocessing, or `n_bootstraps` on final test labels.
- If using rolling offline tests, each test window must only compare past reference/history to current/future window allowed at that decision time.
- Do not let `update_x_ref` absorb validation/test future data unless that is the explicit production policy.
- For learned preprocessing, train encoders/classifiers on training data only.

## Common Errors

- Presenting Alibi Detect as a native changepoint segmentation library.
- Passing a batch to online `predict`; online detectors expect one instance at a time.
- Forgetting that `CVMDriftOnline` and `FETDriftOnline` need full windows before useful statistics exist.
- Using `FETDriftOnline` on non-binary data.
- Setting `n_bootstraps` far below the requested `ert`, making thresholds noisy.
- Losing timestamps after converting to arrays and then misreporting alarm times.
- Globally scaling or embedding all data before the time split.

## References

- Read `references/alibi-detect-data-workflow.md` for array shapes, windows, panels, labels, and leakage.
- Read `references/alibi-detect-api-map.md` for detector families, parameters, outputs, and limitations.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_alibi_detect_changepoints.py` to sanity-check CSV inputs and online detector settings.

## Ready Checklist

- Task is framed as drift/change alarm detection, not offline optimal segmentation.
- `x_ref` is stable, chronological, and leakage-free.
- Detector, backend, ERT, window sizes, preprocessing, and binary/continuous assumptions match the data.
- Alarm timestamps are mapped back to the original time index.
- Evaluation uses temporal labels/tolerance windows and reports detection delay plus false alarms.
