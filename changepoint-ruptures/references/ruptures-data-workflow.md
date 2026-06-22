# ruptures Data and Workflow

Use this reference before converting prepared time-series data into `ruptures` inputs.

## Signal Shape

`ruptures` examples and docs fit algorithms on numeric numpy signals:

- Univariate: shape `(n_samples,)`.
- Multivariate: shape `(n_samples, n_features)`.

Rows are ordered samples. Columns are signal dimensions. Keep timestamps outside the signal and map predicted sample indexes back to timestamps after detection.

## Breakpoint Convention

Predicted and true breakpoint lists include the final sample index `n_samples`.

Example:

```python
signal, true_bkps = rpt.pw_constant(1000, 3, 4)
# true_bkps might be [196, 394, 603, 801, 1000]
```

The first elements are regime boundaries; the last element marks the end of the signal.

## Multiple Entities or Panels

`ruptures` detects change points on one signal at a time. For panel data:

1. Split by entity.
2. Sort each entity by time.
3. Convert each entity to a signal matrix.
4. Fit/predict separately unless the scientific question really defines a joint multivariate signal.

Do not concatenate unrelated entities into one signal unless a boundary between entities is intentionally modeled as a change.

## Offline vs Online

Official docs describe `ruptures` as an off-line change point detection package. Offline segmentation uses the complete signal, including future observations relative to earlier timestamps.

For online or operational anomaly detection:

- Use `ruptures` only inside an explicit rolling/expanding simulation wrapper.
- Record detection delay against known or labeled change times.
- Do not use full-series breakpoints to claim real-time detection performance.

## Preprocessing

Recommended checks before fit:

- Finite numeric values.
- Sorted rows.
- Consistent sampling or documented irregular sampling policy.
- Train-only scaling, detrending, embedding, PCA, or feature extraction.
- Domain-informed `min_size`, `jump`, and `width`.

## Leakage Notes

- If the goal is retrospective segmentation, fitting on a full historical series is acceptable.
- If the goal is prediction, forecasting, or online alerting, full-series segmentation leaks future information.
- If breakpoints become downstream features, compute them inside each train fold or historical window only.
