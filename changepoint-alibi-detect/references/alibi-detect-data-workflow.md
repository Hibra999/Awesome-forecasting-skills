# Alibi Detect Changepoint Data Workflow

Use this reference when converting prepared time-series data into Alibi Detect drift-monitoring arrays.

## Accepted Data

- Core detector inputs are `numpy.ndarray` or Python lists.
- Offline drift detectors take a batch `x` in `predict(x, ...)`.
- Online drift detectors take one instance `x_t` in `predict(x_t, return_test_stat=True)`.
- Timestamps are not part of the detector input. Keep a parallel timestamp index to map `data["time"]` and loop indexes back to calendar time.

## Reference and Stream Split

Define:

- `x_ref`: stable reference distribution, usually training/pre-change history.
- validation stream: chronological period for detector/parameter selection.
- test stream: final chronological period for reporting.

Do not choose reference rows after inspecting test labels. For operational use, `x_ref` should represent what would be available before deployment.

## Shapes

Typical time-series tabular case:

```python
x_ref = train_df[value_cols].to_numpy(dtype=np.float32)   # (n_ref, n_features)
stream = test_df[value_cols].to_numpy(dtype=np.float32)   # (n_stream, n_features)
```

For online detectors:

```python
for x_t in stream:
    pred = cd.predict(x_t)
```

Do not pass `stream` as one batch to online detectors.

## Detector-Specific Data

- `MMDDriftOnline` and `LSDDDriftOnline`: multivariate numeric samples, with TensorFlow or PyTorch backend.
- `CVMDriftOnline`: continuous streams; applies univariate CVM tests per feature and can apply multivariate correction.
- `FETDriftOnline`: binary values only, encoded as `0/1` or `False/True`; useful for model error streams.
- Offline `KSDrift`, `CVMDrift`, `FETDrift`, `ChiSquareDrift`, and `TabularDrift` have feature-type assumptions; read the API map before use.

## Preprocessing

Alibi Detect supports `preprocess_fn` and backend-specific helpers. Treat preprocessing like a model:

- Fit scalers, PCA, encoders, or classifiers only on train/reference data.
- If `x_ref_preprocessed=True`, verify `x_ref` was transformed using train-only logic.
- For high-dimensional time series, use train-only dimensionality reduction before drift tests.
- Keep preprocessing deterministic during evaluation unless stochasticity is part of the validated production policy.

## Panels and Multiple Series

For independent entities, loop per entity:

```python
for entity_id, frame in df.groupby("entity_id"):
    # build x_ref and stream for this entity only
```

Use a multivariate detector for multiple channels of the same process, not unrelated entities.

## Labels and Evaluation Windows

If labels are exact changepoints, convert alarms to timestamps and score with tolerance/delay windows. If labels are anomalous intervals, count a true positive when an alarm falls within the allowed interval or within a documented delay from the interval start.

For online detectors, the alarm time is the time of the newly observed point that triggered `is_drift == 1`, not the start of the sliding window unless your project defines that convention explicitly.
