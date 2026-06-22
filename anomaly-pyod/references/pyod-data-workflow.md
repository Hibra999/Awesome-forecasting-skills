# PyOD Data Workflow

Use this reference when converting prepared time-series or tabular data into PyOD anomaly detection inputs.

## Workflow from prepared time-series data

1. Define the detection unit: one timestamp, one entity-timestamp row, one sliding window, one file, or one graph node.
2. Split chronologically before fitting transforms or detectors.
3. Build numeric features inside each fold only. Examples: current value, past lags, rolling means/stds, calendar flags known at detection time, residuals from train-fit baselines, or domain features.
4. Drop ID/time/label columns from `X`. Keep them separately for mapping predictions back.
5. Fit imputers/scalers/encoders on train/reference data only.
6. Fit PyOD detector on train/reference features.
7. Score validation/test with `decision_function`, then apply a preselected threshold or `predict`.
8. Map row-level scores back to timestamps/entities and evaluate.

## Split policy

- Time series: use chronological splits or rolling-origin validation.
- Panel data: split by time within each entity, and ensure features for each entity use only that entity's past unless cross-entity information is available in production.
- Non-temporal tabular data: random/stratified splits may be acceptable, but only after confirming there is no time/order leakage.
- Semi-supervised novelty detection: train on known-normal reference data where possible.
- Supervised outlier classification: PyOD includes `XGBOD`, but if full labels drive the task, compare against standard imbalanced classification workflows too.

## Feature and threshold leakage

Leakage-prone operations:

- Standardization, normalization, imputation, PCA, embeddings, target encoders.
- Rolling features using future or centered windows.
- Learning residual baselines from full history.
- Selecting `contamination` or alert threshold from final test labels.
- Running ADEngine on all data before splitting.

Safe pattern:

- Fit every transform and detector on train/reference.
- Select thresholds and detector families on validation.
- Freeze the pipeline, then report once on test.

## Output interpretation

- `decision_scores_`: training anomaly scores.
- `decision_function(X_new)`: new anomaly scores.
- `labels_`: training binary labels after thresholding.
- `predict(X_new)`: new binary labels.
- `predict_proba(X_new)`: score-derived probabilities; do not assume calibration.
- `predict(..., return_confidence=True)` or `predict_confidence`: confidence diagnostics where supported.

## Evaluation

Recommended metrics when labels exist:

- ROC-AUC for ranking when both classes are present.
- PR-AUC or average precision when anomalies are rare.
- Precision@N when operators can inspect only top alerts.
- Recall@N, F1, confusion matrix, false positives/day, and false negatives.
- For time series, also report contiguous anomaly events and detection delay if labels are event-based.

If labels are unavailable, report score distributions, top-N examples, stability across detectors, known incident overlap, and expert review outcomes. Do not call unsupervised clusters "accuracy" without labels.

## Plotting

- Time series: plot score vs timestamp, threshold, predicted anomalies, and known labels/incidents.
- Tabular: plot score histogram, top-N table, feature slices for top anomalies, and low-dimensional projections for diagnostics only.
- Multimodal: show identifiers, thumbnails/text snippets/audio IDs only when privacy policy allows it.
