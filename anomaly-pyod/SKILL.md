---
name: anomaly-pyod
description: Use PyOD for anomaly/outlier detection after validating prepared tabular or time-indexed feature matrices, including classic fit/predict detectors, time-series detectors, ADEngine selection, contamination thresholds, anomaly scores, labels, probabilities, evaluation metrics, plotting, model persistence, and anti-leakage safeguards.
---

# PyOD Anomaly Detection

Use this skill when the task is point anomaly or outlier detection: assign each observation or timestamp to normal/abnormal, rank observations by anomaly score, or compare PyOD detectors on prepared features.

Important scope: PyOD is a general anomaly detection library, not a forecasting or raw sequence-classification framework. For time series, use either PyOD time-series detectors on ordered arrays or create leakage-safe point/window features first.

## Minimum Install

```bash
pip install pyod
```

PyPI lists `pyod 3.6.1` as latest, released June 17, 2026, requiring Python `>=3.9`. Optional extras include `torch`, `suod`, `xgboost`, `combo`, `pythresh`, `embedding`, `openai`, `huggingface`, `graph`, `mcp`, `audio`, and `all`.

## Data Contract

- Classic tabular detectors expect `X` as a numeric array of shape `(n_samples, n_features)`.
- Time-series detectors accept arrays of shape `(n_timestamps,)` for univariate or `(n_timestamps, n_channels)` for multivariate; pandas DataFrames and lists are documented as auto-converted.
- Labels, when available, are for validation/evaluation. Most classic detectors ignore `y` during `fit`.
- PyOD labels use `0` for inliers and `1` for outliers/anomalies.
- Raw variable-length series, event intervals, and panel entities must be converted into aligned point/window features before classic detection.
- For graph detection, use `pyod[graph]` and PyG `Data`; for text/image/audio, use embedding/multimodal APIs and required extras.

Read `references/pyod-data-workflow.md` before preparing time-indexed data, panels, labels, or window features.

## Core Pattern

```python
from pyod.models.iforest import IForest

clf = IForest(contamination=0.05, random_state=42)
clf.fit(X_train)

y_train_labels = clf.labels_
y_train_scores = clf.decision_scores_
y_test_labels = clf.predict(X_test)
y_test_scores = clf.decision_function(X_test)
y_test_proba = clf.predict_proba(X_test)
```

Time-series detector example:

```python
from pyod.models.ts_kshape import KShape

clf = KShape(window_size=20)
clf.fit(X_train_ts)  # shape (n_timestamps,) or (n_timestamps, n_channels)
scores = clf.decision_scores_
labels = clf.labels_
```

Use `ADEngine` when the agent should profile data, select detectors, compare results, and explain findings automatically. Use the classic API when the detector is already chosen.

## Detector Choice

- Fast tabular baselines: `ECOD`, `COPOD`, `HBOS`, `IForest`, `LODA`, `KNN`, `LOF`.
- Linear/subspace and covariance: `PCA`, `KPCA`, `MCD`, `OCSVM`, `RGraph`, `ROD`.
- Clustering/density/local methods: `CBLOF`, `COF`, `HDBSCAN`, `SOD`, `SOS`, `KDE`, `GMM`, `Sampling`, `LOCI`, `LMDD`.
- Ensembles and scalable combinations: `FeatureBagging`, `LSCP`, `SUOD`, `INNE`, `DIF`, `XGBOD`.
- Deep learning: `AutoEncoder`, `VAE`, `DeepSVDD`, `DevNet`, `LUNAR`, `DIF`, `MO_GAAL`, `SO_GAAL`.
- Time series: `TimeSeriesOD`, `MatrixProfile`, `SpectralResidual`, `KShape`, `SAND`, `LSTMAD`, `AnomalyTransformer`.
- Multimodal and graph: `EmbeddingOD`, `MultiModalOD`, and PyG graph detectors where installed.

Read `references/pyod-api-map.md` before claiming exact detector support, extras, modules, or output behavior.

## Evaluation and Plotting

- If labels exist, report ROC-AUC, PR-AUC or average precision, precision@N, recall@N, F1 at the chosen threshold, confusion matrix, and alert volume.
- PyOD utility examples use `evaluate_print` and `precision_n_scores`; scikit-learn metrics are fine for custom reports.
- For time-indexed data, use chronological validation windows and report alert delay and contiguous anomaly segments in addition to point metrics.
- Plot anomaly scores over time, threshold, predicted labels, and true labels if available. For non-temporal tabular data, plot score distributions and 2D projections only as diagnostics.

## Anti-Leakage Rules

- Split time-indexed data chronologically before scaling, encoding, imputation, feature extraction, threshold selection, ADEngine runs, or detector fitting.
- Fit scalers, PCA/embeddings, window aggregations, PyOD detectors, contamination thresholds, and PyThresh thresholders on train/reference data only.
- Do not use random splits for time series unless the task is explicitly non-temporal tabular anomaly detection.
- Rolling/window features must use only past or currently available values for online detection. Centered windows are retrospective.
- Use labels only for evaluation and model selection on validation folds, not for fitting unsupervised detectors.
- Tune `contamination`, detector family, window size, thresholding method, and ensemble weights on validation only; final test is one-time reporting.

## Common Errors

- Calling `fit_predict` or fitting scalers on the full time series before temporal validation.
- Treating `predict_proba` as calibrated probability without checking method and calibration.
- Setting `contamination` from final test labels.
- Using `MatrixProfile` as an out-of-sample detector; official docs call it transductive.
- Feeding raw timestamps, IDs, or label columns as numeric features.
- Mixing entities in a panel without entity-safe splits and per-entity lag/window construction.
- Using graph/text/image/audio APIs without installing the documented extras.

## References

- Read `references/pyod-api-map.md` for detector inventory, APIs, extras, outputs, utilities, and limitations.
- Read `references/pyod-data-workflow.md` for tabular/time-series data preparation, panels, thresholds, and leakage controls.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_pyod_anomaly_input.py` to sanity-check CSV features, labels, temporal order, and contamination before modeling.

## Ready Checklist

- Task is point anomaly/outlier detection or anomaly score ranking.
- Input features are numeric, finite, leakage-safe, and split according to temporal assumptions.
- Detector choice, extras, and data shape match official PyOD docs.
- Threshold/contamination is selected on train/validation only.
- Metrics and plots report scores, binary alerts, and known label quality without test leakage.
