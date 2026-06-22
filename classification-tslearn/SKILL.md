---
name: classification-tslearn
description: Use tslearn 0.8.1 for time-series classification after ts-classification-data-prep, including 3D arrays shaped (n_ts, sz, d), variable-length NaN padding via to_time_series_dataset, univariate and multivariate samples, KNeighborsTimeSeriesClassifier, TimeSeriesSVC, LearningShapelets, TimeSeriesMLPClassifier, NonMyopicEarlyClassifier, tslearn preprocessing, fit/predict/predict_proba, stratified validation, and anti-leakage pipelines.
---

# tslearn Classification

Use this skill after `ts-classification-data-prep`. tslearn is best when the user has labeled complete time-series instances and wants sklearn-compatible time-series classifiers using DTW/GAK distances, learned shapelets, a simple MLP wrapper, or early classification.

Do not use this skill for forecasting. A sample is a whole time series; `y` is one class label per series.

## Minimum Install

```bash
python -m pip install tslearn
conda install -c conda-forge tslearn
python -m pip install "tslearn[all_features]"
```

Use the core install first. Use `[all_features]` or explicit backend packages when using `LearningShapelets`, because the shapelets module requires Keras v3+ and a backend.

## Data Contract

- Require a completed `ts-classification-data-prep` contract: `X`, `y`, split IDs, class balance, channel order, sampling rate, padding/truncation or variable-length policy, missing-value policy, and leakage notes.
- tslearn format is `X.shape == (n_ts, sz, d)`: number of series, timestamps, dimensions/features. Univariate data has `d=1`.
- `y` is 1D array-like with `len(y) == n_ts`.
- Use `tslearn.utils.to_time_series_dataset` for lists/arrays; unequal lengths are padded with trailing `NaN` to `(n_ts, max_sz, d)`.
- Do not confuse tslearn with sktime `numpy3D`: tslearn is `(samples, time, channels)`, not `(samples, channels, time)`.
- Read `references/tslearn-data-validation.md` before using variable length, resampling, external sklearn classifiers, or custom CV.

## Classifier Selection

Official documented classifiers:

- Distance based: `tslearn.neighbors.KNeighborsTimeSeriesClassifier` with metrics such as `dtw`.
- Kernel SVM: `tslearn.svm.TimeSeriesSVC`, default kernel `gak`; set `probability=True` before `fit` if probabilities are required.
- Shapelets: `tslearn.shapelets.LearningShapelets`; learns shapelets, classifies, and can `transform` series to shapelet-distance features.
- Neural network wrapper: `tslearn.neural_network.TimeSeriesMLPClassifier`; reshapes equal-sized series for sklearn `MLPClassifier`.
- Early classification: `tslearn.early_classification.NonMyopicEarlyClassifier`; predicts class and earliness timestamp.

tslearn docs do not list dictionary, ROCKET, HIVE-COTE-style ensemble, CNN/ResNet/InceptionTime, or transformer classifier families.

Read `references/tslearn-classifier-map.md` before claiming variable-length support, probability support, backend needs, or limitations.

## Training Pattern

Fit every preprocessing step inside a pipeline or inside each train fold:

```python
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

pipe = Pipeline([
    ("scale", TimeSeriesScalerMeanVariance()),
    ("clf", KNeighborsTimeSeriesClassifier(n_neighbors=3, metric="dtw")),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(
    pipe,
    X_train,
    y_train,
    cv=cv,
    scoring=["accuracy", "balanced_accuracy", "f1_macro"],
)
print(classification_report(y_test, y_pred))
```

For `TimeSeriesSVC`, configure probabilities before training:

```python
from tslearn.svm import TimeSeriesSVC

svc = TimeSeriesSVC(kernel="gak", probability=True, class_weight="balanced")
svc.fit(X_train, y_train)
proba = svc.predict_proba(X_test)
```

## Feature Extraction And Shape Handling

- Use `LearningShapelets.fit(X_train, y_train)` to learn supervised shapelets from train only; `transform(X)` returns shapelet-distance features.
- For unequal length, prefer documented variable-length classifiers: `KNeighborsTimeSeriesClassifier`, `TimeSeriesSVC`, and `LearningShapelets`.
- For classifiers requiring equal size, resample or pad/truncate only according to the train/fold contract. `TimeSeriesMLPClassifier` requires equal-sized time series.
- Use `TimeSeriesScalerMeanVariance`, `TimeSeriesScalerMinMax`, `TimeSeriesResampler`, or `TimeSeriesImputer` inside the train/CV pipeline.
- `NonMyopicEarlyClassifier` is for early-decision tasks; report both class metrics and earliness/early-classification cost.

## Evaluation

- Use stratified splits/CV for class imbalance: `train_test_split(..., stratify=y)` and `StratifiedKFold`.
- Balanced labels: report accuracy plus macro/weighted F1.
- Imbalanced labels: prefer balanced accuracy, F1-macro, per-class precision/recall, confusion matrix, and ROC-AUC/average precision when `predict_proba` is available and the class setup supports it.
- Use sklearn `GridSearchCV` or `cross_validate`; keep scalers, imputers, resamplers, shapelet learning, flattening, and classifiers inside the fold.
- If samples are ordered by subject/device/time, use group-aware or blocked split logic from `ts-classification-data-prep`; do not let related instances cross folds.

## Anti-Leakage Rules

- Split train/validation/test before fitting scalers, imputers, resamplers, shapelets, SVM gamma selection, feature extraction, or class balancing.
- Fit transformations only on train folds through a pipeline or explicit fold loop.
- Do not compute padding/truncation lengths, resampling size, imputation values, or normalization statistics from held-out test data unless they are fixed external metadata.
- Use stratified CV for imbalanced classes; use grouped stratification when the prep contract identifies subjects/devices/entities.
- Future-looking rolling features or labels created before `ts-classification-data-prep` must use only past data relative to each sample.

## Common Errors

- Passing arrays shaped `(n_samples, n_channels, series_length)` instead of tslearn `(n_samples, series_length, n_channels)`.
- Assuming all classifiers accept variable length; the official variable-length classification list is KNN, SVC, and LearningShapelets.
- Calling `predict_proba` on `TimeSeriesSVC` without setting `probability=True` before `fit`.
- Learning shapelets or scalers on the full dataset before CV.
- Using `TimeSeriesMLPClassifier` on NaN-padded unequal-length data.
- Expecting documented built-in stratified CV, calibration, residual diagnostics, or dictionary/ROCKET/ensemble classifiers from tslearn.

## References

- Read `references/tslearn-classifier-map.md` for classifier capabilities and limitations.
- Read `references/tslearn-data-validation.md` for data formats, variable length, CV, and leakage-safe preprocessing.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_tslearn_array.py` to validate `.npy` `X` and `.npy`/CSV labels before fitting.

## Ready Checklist

- `ts-classification-data-prep` contract is complete and leakage risks are documented.
- `X` is 3D `(n_ts, sz, d)` and `len(y) == n_ts`.
- Equal/variable length, multivariate channels, missing values, and imbalance are explicitly handled.
- Every scaler, imputer, resampler, shapelet learner, and classifier is fit only on train folds.
- Classifier choice matches data shape, variable-length needs, probability needs, and backend dependencies.
- Evaluation uses stratified or group-aware folds and reports imbalance-aware metrics.
