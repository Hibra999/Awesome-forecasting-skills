---
name: classification-sktime
description: Use sktime for time-series classification after ts-classification-data-prep, including Panel data formats such as numpy3D, pd-multiindex, df-list, nested_univ, equal/unequal length handling, univariate and multivariate samples, distance/kernel, dictionary, interval, feature, shapelet, ROCKET, hybrid, ensemble, early, deep learning and foundation classifiers, ClassifierPipeline, SklearnClassifierPipeline, predict/predict_proba, stratified cross-validation, model selection, padding/truncation, and anti-leakage safeguards.
---

# sktime Classification

Use this skill after `ts-classification-data-prep`. sktime is best when the task is supervised assignment of complete time-series instances to predefined classes, using sklearn-like `fit`, `predict`, `predict_proba`, pipelines, cross-validation, and a broad zoo of time-series classifiers.

Do not use forecasting APIs for classification. Each row/instance is a whole time series sample; labels are class labels for instances, not future values.

## Minimum Install

```bash
pip install sktime
pip install "sktime[all_extras]"
```

Start with core `sktime`; install soft dependencies only when an estimator asks for them. Deep learning, `pyts`, `tslearn`, `torch`, or other adapters may require extras or separate packages.

## Data Contract

- Require a completed `ts-classification-data-prep` contract: `X`, `y`, class labels, split IDs, instance order, channel names, sampling policy, padding/truncation policy, missing-value policy, and leakage notes.
- Preferred `Panel` formats: `numpy3D` with shape `(n_instances, n_variables, n_timepoints)` or `pd-multiindex` with row index `(instance, time)` and columns as variables.
- `df-list` can represent variable length, unequal indexes, and different variable sets; `pd-multiindex` can represent unequal support but not different variable sets; `numpy3D` requires equal length/index and the same variables.
- `y` is a 1D array-like label vector with one label per instance.
- Use `sktime.datatypes.check_raise`/`check_is_mtype` or `scripts/validate_numpy3d.py` before fitting.

Read `references/sktime-data-validation.md` before using variable length, multivariate data, missing values, padding/truncation, sklearn pipelines, or CV.

## Classifier Selection

Use official class names from `sktime.classification.*`. Main families:

- Distance/kernel: `KNeighborsTimeSeriesClassifier`, `ElasticEnsemble`, `ShapeDTW`, proximity tree/forest, `TimeSeriesSVC`.
- Dictionary: `BOSSEnsemble`, `ContractableBOSS`, `IndividualBOSS`, `IndividualTDE`, `MUSE`, `TemporalDictionaryEnsemble`, `WEASEL`, `MrSEQL`, `MrSQM`.
- Interval/forest: `TimeSeriesForestClassifier`, `CanonicalIntervalForest`, `DrCIF`, `RandomIntervalSpectralEnsemble`, `SupervisedTimeSeriesForest`.
- Feature extraction: `Catch22Classifier`, `SummaryClassifier`, `TSFreshClassifier`, `FreshPRINCE`, `RandomIntervalClassifier`, `SignatureClassifier`, `MatrixProfileClassifier`.
- Shapelet/ROCKET/hybrid: `ShapeletTransformClassifier`, shapelet-learning adapters, `RocketClassifier`, `Arsenal`, `HIVECOTEV1`, `HIVECOTEV2`.
- Deep/foundation: CNN/FCN/ResNet/RNN/GRU/LSTM/InceptionTime/TapNet/MVTSTransformer variants plus `MantisClassifier`, `MomentFMClassifier`, `TSPulseClassifier`.
- Composition: `ClassifierPipeline`, `SklearnClassifierPipeline`, `ColumnEnsembleClassifier`, `MultiplexClassifier`, `BaggingClassifier`, `WeightedEnsembleClassifier`, `TSCGridSearchCV`.

Read `references/sktime-classifier-map.md` before claiming a classifier, dependency, multivariate support, unequal-length support, or `predict_proba` support.

## Training Pattern

```python
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sktime.classification.feature_based import SummaryClassifier

clf = SummaryClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test) if clf.get_tag("capability:predict_proba") else None

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(
    clf,
    X_train,
    y_train,
    cv=cv,
    scoring=["accuracy", "f1_macro", "balanced_accuracy"],
)
print(classification_report(y_test, y_pred))
```

Use `ClassifierPipeline` or `make_pipeline` so transformations are fit inside each train fold:

```python
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.transformations.exponent import ExponentTransformer

pipe = ExponentTransformer() * KNeighborsTimeSeriesClassifier()
```

## Feature Extraction And Shape Handling

- Use classifier-native feature extractors where possible, e.g. `Catch22Classifier`, `SummaryClassifier`, `TSFreshClassifier`, ROCKET/shapelet classifiers.
- If using standalone transforms such as `Rocket`, `MiniRocket`, `ShapeletTransform`, `RandomShapeletTransform`, `SummaryTransformer`, `Catch22`, or `TSFreshFeatureExtractor`, put them in a pipeline before a classifier.
- For unequal length, either select classifiers/tags with `capability:unequal_length=True` or add a train-fitted transformer such as `PaddingTransformer` or `TruncationTransformer` inside the pipeline.
- For missing values, use `Imputer` or a classifier that documents missing-value capability.

## Evaluation

- Use stratified splits for class imbalance: `train_test_split(..., stratify=y)` and `StratifiedKFold`.
- Balanced classes: report accuracy plus macro/weighted F1.
- Imbalanced classes: prefer `balanced_accuracy`, `f1_macro`, per-class recall/precision, confusion matrix, and ROC-AUC/average precision when probabilities and problem shape support them.
- Use `TSCGridSearchCV` or sklearn `GridSearchCV`/`RandomizedSearchCV`; keep every scaler, padder, shapelet/dictionary/ROCKET feature extractor, and model inside the CV pipeline.

## Anti-Leakage Rules

- Split into train/validation/test before fitting any scaler, padder, imputer, shapelet, dictionary, ROCKET, TSFresh, PCA, feature selector, or resampler.
- Fit transformations only on train folds via `ClassifierPipeline`, `SklearnClassifierPipeline`, or sklearn/sktime CV wrappers.
- Do not compute global padding/truncation lengths from test data unless that length is a fixed external contract from `ts-classification-data-prep`.
- Use stratified CV for imbalanced labels.
- Keep grouped subjects/devices out of both train and test if the prep contract identifies grouping; use group-aware stratified splitting where needed.

## Common Errors

- Passing `(n_samples, series_length, n_channels)` instead of sktime `numpy3D` `(n_samples, n_channels, series_length)`.
- Using a classifier whose tags do not support multivariate, unequal length, missing values, or `predict_proba`.
- Running feature extraction on the full dataset before CV.
- Using a plain sklearn classifier directly on `numpy3D`; wrap feature extraction with `SklearnClassifierPipeline`.
- Assuming all API-listed deep/foundation classifiers are available without soft dependencies.
- Using non-stratified CV on heavily imbalanced labels.

## References

- Read `references/sktime-classifier-map.md` for classifier families, official class names, tags, and dependency caveats.
- Read `references/sktime-data-validation.md` for accepted mtypes, padding/truncation, cross-validation, metrics, and leakage-safe pipelines.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_numpy3d.py` to validate `.npy` `X` and `.npy`/CSV labels before fitting.

## Ready Checklist

- `ts-classification-data-prep` contract is complete and leakage risks are documented.
- `X` is a valid sktime `Panel` mtype and `len(y) == n_instances`.
- Equal/variable length, multivariate channels, missing values, and class imbalance are explicitly handled.
- All transformations and feature extraction live inside the fitted/CV pipeline.
- Classifier tags match data requirements and `predict_proba` expectations.
- Evaluation uses stratified temporal/group-aware split if required by the task, and reports imbalance-aware metrics.
