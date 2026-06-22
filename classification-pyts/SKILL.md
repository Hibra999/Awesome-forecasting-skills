---
name: classification-pyts
description: Use pyts 0.13.0 for time-series classification after ts-classification-data-prep, including univariate 2D arrays shaped (n_samples, n_timestamps), multivariate 3D arrays shaped (n_samples, n_features, n_timestamps), KNeighborsClassifier with DTW/BOSS metrics, SAXVSM, BOSSVS, LearningShapelets, TimeSeriesForest, TSBF, MultivariateClassifier, WEASELMUSE pipelines, pyts preprocessing and transformations, fit/predict/predict_proba where documented, stratified validation, and anti-leakage safeguards.
---

# pyts Classification

Use this skill after `ts-classification-data-prep`. pyts is best when the user wants sklearn-compatible time-series classification with fixed-length arrays, symbolic/bag-of-words classifiers, shapelets, interval forests, DTW/BOSS KNN, or pyts feature extractors inside sklearn pipelines.

Do not use this skill for forecasting. A sample is a whole time series; `y` is one class label per sample.

## Minimum Install

```bash
pip install pyts
conda install -c conda-forge pyts
```

Matplotlib is only required for running gallery examples and plotting.

## Data Contract

- Require a completed `ts-classification-data-prep` contract: `X`, `y`, split IDs, class balance, channel order, padding/truncation policy, missing-value policy, and leakage notes.
- Univariate pyts classifiers use `X.shape == (n_samples, n_timestamps)`.
- Multivariate pyts tools use `X.shape == (n_samples, n_features, n_timestamps)`.
- `y` is 1D array-like with `len(y) == n_samples`.
- pyts docs do not document native unequal-length classifier input; pad, truncate, or resample to a fixed length before pyts, and record the policy.
- Read `references/pyts-data-validation.md` before adapting tensor dimensions, missing values, multivariate data, or sklearn pipelines.

## Classifier Selection

Official `pyts.classification` classes:

- `KNeighborsClassifier`: KNN baseline; supports sklearn metrics plus pyts DTW variants and BOSS metric.
- `SAXVSM`: SAX-VSM symbolic bag-of-words classifier with tf-idf class vectors.
- `BOSSVS`: Bag-of-SFA Symbols in Vector Space classifier.
- `LearningShapelets`: supervised shapelet learner with logistic-regression-style decision layer.
- `TimeSeriesForest`: extracts mean, standard deviation, and slope from random windows, then fits a random forest.
- `TSBF`: Time Series Bag-of-Features using subsequence/interval features and random forests.

Multivariate options:

- `pyts.multivariate.classification.MultivariateClassifier`: clones/fits a univariate classifier per feature and predicts by hard voting.
- `pyts.multivariate.transformation.WEASELMUSE`: multivariate WEASEL+MUSE feature extractor; combine with sklearn classifiers in a pipeline.

Read `references/pyts-classifier-map.md` before claiming probability support, multivariate behavior, or feature-extractor status.

## Training Pattern

Use sklearn-compatible fit/predict and put all preprocessing/extraction inside the pipeline:

```python
from pyts.classification import BOSSVS
from pyts.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", BOSSVS(window_size=0.3)),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

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

For feature extraction plus sklearn classifiers:

```python
from pyts.transformation import WEASEL
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

clf = make_pipeline(
    WEASEL(sparse=True),
    LogisticRegression(solver="liblinear", class_weight="balanced"),
)
clf.fit(X_train, y_train)
```

For multivariate data:

```python
from pyts.classification import BOSSVS
from pyts.multivariate.classification import MultivariateClassifier

clf = MultivariateClassifier(BOSSVS())
clf.fit(X_train_3d, y_train)
```

## Feature Extraction And Shape Handling

- Use `pyts.transformation` inside pipelines for train-only feature extraction: `ShapeletTransform`, `BagOfPatterns`, `BOSS`, `WEASEL`, and `ROCKET`.
- Use `pyts.preprocessing` for sample-wise imputation/scaling: `InterpolationImputer`, `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `PowerTransformer`, `QuantileTransformer`, and `KBinsDiscretizer`.
- For image workflows, `pyts.image` transforms series to images; pyts does not document image classifiers, so pair image features with an explicit external classifier.
- For variable-length series, enforce one fixed `n_timestamps` before pyts; do not let test data choose the length.

## Evaluation

- Use stratified splits/CV for class imbalance: `train_test_split(..., stratify=y)` and `StratifiedKFold`.
- Balanced labels: report accuracy plus macro/weighted F1.
- Imbalanced labels: prefer balanced accuracy, F1-macro, per-class precision/recall, confusion matrix, and ROC-AUC/average precision only when valid probabilities or decision scores are available.
- `predict_proba` is documented for `KNeighborsClassifier`, `LearningShapelets`, `TimeSeriesForest`, and `TSBF`; `SAXVSM`/`BOSSVS` document `decision_function`, not `predict_proba`.
- Use `GridSearchCV`/`RandomizedSearchCV` with pipelines so window sizes, SAX/SFA/BOSS/WEASEL vocabularies, shapelets, scalers, and classifiers are fit inside each train fold.

## Anti-Leakage Rules

- Split train/validation/test before fitting imputation, scaling, discretization, symbolic vocabularies, tf-idf, shapelets, ROCKET kernels, WEASEL/WEASELMUSE selection, or forests.
- Fit every transformer and classifier only on train folds through sklearn `Pipeline` or an explicit fold loop.
- Do not compute padding/truncation length, resampling size, imputation values, or class weights from held-out test data.
- Use stratified CV for imbalanced classes; use grouped stratification when the prep contract identifies subjects/devices/entities.
- If a sample is built from a time window, any rolling/lag features must use only past values within that sample definition.

## Common Errors

- Passing multivariate arrays `(n_samples, n_timestamps, n_features)` instead of pyts `(n_samples, n_features, n_timestamps)`.
- Feeding unequal-length lists directly to pyts classifiers.
- Calling `predict_proba` on `SAXVSM`, `BOSSVS`, or `MultivariateClassifier`; use documented decision scores or wrap a probabilistic external classifier.
- Running `ShapeletTransform`, `WEASEL`, `WEASELMUSE`, `BOSS`, `ROCKET`, or scaling on the full dataset before CV.
- Treating pyts sample-wise scalers as sklearn feature-wise scalers; pyts preprocessing is designed for each time series independently.
- Assuming pyts documents deep learning, HIVE-COTE, or native early-classification classifiers.

## References

- Read `references/pyts-classifier-map.md` for supported classifiers, feature extractors, probability support, and limitations.
- Read `references/pyts-data-validation.md` for 2D/3D shape contracts, fixed-length handling, CV, metrics, and pipelines.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_pyts_array.py` to validate `.npy` `X` and `.npy`/CSV labels before fitting.

## Ready Checklist

- `ts-classification-data-prep` contract is complete and leakage risks are documented.
- `X` is 2D univariate or 3D multivariate in pyts axis order, and `len(y) == n_samples`.
- Variable length, missing values, multivariate channels, and class imbalance are explicitly handled.
- Every preprocessing and feature-extraction step lives inside the fitted/CV pipeline.
- Classifier choice matches probability, decision-score, multivariate, and runtime requirements.
- Evaluation uses stratified or group-aware folds and reports imbalance-aware metrics.
