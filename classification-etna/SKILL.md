---
name: classification-etna
description: Use ETNA experimental time-series classification after ts-classification-data-prep, including etna[classification], TimeSeriesBinaryClassifier, TSFreshFeatureExtractor, WEASELFeatureExtractor, sklearn-compatible binary classifiers, PredictabilityAnalyzer, UCR-style 2D arrays or lists of 1D series, TSDataset segment predictability, masked cross-validation, predict/predict_proba, and anti-leakage safeguards.
---

# ETNA Classification

Use this skill after `ts-classification-data-prep`. ETNA classification is best for binary time-series classification with feature extraction plus an sklearn-compatible classifier, or for predictability analysis of ETNA `TSDataset` segments.

Treat `etna.experimental.classification` as experimental. Do not use this skill for ETNA forecasting pipelines; use `etna-forecasting` for forecasting.

## Minimum Install

```bash
pip install "etna[classification]"
```

The classification tutorial uses this extra. The installation page has a typo spelling `classiciation`, but `pyproject.toml` defines the extra as `classification`.

## Data Contract

- Require a completed `ts-classification-data-prep` contract: `X`, `y`, split IDs or fold mask, class balance, series length policy, missing-value policy, and leakage notes.
- Labels must be binary integers: `0` negative and `1` positive. Map UCR `-1/1` labels to `0/1` before fitting.
- `TimeSeriesBinaryClassifier.fit(x, y)` expects `x` as an iterable/list of 1D `numpy.ndarray` series and `y` as an array of labels.
- Equal-length univariate data can be a 2D array `(n_samples, n_timestamps)` because rows iterate as individual 1D series.
- Variable-length univariate samples are supported by the feature extractors as lists of 1D arrays.
- Multivariate or multi-channel classification is not documented in ETNA classification; do not pass 3D tensors unless writing custom extractors.
- `PredictabilityAnalyzer.get_series_from_dataset(ts)` converts each ETNA `TSDataset` segment target into one classification sample.

Read `references/etna-classification-data.md` before adapting custom data or ETNA segments.

## Supported Components

Official experimental classes:

- `TimeSeriesBinaryClassifier`: wraps a feature extractor and any classifier with sklearn interface.
- `TSFreshFeatureExtractor`: uses `tsfresh.extract_features`; supports configurable tsfresh feature settings and `fill_na_value`.
- `WEASELFeatureExtractor`: uses WEASEL-style symbolic features from pyts and supports variable lengths with train-fitted padding/window logic.
- `PredictabilityAnalyzer`: subclass for segment predictability, with pretrained model names `weasel`, `tsfresh`, and `tsfresh_min`.

ETNA does not document built-in distance classifiers, shapelet classifiers, ROCKET, deep TSC classifiers, multiclass classifiers, or native multivariate TSC classifiers in this module.

## Training Pattern

```python
import numpy as np
from etna.experimental.classification import TimeSeriesBinaryClassifier
from etna.experimental.classification.feature_extraction import WEASELFeatureExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

X_train = [np.asarray(row) for row in X_train_raw]
y_train = np.asarray(y_train_raw, dtype="int64")

feature_extractor = WEASELFeatureExtractor(
    padding_value=-10,
    word_size=4,
    n_bins=4,
)
model = LogisticRegression(max_iter=1000, class_weight="balanced")
clf = TimeSeriesBinaryClassifier(feature_extractor=feature_extractor, classifier=model)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)

mask = np.zeros(len(X_train), dtype=int)
for fold_idx, (_, valid_idx) in enumerate(StratifiedKFold(n_splits=5).split(X_train, y_train)):
    mask[valid_idx] = fold_idx
metrics = clf.masked_crossval_score(x=X_train, y=y_train, mask=mask)
```

Use `TSFreshFeatureExtractor(default_fc_parameters=MinimalFCParameters(), fill_na_value=-100)` for tsfresh features when symbolic WEASEL features are not appropriate.

## Evaluation

- `masked_crossval_score` returns per-fold `precision`, `recall`, `fscore`, and `AUC`.
- Prefer stratified fold masks instead of the tutorial's plain `KFold` when classes are imbalanced.
- Report accuracy only as secondary context. Use F1-macro, balanced accuracy, precision/recall, ROC-AUC, PR-AUC, and confusion matrix for skewed classes.
- `predict_proba` returns the positive-class probability and requires the wrapped sklearn classifier to implement `predict_proba`.
- For threshold tuning, select `threshold` on validation folds only, then lock it before test evaluation.

## Predictability Analysis

- Use `PredictabilityAnalyzer` only when the label means "forecastable enough" versus "not forecastable enough".
- Available pretrained analyzer names are `weasel`, `tsfresh`, and `tsfresh_min`.
- `download_model(model_name, dataset_freq, path)` loads a pickled analyzer from the official public bucket; load pickle files only from trusted sources.
- `analyze_predictability(ts)` returns indicators by segment; `predict_proba(series)` supports custom thresholding.

Read `references/etna-classification-model-map.md` before using pretrained analyzers or custom sklearn classifiers.

## Anti-Leakage Rules

- Split train/validation/test before fitting feature extraction, padding/window settings, tsfresh parameters, WEASEL vocabulary, chi-square feature selection, scalers, threshold tuning, or class weights.
- Build fold masks from stratified or group-aware splits; do not use random unstratified splits for imbalanced labels.
- Let `TimeSeriesBinaryClassifier.masked_crossval_score` fit feature extractors inside each fold; do not precompute features on all samples.
- For variable-length series, fit padding/window decisions on train folds only. Do not choose lengths from validation/test.
- If samples come from rolling windows over longer time series, each sample must use only past data relative to its label time.

## Common Errors

- Passing labels `-1/1` from UCR without remapping to `0/1`.
- Calling `predict_proba` with an sklearn classifier that lacks `predict_proba`.
- Passing 3D multivariate tensors; ETNA classification docs show univariate series samples.
- Precomputing tsfresh/WEASEL features before cross-validation.
- Using ETNA forecasting metrics or `Pipeline.backtest` as classifier evaluation, except for creating predictability labels in a separate leakage-safe process.
- Loading untrusted `.pickle` analyzers despite pickle execution risk.

## References

- Read `references/etna-classification-model-map.md` for official classes, extractors, metrics, and documented gaps.
- Read `references/etna-classification-data.md` for accepted data forms, fold masks, variable length, and segment predictability.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_etna_classification_data.py` to validate UCR-style TSV or fixed `.npy` train/test data before fitting.

## Ready Checklist

- `ts-classification-data-prep` contract is complete and labels are binary `0/1`.
- Data is univariate 1D-per-sample, either equal-length 2D rows or variable-length lists.
- Feature extractor choice is `TSFreshFeatureExtractor` or `WEASELFeatureExtractor`.
- Wrapped classifier implements sklearn `fit` and, when needed, `predict_proba`.
- Fold mask is stratified or group-aware; feature extraction is fit only inside train folds.
- Experimental API, multivariate limitations, and pickle trust risks are documented.
