# sktime Classifier Map

Use this file before selecting or claiming sktime time-series classification support. Class names below come from the official stable API reference consulted on 2026-06-22. Verify tags on the installed version with `estimator.get_tags()` or `sktime.registry.all_estimators`.

## Composition, Model Selection, And Ensembles

- `ClassifierPipeline`
- `ColumnEnsembleClassifier`
- `SklearnClassifierPipeline`
- `MultiplexClassifier`
- `TSCGridSearchCV`
- `TSCOptCV`
- `BaggingClassifier`
- `ComposableTimeSeriesForestClassifier`
- `WeightedEnsembleClassifier`

These are wrappers/meta-estimators, not always standalone algorithms. Use them to keep transformations inside CV and to tune or combine classifiers.

## Deep Learning And Foundation/Adapter Classifiers

- `ConvTimeNetClassifier`
- `ConvTranClassifierTorch`
- `CNNClassifier`
- `CNNClassifierTorch`
- `CNTCClassifier`
- `FCNClassifier`
- `GRUClassifier`
- `GRUFCNNClassifier`
- `InceptionTimeClassifier`
- `InceptionTimeClassifierTorch`
- `LSTMFCNClassifier`
- `LSTMFCNClassifierTorch`
- `MACNNClassifier`
- `MCDCNNClassifier`
- `MCDCNNClassifierTorch`
- `MLPClassifier`
- `MVTSTransformerClassifier`
- `ResNetClassifier`
- `SimpleRNNClassifier`
- `SimpleRNNClassifierTorch`
- `TapNetClassifier`
- `TapNetClassifierTorch`
- `MantisClassifier`
- `MomentFMClassifier`
- `TSPulseClassifier`

Install soft dependencies as needed. The classification tutorial shows dependency warnings when optional packages such as `torch` are absent.

## Dictionary-Based Classifiers

- `BOSSVSClassifierPyts`
- `BOSSEnsemble`
- `ContractableBOSS`
- `IndividualBOSS`
- `IndividualTDE`
- `MUSE`
- `TemporalDictionaryEnsemble`
- `WEASEL`
- `MrSEQL`
- `MrSQM`

Dictionary methods learn symbolic/word-based representations. Fit them only on training folds because representation parameters are learned from the data.

## Distance, Kernel, And Proximity Classifiers

- `ElasticEnsemble`
- `KNeighborsTimeSeriesClassifier`
- `KNeighborsTimeSeriesClassifierPyts`
- `KNeighborsTimeSeriesClassifierTslearn`
- `ProximityForest`
- `ProximityStump`
- `ProximityTree`
- `ShapeDTW`
- `TimeSeriesSVC`
- `TimeSeriesSVCTslearn`

Distance-based methods can be expensive on large datasets, especially DTW-like distances and ensembles. Check runtime, `n_jobs`, and approximation/contract options. For multivariate distances, official examples show `IndepDist` and `CombinedDistance`.

## Early Classification

- `ProbabilityThresholdEarlyClassifier`
- `TEASER`
- `BaseEarlyClassifier`

Use early classifiers only when the task requires predictions before observing the full series. Evaluate both accuracy and earliness/decision time.

## Feature-Based Classifiers

- `Catch22Classifier`
- `FreshPRINCE`
- `MatrixProfileClassifier`
- `RandomIntervalClassifier`
- `SignatureClassifier`
- `SummaryClassifier`
- `TSFreshClassifier`

These extract tabular features from series. Any feature extractor or feature selector must be fit only on train folds. Missing-value or unequal-length support depends on classifier tags or prep transformers.

## Hybrid, Interval, ROCKET, Shapelet, And Tree Utilities

Hybrid:
- `HIVECOTEV1`
- `HIVECOTEV2`

Interval/forest:
- `CanonicalIntervalForest`
- `DrCIF`
- `RandomIntervalSpectralEnsemble`
- `SupervisedTimeSeriesForest`
- `TimeSeriesForestClassifier`

ROCKET/kernel:
- `Arsenal`
- `RocketClassifier`

Shapelet:
- `ShapeletTransformClassifier`
- `ShapeletLearningClassifierPyts`
- `ShapeletLearningClassifierTslearn`

Tree/scikit helpers listed in the classification API:
- `ContinuousIntervalTree`
- `RotationForest`
- `DummyClassifier`

Base classes:
- `BaseClassifier`
- `BaseDeepClassifier`
- `BaseEarlyClassifier`

Base classes are for extension and should not be selected as concrete models.

## Tag-Guided Selection

Use:

```python
from sktime.registry import all_estimators

all_estimators("classifier", as_dataframe=True)
clf.get_tags()
clf.get_tag("capability:multivariate")
clf.get_tag("capability:unequal_length")
clf.get_tag("capability:missing_values")
clf.get_tag("capability:predict_proba")
```

Important tags:
- `capability:multivariate`
- `capability:unequal_length`
- `capability:missing_values`
- `capability:predict_proba`
- `capability:multithreading`
- `capability:contractable`
- `classifier_type`
- `X_inner_mtype`
- `y_inner_mtype`

Do not assume a classifier supports multivariate, variable-length, missing-value, or probability workflows just because it appears in the API.
