# pyts Classifier Map

Official documentation checked against pyts 0.13.0.

## Direct Classifiers

| Family | Class | Main use | Probability notes |
| --- | --- | --- | --- |
| Distance based | `pyts.classification.KNeighborsClassifier` | KNN over fixed-length univariate series; supports sklearn metrics, DTW variants, and BOSS metric. | `predict_proba` documented. |
| Symbolic dictionary/vector-space | `pyts.classification.SAXVSM` | SAX words, class tf-idf vectors, cosine similarity. | `decision_function` documented; no `predict_proba` in API docs. |
| Symbolic dictionary/vector-space | `pyts.classification.BOSSVS` | BOSS/SFA words, class tf-idf vectors, cosine similarity. | `decision_function` documented; no `predict_proba` in API docs. |
| Shapelet | `pyts.classification.LearningShapelets` | Learns shapelets and a logistic regression decision layer. | `predict_proba` documented. |
| Interval forest | `pyts.classification.TimeSeriesForest` | Random windows with mean/std/slope features plus random forest. | `predict_proba` documented; exposes feature importances. |
| Bag of features | `pyts.classification.TSBF` | Random subsequences/interval features plus random forests. | `predict_proba` documented; controls tree size to manage memory. |

## Multivariate Support

- Most `pyts.classification` classes operate on univariate arrays shaped `(n_samples, n_timestamps)`.
- `pyts.multivariate.classification.MultivariateClassifier(estimator)` extends univariate classifiers to 3D arrays shaped `(n_samples, n_features, n_timestamps)` by fitting one classifier per feature and hard-voting predictions.
- `pyts.multivariate.transformation.WEASELMUSE` is a multivariate feature extractor, not a classifier. Combine it with sklearn classifiers in a pipeline.

## Feature Extractors For Classification Pipelines

`pyts.transformation` turns `(n_samples, n_timestamps)` into `(n_samples, n_extracted_features)`:

- `ShapeletTransform`
- `BagOfPatterns`
- `BOSS`
- `WEASEL`
- `ROCKET`

These are not direct classifiers unless paired with a classifier in a sklearn `Pipeline`. Fit them only on train folds.

## Preprocessing

`pyts.preprocessing` applies time-series preprocessing sample-wise:

- Missing values: `InterpolationImputer`
- Scaling: `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`
- Non-linear transforms: `PowerTransformer`, `QuantileTransformer`
- Discretization: `KBinsDiscretizer`

Use these inside sklearn pipelines when they are part of model selection or evaluation.

## Documented Limitations And Caveats

- pyts classifiers document fixed 2D univariate input; multivariate tools document fixed 3D input. Native unequal-length classifier input is not documented.
- DTW KNN metrics force brute-force neighbor search and can be slow on large datasets or long series.
- `TimeSeriesForest` and `TSBF` default to many trees; docs warn fully grown trees can be large and recommend controlling depth/leaf parameters for memory.
- Randomized methods require `random_state` for deterministic behavior.
- `SAXVSM` and `BOSSVS` expose cosine-similarity decision scores but not documented probabilities.
- `MultivariateClassifier` predicts by hard voting and does not document `predict_proba`.
- pyts does not document deep learning classifiers, HIVE-COTE-style ensembles, or native early-classification estimators.
