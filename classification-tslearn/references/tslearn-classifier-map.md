# tslearn Classifier Map

Official documentation checked against tslearn 0.8.1.

## Documented Classifiers

| Family | Class | Main use | Key notes |
| --- | --- | --- | --- |
| Distance based | `tslearn.neighbors.KNeighborsTimeSeriesClassifier` | Baseline time-series KNN using time-series metrics such as DTW. | Supports `fit`, `predict`, `predict_proba`, `score`; stores training data, so serialized models can be large. |
| Kernel SVM | `tslearn.svm.TimeSeriesSVC` | SVM with default global alignment kernel (`kernel="gak"`) or sklearn SVC kernels. | Set `probability=True` before `fit`; this slows `fit` and probabilities may not match `predict`. `class_weight="balanced"` is documented. |
| Shapelets | `tslearn.shapelets.LearningShapelets` | Learns discriminative subsequences and a classifier; can produce shapelet-distance features. | Supports `fit`, `predict`, `predict_proba`, `transform`; requires Keras v3+ and backend packages. Normalize data or use `scale=True` when appropriate. |
| Neural network wrapper | `tslearn.neural_network.TimeSeriesMLPClassifier` | Simple MLP over flattened equal-size time series. | Wraps sklearn `MLPClassifier`; requires equal-sized time series and supports `predict_proba`. Not a CNN/RNN model. |
| Early classification | `tslearn.early_classification.NonMyopicEarlyClassifier` | Classify an incoming series before observing all timestamps. | Supports `predict_class_and_earliness`, `predict_proba_and_earliness`, and `early_classification_cost`; report earliness as well as class metrics. |

## Variable-Length Support

The official variable-length methods page lists these classification methods:

- `KNeighborsTimeSeriesClassifier`
- `TimeSeriesSVC`
- `LearningShapelets`

For other classifiers, convert to equal size with a train/fold-defined policy. `TimeSeriesMLPClassifier` explicitly requires equal-sized time series.

## Probability Support

- `KNeighborsTimeSeriesClassifier.predict_proba(X)` returns class probabilities.
- `TimeSeriesSVC.predict_proba(X)` is available only when `probability=True` is set before `fit`.
- `LearningShapelets.predict_proba(X)` returns class probabilities.
- `TimeSeriesMLPClassifier.predict_proba(X)` and `predict_log_proba(X)` are documented.
- `NonMyopicEarlyClassifier.predict_proba(X)` and `predict_proba_and_earliness(X)` are documented.

## Feature Extraction

`LearningShapelets.transform(X)` returns a `(n_ts, n_shapelets)` shapelet-distance feature matrix. Treat shapelet learning as supervised feature extraction and fit it only on train folds.

tslearn also documents conversion helpers such as `to_sklearn_dataset`, but external sklearn models are not tslearn classifiers. If you flatten and use sklearn classifiers, make that an explicit modeling choice outside the documented tslearn classifier list.

## Limitations To Surface

- Distance methods and GAK/DTW kernels can be expensive for large `n_ts`, long `sz`, or large CV grids.
- KNN stores training data; model files can become large.
- SVC probabilities are slower and are not guaranteed to match `predict`.
- Shapelets can have numerical issues because they are optimized by gradient descent; the guide recommends normalization.
- Shapelets require Keras v3+ and a backend.
- MLP requires equal-sized series and is a reshape wrapper around sklearn MLP, not a sequence neural network.
- tslearn docs do not document dictionary, ROCKET, HIVE-COTE, deep CNN/RNN/transformer, or ensemble classifier families.
