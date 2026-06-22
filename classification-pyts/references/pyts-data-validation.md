# pyts Data Validation And Leakage Notes

Use this reference when adapting `ts-classification-data-prep` outputs into pyts.

## Accepted Shapes

Univariate classifiers:

```text
X: (n_samples, n_timestamps)
y: (n_samples,)
```

Multivariate tools:

```text
X: (n_samples, n_features, n_timestamps)
y: (n_samples,)
```

Axis order matters: pyts multivariate data is `(samples, features, time)`.

## Fixed Length

pyts classifier APIs document numpy-like arrays with one `n_timestamps` dimension. Variable-length lists/panels are not documented as classifier input.

If raw samples have different lengths:

1. Split first.
2. Choose a fixed length from training data or an external contract.
3. Pad, truncate, interpolate, or resample each fold using only train-derived rules.
4. Document temporal distortion risks for any resampling/interpolation.

## Missing Values And Scaling

`InterpolationImputer` imputes each time series independently. pyts scalers are sample-wise, not sklearn feature-wise. This reduces some global-statistic leakage risk, but still keep these steps inside the train/CV pipeline so hyperparameters and any downstream learned features remain fold-local.

## Pipeline Pattern

```python
from pyts.preprocessing import StandardScaler
from pyts.transformation import WEASEL
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("features", WEASEL(sparse=True)),
    ("clf", LogisticRegression(solver="liblinear")),
])
pipe.fit(X_train, y_train)
```

Use sklearn `Pipeline`, `GridSearchCV`, `RandomizedSearchCV`, and `StratifiedKFold`; pyts documents sklearn compatibility for model selection and pipelines.

## Metrics

- Balanced classes: accuracy, macro F1, weighted F1, confusion matrix.
- Imbalanced classes: balanced accuracy, F1-macro, per-class precision/recall, ROC-AUC/average precision when `predict_proba` or appropriate decision scores are available.
- For `SAXVSM`/`BOSSVS`, use `decision_function` scores for ranking-style metrics only after confirming the metric accepts class decision scores for the task shape.

## Plotting

pyts examples use Matplotlib. Useful plots include raw series by class, tf-idf vectors from `SAXVSM`/`BOSSVS`, cosine similarities from `decision_function`, selected windows/feature importances from `TimeSeriesForest`/`TSBF`, and selected shapelets from shapelet-based workflows.

## Quick Checks

- `X.ndim == 2` for univariate direct classifiers.
- `X.ndim == 3` for `MultivariateClassifier` or `WEASELMUSE`.
- `X.shape[0] == len(y)`.
- No object arrays or ragged arrays.
- No transform is fit before the train/test split.

Use `scripts/validate_pyts_array.py X.npy y.npy --expected-dim 2` or `--expected-dim 3`.
