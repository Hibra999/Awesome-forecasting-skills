# sktime Classification Data, Validation, And Leakage Notes

Use this file when adapting outputs from `ts-classification-data-prep` to sktime, selecting data formats, padding/truncating, running CV, or debugging shape errors.

## Panel Formats

Primary `Panel` mtypes:
- `pd-multiindex`: pandas `DataFrame` with row `MultiIndex` `(instance, time)` and variables in columns. Supports multivariate panels, unequal spacing, and unequally supported series, but not different variable sets across instances.
- `numpy3D`: `np.ndarray` with shape `(n_instances, n_variables, n_timepoints)`. Supports multivariate panels but requires equal length, equal index, and the same variables for every instance.
- `df-list`: list of pandas `DataFrame`s. List index is instance; rows are time points; columns are variables. Supports multivariate panels, unequal spacing/support, and different variable sets.

Minor mtypes listed in official docs:
- `nested_univ`: pandas `DataFrame` with `Series` in cells.
- `numpyflat`: 2D array with rows as instances and columns indexed by `(variable, time)`; conversion is one-way because dimensions can be ambiguous.
- `pd-wide`: wide pandas `DataFrame` with column multi-index `(variable, time)`.
- `pd-long`: columns `instances`, `timepoints`, `variable`, `value`.

For `numpy3D`, axis order is not Keras/PyTorch style. It is `(samples, channels, time)`, not `(samples, time, channels)`.

## Equal Length, Variable Length, And Missing Values

- Equal-length data: use `numpy3D` when every instance has the same time index and variable set.
- Variable length/unequal index: use `pd-multiindex` or `df-list`, then select classifiers tagged with `capability:unequal_length=True` or add a pipeline transformer.
- Official tutorial lists transformers that can remove unequal-length capability needs, including `PaddingTransformer` and `TruncationTransformer`.
- Missing values: use a classifier tagged with `capability:missing_values=True` or put `Imputer` in the pipeline.
- Do not compute pad/truncate length from test data unless `ts-classification-data-prep` defines a fixed external length before any split.

## Leakage-Safe Pipelines

Use sktime pipeline composition:

```python
from sktime.classification.feature_based import SummaryClassifier
from sktime.transformations.impute import Imputer

clf = Imputer() * SummaryClassifier()
```

Use sklearn classifiers after sktime feature extraction:

```python
from sklearn.ensemble import RandomForestClassifier
from sktime.transformations.summarize import SummaryTransformer

clf = SummaryTransformer() * RandomForestClassifier()
```

Pipeline fit semantics from official docs:
- During `fit`, transformers run `fit_transform`, then the classifier fits.
- During `predict`, fitted transformers run `transform`, then the classifier predicts.

This is the required pattern for scalers, imputers, padding/truncation, shapelets, ROCKET, TSFresh, summary features, and feature selectors.

## Validation And Model Selection

Use stratification by label:

```python
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

sktime classifiers are compatible with sklearn `cross_val_score`, `cross_validate`, and `GridSearchCV` for compatible mtypes. sktime also documents `TSCGridSearchCV` and `TSCOptCV`.

Use group-aware splitting if multiple samples come from the same subject/device/session. Stratification alone does not prevent subject leakage.

## Metrics

Balanced classes:
- `accuracy`
- `f1_macro`
- `precision_macro`
- `recall_macro`
- confusion matrix

Imbalanced classes:
- `balanced_accuracy`
- `f1_macro`
- per-class precision/recall
- PR-AUC or average precision for binary or one-vs-rest settings where probabilities exist
- ROC-AUC only when class/probability setup supports it; use macro/weighted one-vs-rest for multiclass as appropriate.

If `predict_proba` is needed, check `clf.get_tag("capability:predict_proba")`. Some classifiers expose labels only or use default behavior that may not be meaningful for probability metrics.

## Data Checks

In Python:

```python
from sktime.datatypes import check_raise, check_is_mtype

check_raise(X, mtype="numpy3D", scitype="Panel")
valid, msg, metadata = check_is_mtype(
    X, mtype="pd-multiindex", scitype="Panel", return_metadata=True
)
```

In this skill:

```bash
python classification-sktime/scripts/validate_numpy3d.py X.npy y.npy
```

The script only validates basic `numpy3D` shape and label alignment. Use sktime `check_raise` for full mtype semantics.

## Common Shape Decisions

- Single-channel equal-length data: reshape to `(n_samples, 1, series_length)`.
- Multichannel equal-length data: ensure channels are axis 1.
- Variable-length data: use `df-list` or `pd-multiindex`, or apply `PaddingTransformer`/`TruncationTransformer` inside CV.
- Tabular sklearn estimator: use an sktime feature transformer first, e.g. `SummaryTransformer() * RandomForestClassifier()`.

## Anti-Leakage Checklist

- Split `X`, `y`, groups before fitting any data-dependent transform.
- Put every transform inside pipeline/CV.
- Fit padding/truncation length, imputation, scaling, dictionary/shapelet/ROCKET/TSFresh extraction, PCA, and feature selection only on training folds.
- Use stratified CV for imbalanced labels.
- Keep subject/device/session groups isolated when needed.
- Tune hyperparameters on validation/CV only; report final metrics once on untouched test data.
