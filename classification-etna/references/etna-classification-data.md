# ETNA Classification Data Notes

ETNA classification works with samples as individual univariate series, not ETNA forecasting target horizons.

## Accepted Data Forms

- `x`: iterable/list of 1D `numpy.ndarray` time series.
- `y`: `numpy.ndarray` of binary labels with values `0` and `1`.
- Fixed-length univariate arrays shaped `(n_samples, n_timestamps)` are acceptable because each row is a 1D series.
- Variable-length samples can be passed as a list of 1D arrays.
- The classification tutorial loads UCR-style TSV files where the first column is the label and the remaining columns are time values.

## Unsupported Or Undocumented Forms

- Multiclass labels are rejected by `TimeSeriesBinaryClassifier`.
- Multivariate 3D tensors are not documented for ETNA classification.
- ETNA does not document native classifiers for nested pandas panel formats or sktime-style panel containers.
- `TSDataset` is used by `PredictabilityAnalyzer` to create one sample per segment, not by `TimeSeriesBinaryClassifier.fit` directly.

## Fold Masks

`masked_crossval_score(x, y, mask)` expects `mask` to contain one fold id per sample. For each fold value, ETNA uses samples where `mask != fold` as train and `mask == fold` as validation.

Use `StratifiedKFold` or a group-aware stratified splitter to build the mask. The official tutorial demonstrates `KFold`, but that does not preserve class proportions.

## Leakage Controls

- Split or create fold masks before feature extraction.
- Do not call `TSFreshFeatureExtractor.transform`, `WEASELFeatureExtractor.transform`, or external scalers on the full dataset before CV.
- For WEASEL, train folds determine SFA, vectorizer vocabulary, chi-square selected features, and padding/window behavior.
- For tsfresh, feature extraction settings are applied in `transform`; any imputation/filling strategy and downstream selection must be decided without held-out labels.
- If predictability labels come from forecasting backtests, generate those labels on a separate historical calibration period so classifier evaluation does not reuse the same target windows.

## Minimal Data Example

```python
import numpy as np

X = [np.asarray(row, dtype=float) for row in X_2d]
y = np.asarray(y_raw, dtype="int64")

if set(np.unique(y)) != {0, 1}:
    raise ValueError("ETNA TimeSeriesBinaryClassifier requires labels 0/1.")
```

For UCR labels `-1/1`:

```python
y = np.asarray(y_raw, dtype="int64")
y[y == -1] = 0
```
