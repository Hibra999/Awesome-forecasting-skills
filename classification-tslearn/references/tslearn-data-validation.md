# tslearn Data Validation And Leakage Notes

Use this reference when adapting data from `ts-classification-data-prep` into tslearn.

## Accepted Data Shapes

- A single time series is a 2D numpy-like array `(sz, d)`, where `sz` is time and `d` is feature/channel dimension.
- A dataset is `(n_ts, sz, d)`.
- Univariate series use `d=1`.
- Labels are array-like `(n_ts,)`.

`tslearn.utils.to_time_series_dataset(dataset)` accepts shapes `(n_ts, sz, d)`, `(n_ts, sz)`, or `(sz,)` and returns `(n_ts, sz, d)`.

## Variable Length

`to_time_series_dataset` pads shorter series with trailing `NaN` to shape `(n_ts, max_sz, d)`. Official docs list `KNeighborsTimeSeriesClassifier`, `TimeSeriesSVC`, and `LearningShapelets` as classification methods for variable-length data.

If using a classifier that cannot run on variable length, use `TimeSeriesResampler` or a fixed padding/truncation policy from `ts-classification-data-prep`. Do not derive the target length from held-out test data unless the length is fixed external metadata. The docs warn that resampling introduces temporal distortions and should be used carefully.

## Text And Dataset Loading

Official options include:

- `UCR_UEA_datasets().load_dataset(name)` for standard UCR/UEA datasets, returning `X_train, y_train, X_test, y_test`.
- `load_time_series_txt` / `save_time_series_txt`; each line is one series, modalities are separated by `|`, and observations within each modality are separated by spaces.
- Conversion helpers to and from other Python time-series package formats.

## Leakage-Safe Pipeline

Fit order:

1. Split labels and instances first, using stratification or group-aware stratification.
2. Build a pipeline containing tslearn scalers/imputers/resamplers/shapelets and the classifier.
3. Run `fit` only on `X_train, y_train` or inside each CV train fold.
4. Run `predict`/`predict_proba` only on held-out validation/test folds.
5. Report metrics from out-of-fold or held-out predictions.

Recommended metrics:

- Balanced classes: accuracy, macro F1, weighted F1, confusion matrix.
- Imbalanced classes: balanced accuracy, F1-macro, per-class precision/recall, ROC-AUC or PR-AUC when probabilities are available.
- Early classification: also report prediction timestamps/earliness and `early_classification_cost`.

## Common Shape Checks

- Confirm `X.ndim == 3`.
- Confirm `X.shape[0] == len(y)`.
- Confirm channel order is `(samples, time, channels)`.
- Confirm `TimeSeriesMLPClassifier` input is equal-sized and not NaN-padded variable length.
- Confirm `TimeSeriesSVC(probability=True)` is configured before fitting when probability metrics are required.

Use `scripts/validate_tslearn_array.py X.npy y.npy` for a lightweight `.npy` header check before loading large arrays.
