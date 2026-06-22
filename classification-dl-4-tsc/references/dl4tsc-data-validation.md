# dl-4-tsc Data Validation Notes

`dl-4-tsc` uses archive-style files instead of a general in-memory API. Prepare data with `ts-classification-data-prep` first, then write one of the documented layouts.

## UCRArchive_2018 TSV Layout

Expected path pattern:

```text
/dl-4-tsc/archives/UCRArchive_2018/<Dataset>/<Dataset>_TRAIN.tsv
/dl-4-tsc/archives/UCRArchive_2018/<Dataset>/<Dataset>_TEST.tsv
```

Each row is one complete time-series sample. The first column is the class label; all remaining columns are ordered time points. `utils.read_dataset()` loads the files with tab separator and returns:

- `x_train`: `(n_train, n_timestamps)`
- `y_train`: `(n_train,)`
- `x_test`: `(n_test, n_timestamps)`
- `y_test`: `(n_test,)`

Then `main.py` reshapes 2D univariate arrays to `(n_samples, n_timestamps, 1)` before model creation.

## MTS NPY Layout

Expected path pattern:

```text
/dl-4-tsc/archives/mts_archive/<Dataset>/x_train.npy
/dl-4-tsc/archives/mts_archive/<Dataset>/y_train.npy
/dl-4-tsc/archives/mts_archive/<Dataset>/x_test.npy
/dl-4-tsc/archives/mts_archive/<Dataset>/y_test.npy
```

Use array shape `(n_samples, n_timestamps, n_variables)` for `x_*` and one label per row in `y_*`. Do not use channel-first order.

## Equal Length And Conversion

- The models consume fixed-length tensors.
- The official MTS conversion helper interpolates variable-length series to a common length.
- The helper computes the target maximum length from both train and test. That is acceptable for reproducing the official benchmark pipeline, but it is not acceptable for strict custom validation.
- For strict validation, choose padding/truncation/interpolation length from train folds or fixed external metadata and apply that policy unchanged to validation/test.

## Leakage-Safe Validation

- Create train/validation/test split IDs before writing archive files.
- Use stratified splits for class imbalance and group-aware splits when samples share subject, device, patient, entity, or collection source.
- Fit label encoders, normalization, interpolation parameters, augmentation, and any class-balancing weights inside each train fold only.
- Do not select model, iteration, stopping rule, padding length, or hyperparameters from test-set metrics.
- Official code normalizes each UCR sample independently. If replacing that behavior, document the new scaler and fit it on train folds only.

## Diagnostics To Run

- Check that train and test files exist in the expected archive path.
- Check that all rows have the same number of time points.
- Check `len(y_train) == x_train.shape[0]` and `len(y_test) == x_test.shape[0]`.
- Check that validation/test labels are a subset of the train label vocabulary unless the task explicitly includes open-set classification.
- Check class counts before training; accuracy alone is not enough for skewed labels.
