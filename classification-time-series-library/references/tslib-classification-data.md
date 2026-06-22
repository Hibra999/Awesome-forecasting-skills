# TSLib Classification Data, Validation, And Leakage Notes

Use this file when adapting outputs from `ts-classification-data-prep` to THUML Time-Series-Library classification.

## Expected Dataset Layout

The official classification loader is `UEAloader`, selected by:

```bash
--task_name classification --data UEA --root_path ./dataset/<DatasetName>/ --model_id <DatasetName>
```

Expected files:

```text
dataset/<DatasetName>/
  <DatasetName>_TRAIN.ts
  <DatasetName>_TEST.ts
```

The loader resolves filenames from `model_id` and `flag`, not from a `data_path` argument.

## Input Format And Tensor Shapes

Official loader path:
- Loads `.ts` files with `sktime.datasets.load_from_tsfile_to_dataframe(..., return_separate_X_and_y=True, replace_missing_vals_with='NaN')`.
- Converts labels to pandas categorical codes and stores `class_names`.
- Converts each sample to a `(seq_len, feat_dim)` frame.
- Concatenates samples into `all_df` indexed by sample ID.
- `collate_fn` returns:
  - `X`: `(batch_size, padded_length, feat_dim)`.
  - `targets`: `(batch_size, num_labels)`, used as class indices.
  - `padding_masks`: `(batch_size, padded_length)`, where 1 means real time step and 0 means padding.

This differs from common `numpy3D` conventions. TSLib model input is time-major per sample: `(batch, time, channels)`.

## Equal Length, Variable Length, Missing Values

- Equal length: all samples have the same sequence length and feature dimensions; padding mask is still passed.
- Variable length across samples: loader sets `max_seq_len` from the maximum length and `collate_fn` pads shorter samples with zeros.
- Different length across dimensions inside one sample: loader calls `subsample` on long series above a limit, then reconstructs samples.
- Missing values: loader replaces missing values with `NaN` on load and then interpolates missing values by sample/group using `interpolate_missing`.
- The model build sets `args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)`. For strict held-out testing, avoid using final test metadata to choose sequence length unless reproducing a fixed public benchmark.

## Normalization And Leakage

`UEAloader` applies:

```python
normalizer = Normalizer()
self.feature_df = normalizer.normalize(self.feature_df)
```

The default `Normalizer` computes mean/std over all rows in the loaded split. Because TRAIN and TEST are loaded separately, this is not a train-fitted transform reused on test. For strict production or custom CV:

- Fit normalization parameters on train fold only.
- Reuse train parameters on validation/test.
- Do not normalize each test fold independently if the metric must reflect deployment behavior.

Document any deliberate deviation from stock TSLib behavior.

## Augmentation

`run.py` exposes classification/detection augmentations such as jitter, scaling, permutation, magnitude warp, time warp, window slice/warp, rotation, SPAWNER, DTW warp, shapeDTW warp, WDBA, and discriminative DTW variants.

`UEAloader.__getitem__` applies augmentation only when `flag == "TRAIN"` and `augmentation_ratio > 0`. Keep it that way. Never augment validation/test before metric calculation.

## Validation And Cross-Validation

Stock TSLib uses UEA TRAIN/TEST files and, in `Exp_Classification.train`, validation and test both load `flag='TEST'`. That is benchmark-style but not a full model-selection protocol.

For stratified CV:
- Generate fold-specific directories such as `Fold01/<Dataset>_TRAIN.ts` and `Fold01/<Dataset>_TEST.ts`.
- Preserve class proportions with stratified split logic before writing `.ts` files.
- Fit all normalization/interpolation choices, augmentation setup, and model selection inside each fold.
- Keep a final untouched test split when comparing models.

For grouped data, split by subject/device/session before writing `.ts` files.

## Metrics

Native:
- Accuracy only, computed by `cal_accuracy(predictions, trues)`.

Recommended external metrics:
- Balanced classes: accuracy, macro/weighted F1, confusion matrix.
- Imbalanced classes: balanced accuracy, macro F1, per-class precision/recall, PR-AUC or ROC-AUC if logits/probabilities are exported.

To get probability-based metrics, modify `Exp_Classification.test` to save `softmax` probabilities or logits.

## Common Checks Before Running

Use this skill's script:

```bash
python classification-time-series-library/scripts/validate_uea_ts_split.py ./dataset/Heartbeat Heartbeat
```

Then verify:
- TRAIN/TEST `.ts` files exist and are non-empty.
- `@data` appears in both files.
- `@classLabel` metadata appears and labels are present.
- Class labels and counts are plausible.
- Sequence lengths and channels match the `ts-classification-data-prep` contract.
