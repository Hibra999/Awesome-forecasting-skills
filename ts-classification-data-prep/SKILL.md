---
name: ts-classification-data-prep
description: Prepare, validate, split, and document time-series classification datasets before using sktime, tslearn, pyts, ETNA classification, dl-4-tsc, THUML Time-Series-Library, or similar classifiers. Use this skill whenever an agent must define sample labels, tensor/panel shape, channel order, padding or truncation policy, split IDs, class balance, group/time leakage risks, and preprocessing fit boundaries before classification.
---

# Time-Series Classification Data Prep

Use this skill before any time-series classification skill. The output should be a classification data contract, leakage-safe splits, shape diagnostics, and accepted preprocessing decisions. Do not start classifier selection until the checklist passes or the remaining risks are explicitly accepted.

## Default Workflow

1. **Define the sample**
   - State what one sample represents: full entity history, fixed window, event-centered window, sensor trial, patient visit, or benchmark instance.
   - Identify label source, label timestamp or observation window, class names, and whether labels are binary, multiclass, multilabel, or ordinal.
   - For windowed samples, record window start/end, prediction point, horizon if any, stride, overlap, and whether samples from one entity can cross splits.

2. **Validate arrays or panels**
   - Require sample-major data: first dimension or row group is `n_samples`.
   - Document shape and axis order, for example `(n_samples, n_timestamps)`, `(n_samples, n_timestamps, n_channels)`, or `(n_samples, n_channels, n_timestamps)`.
   - Keep channel names, units, sampling rate, timestamp semantics, and variable-length policy explicit.
   - Validate that `X`, `y`, sample IDs, group IDs, and split IDs have the same sample count.

3. **Normalize time and length**
   - Sort each sample by time before tensor conversion.
   - Use one sampling policy per dataset: original unequal length, resampled, padded, truncated, interpolated, or masked.
   - Choose padding/truncation length from training data or external metadata only. Do not derive it from validation/test samples.
   - Keep missing values distinct from padding values when the target library supports masks or NaN padding.

4. **Create leakage-safe splits**
   - Prefer predefined benchmark train/test splits when reproducing official datasets.
   - Otherwise split by subject, device, entity, site, or chronological cutoff when related samples or future information could leak.
   - Use stratification only within leakage-safe groups or time blocks.
   - Keep the final test split untouched until model selection is complete.

5. **Fit preprocessing only on train**
   - Fit scaling, imputation, resampling parameters, feature extraction, shapelet discovery, vocabulary/binning, PCA, selection, and class weighting on train only.
   - Refit preprocessing inside each cross-validation fold.
   - Apply fitted transforms forward to validation/test without peeking at held-out labels or sample lengths.

6. **Check labels and class balance**
   - Count labels globally and per split.
   - Ensure train contains every class that the model is expected to predict.
   - Flag classes missing from validation/test as evaluation limitations rather than silently dropping them.
   - For time-dependent labels, confirm label creation used only information available at or before the prediction point.

## Deterministic Audit Script

For `.npy` feature arrays and text/CSV/NPY labels, run:

```bash
python ts-classification-data-prep/scripts/validate_classification_contract.py X.npy y.csv \
  --splits splits.csv
```

The script validates sample count alignment, feature dimensionality, label counts, optional split counts, and class coverage in train/holdout splits. It uses only the Python standard library.

Run its built-in self-check with:

```bash
python ts-classification-data-prep/scripts/validate_classification_contract.py --demo
```

## References

- Read `references/data-contract.md` when sample definition, shape, channel order, IDs, or padding policy is unclear.
- Read `references/validation-and-leakage.md` when designing train/validation/test splits, grouped CV, temporal blocking, or fold-local preprocessing.
- Read `references/library-adapters.md` when adapting the prepared contract to sktime, tslearn, pyts, ETNA classification, dl-4-tsc, or THUML Time-Series-Library.

## Ready for Classification Checklist

- One sample, one label, and prediction timing are explicit.
- `X`, `y`, IDs, groups, and splits align by sample count and order.
- Shape, axis order, channel names, sampling policy, and missing/padding semantics are documented.
- Split policy prevents entity, subject, device, site, and future-target leakage.
- Class counts are reported globally and per split.
- Preprocessing and feature extraction are fit on train only or inside each fold.
- Benchmark splits, if used, are preserved exactly unless a deviation is documented.
- Evaluation metrics match the task and class balance.
