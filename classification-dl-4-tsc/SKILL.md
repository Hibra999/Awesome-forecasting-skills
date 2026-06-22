---
name: classification-dl-4-tsc
description: Use hfawaz/dl-4-tsc for time-series classification after ts-classification-data-prep, including UCRArchive_2018 TSV files, MTS .npy arrays, fixed 3D tensors shaped (n_samples, n_timestamps, n_variables), TensorFlow/Keras deep classifiers, main.py experiment runs, metrics/logs, plotting artifacts, and leakage-aware evaluation.
---

# dl-4-tsc Classification

Use this skill after `ts-classification-data-prep`. `dl-4-tsc` is best for reproducing or adapting the official deep-learning time-series classification experiments from hfawaz/dl-4-tsc.

Do not use this skill for forecasting. Treat it as research code and experiment scripts, not a packaged sklearn-style library.

## Minimum Install

Official README workflow:

```bash
docker run --name dl4tsc --gpus all -idt hassanfawaz/dl-4-tsc:0.3
docker exec -it dl4tsc bash
```

Manual setup uses `pip-requirements.txt`. The repository notes that the current code uses TensorFlow 2.0, while the paper results were generated with the older TensorFlow 1.14 implementation.

## Data Contract

- Require a completed `ts-classification-data-prep` contract: labels, split IDs, class balance, channel order, sampling policy, padding/interpolation policy, and leakage notes.
- UCR format: `/dl-4-tsc/archives/UCRArchive_2018/<Dataset>/<Dataset>_TRAIN.tsv` and `_TEST.tsv`, with label in the first column.
- MTS format: `/dl-4-tsc/archives/mts_archive/<Dataset>/x_train.npy`, `y_train.npy`, `x_test.npy`, `y_test.npy`.
- Runtime model input is fixed length. Univariate UCR data starts as `(n_samples, n_timestamps)` and `main.py` reshapes it to `(n_samples, n_timestamps, 1)`.
- Multivariate arrays must be `(n_samples, n_timestamps, n_variables)`, not `(n_samples, n_variables, n_timestamps)`.
- The repo does not document native unequal-length classifier input; pad, truncate, or interpolate before training, and record the policy.

Read `references/dl4tsc-data-validation.md` before converting custom data.

## Classifier Selection

Official `main.py` classifier keys:

- `fcn`: fully convolutional network.
- `mlp`: multilayer perceptron baseline.
- `resnet`: residual network.
- `encoder`: encoder-style convolutional classifier.
- `cnn`: Time-CNN.
- `mcnn`: multi-scale CNN.
- `tlenet`: t-LeNet.
- `twiesn`: time warping invariant echo state network.
- `mcdcnn`: multi-channel deep CNN.
- `inception`: InceptionTime-style classifier included in current code.

Read `references/dl4tsc-model-map.md` before claiming probability support, multivariate behavior, or constructor signatures.

## Run Pattern

```bash
python -m main UCRArchive_2018 Coffee fcn _itr_0
```

Argument order is `archive_name dataset_name classifier_name itr`. `run_all` is documented in code for batch experiments, but agents should prefer explicit dataset/classifier runs for reproducibility.

For custom prepared data, create the expected archive directory structure first, then run `main.py`. The code uses `/dl-4-tsc/` as `root_dir`; change it deliberately if the repo is not mounted there.

## Evaluation And Outputs

- Official logging writes `history.csv`, `df_metrics.csv`, `df_best_model.csv`, saved predictions, model files, and training-loss plots under `results/`.
- Official metrics include accuracy, macro precision, macro recall, and duration.
- Add external F1-macro, balanced accuracy, confusion matrix, ROC-AUC, or average precision only when predictions/probabilities are exported correctly for the class setup.
- The repo does not document a public `predict_proba` API. Keras models use class scores internally, but expose probabilities only through custom model loading/inference code.
- Plotting utilities include training-loss plots plus ResNet filter/CAM visualization helpers.

## Anti-Leakage Rules

- Split train/validation/test before scaling, interpolation, length selection, label encoding, class weighting, augmentation, or model selection.
- Do not use random splits unless the prep contract explicitly says sample order has no temporal, subject, device, or entity dependency. Use stratified or group-aware splits when needed.
- Fit all normalization, interpolation, feature construction, and label encoders on train folds only.
- Official UCR reading standardizes each sample independently; do not replace this with full-dataset scaling.
- Official MTS conversion computes interpolation length from train plus test. For strict custom evaluation, choose length from train data or fixed external metadata.
- Official `main.py` fits one-hot labels on train plus test labels. For strict workflows, fit label mapping on train labels and treat unseen held-out labels as a data-contract error.

## Common Errors

- Using old README examples with archive name `TSC`; current constants use `UCRArchive_2018`.
- Running outside `/dl-4-tsc/` without updating the hard-coded `root_dir`.
- Passing multivariate arrays in channel-first order.
- Expecting sklearn `fit/predict/predict_proba`, stratified CV, or calibration utilities.
- Letting held-out test data choose padding length, interpolation length, class mapping, or model hyperparameters.
- Assuming all README table models cover current code; current `utils/constants.py` includes `inception` in addition to the legacy table models.

## References

- Read `references/dl4tsc-model-map.md` for supported classifier keys and documented caveats.
- Read `references/dl4tsc-data-validation.md` for accepted formats, split handling, tensor axes, and leakage controls.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_dl4tsc_arrays.py` to validate MTS/custom `.npy` train/test tensors before training.

## Ready Checklist

- `ts-classification-data-prep` contract is complete and leakage risks are documented.
- Data is in official UCR TSV or MTS NPY layout, with fixed-length samples.
- Tensor axes are `(samples, timestamps, variables)` after conversion.
- Classifier key is one of the official `main.py` keys.
- Validation is stratified or group-aware where required, and no transform was fit on held-out data.
- Metrics include imbalance-aware scores beyond accuracy when classes are skewed.
