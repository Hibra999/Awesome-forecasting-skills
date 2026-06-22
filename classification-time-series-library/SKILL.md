---
name: classification-time-series-library
description: Use THUML Time-Series-Library/TSLib for deep learning time-series classification after ts-classification-data-prep, including UEA .ts TRAIN/TEST files, UEAloader, variable-length padding masks, multivariate channels, task_name=classification, CrossEntropyLoss training, softmax probabilities, accuracy evaluation, official classification scripts, supported model names such as TimesNet, Transformer, Autoformer, PatchTST, iTransformer, MambaSingleLayer/MambaSL, TimeMixer, and strict anti-leakage preprocessing.
---

# Time-Series-Library Classification

Use this skill after `ts-classification-data-prep`. THUML Time-Series-Library (TSLib) is best when the user wants to run deep learning classification experiments with the repository's unified `run.py`, UEA `.ts` datasets, PyTorch models, official bash recipes, checkpoints, and reproducible benchmark-style runs.

Do not present TSLib as a general sklearn-style classifier library. It is a research codebase: training is driven by CLI scripts, native output is accuracy, and many conveniences such as sklearn `fit/predict` objects, built-in stratified CV, and calibration reports are not documented.

## Minimum Install

```bash
git clone https://github.com/thuml/Time-Series-Library.git
cd Time-Series-Library
conda create -n tslib python=3.11
conda activate tslib
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Install optional model dependencies only when needed, especially `mamba_ssm` for Mamba/MambaSL on Linux/CUDA.

## Data Contract

- Require a completed `ts-classification-data-prep` contract: dataset name, TRAIN/TEST split, labels, class balance, channels/features, maximum sequence length, padding/truncation policy, missing-value policy, and leakage notes.
- Use official UEA layout: `--data UEA`, `--root_path ./dataset/<DatasetName>/`, `--model_id <DatasetName>`, with `<DatasetName>_TRAIN.ts` and `<DatasetName>_TEST.ts`.
- The loader uses `sktime.datasets.load_from_tsfile_to_dataframe(..., replace_missing_vals_with='NaN')`, converts labels to integer class codes, interpolates missing values, normalizes features, and pads/clips batches through `collate_fn`.
- Runtime tensors are `(batch_size, padded_length, feature_dim)` plus `padding_mask`; this is not sklearn/sktime `numpy3D`.
- For unequal length, TSLib sets `seq_len` to the max train/test length and uses padding masks; if dimensions inside one sample have different length, the loader subsamples long series.

Read `references/tslib-classification-data.md` before adapting data, using variable length, changing normalization, or adding custom splits.

## Model Selection

Use `--task_name classification` and an official model name. Source-supported classification models include:

`Autoformer`, `Crossformer`, `DLinear`, `ETSformer`, `FEDformer`, `FiLM`, `Informer`, `LightTS`, `MICN`, `MSGNet`, `MambaSingleLayer`, `Nonstationary_Transformer`, `PatchTST`, `Pyraformer`, `Reformer`, `SegRNN`, `TimeFilter`, `TimeMixer`, `TimesNet`, `Transformer`, and `iTransformer`.

Official bash scripts exist for a smaller set: `Autoformer`, `Crossformer`, `DLinear`, `ETSformer`, `FEDformer`, `FiLM`, `Informer`, `LightTS`, `MICN`, `MambaSL` (`--model MambaSingleLayer`), `PatchTST`, `Pyraformer`, `Reformer`, `TimesNet`, `Transformer`, and `iTransformer`.

Read `references/tslib-classification-models.md` before claiming support for models without classification scripts or models that raise `NotImplementedError`.

## Training And Prediction Pattern

```bash
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Heartbeat/ \
  --model_id Heartbeat \
  --model TimesNet \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 1 \
  --des Exp \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10
```

`Exp_Classification` trains with `RAdam`, `CrossEntropyLoss`, gradient clipping, early stopping on validation accuracy, and saves `checkpoint.pth`. Test mode uses `--is_training 0` and loads the matching checkpoint.

TSLib does not expose a documented sklearn-like `predict_proba`. Internally it computes `softmax(logits)` then `argmax`; use custom code around `Exp_Classification.test` if calibrated probabilities, ROC-AUC, or exported predictions are required.

## Evaluation

- Native metric: accuracy via `cal_accuracy`.
- For balanced classes, report accuracy plus optional external macro/weighted F1 from saved logits/predictions.
- For imbalanced classes, add external `f1_macro`, balanced accuracy, per-class recall/precision, confusion matrix, ROC-AUC/PR-AUC where probabilities are exported.
- TSLib does not document stratified cross-validation. To use stratified CV, create fold-specific UEA TRAIN/TEST directories outside the repo workflow, keep every preprocessing step inside each fold, and run one script per fold.

## Anti-Leakage Rules

- Split TRAIN/TEST before normalization, interpolation choices, augmentation, padding/truncation tuning, and any external feature engineering.
- Fit scalers/normalizers only on TRAIN for each fold. The stock `UEAloader` normalizes each loaded split independently; if strict train-fitted scaling is required, modify the loader deliberately and document it.
- Use augmentation only on TRAIN; the official loader checks `flag == "TRAIN"` before applying augmentation.
- Do not compute max sequence length, class mapping, or missing-value policies from a held-out final test set unless that metadata is part of a fixed public benchmark contract.
- Use stratified folds when you create custom CV for imbalanced classes; preserve subject/device groups if present.

## Common Errors

- Passing arrays directly instead of UEA `.ts` files and `--data UEA`.
- Mismatching `--root_path`, `--model_id`, and filenames; the loader expects `<model_id>_TRAIN.ts` or `<model_id>_TEST.ts`.
- Treating `--seq_len` like a user-chosen fixed window; classification resets it from max train/test sequence length during model build.
- Expecting native `fit`, `predict`, or `predict_proba` methods.
- Using models present in `models/` that explicitly do not support classification, such as `KANAD`, `MultiPatchFormer`, `TiDE`, `WPMixer`, or forecast-only Mamba variants.
- Reading validation/test accuracy printed during training as a leak-free benchmark without checking the experiment split design.

## References

- Read `references/tslib-classification-models.md` for supported models, script coverage, and model limitations.
- Read `references/tslib-classification-data.md` for UEA `.ts` format, tensor shapes, padding, normalization, CV, and metrics.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_uea_ts_split.py` to check required TRAIN/TEST `.ts` files before running TSLib.

## Ready Checklist

- `ts-classification-data-prep` contract is complete and leakage risks are documented.
- Dataset directory has `<DatasetName>_TRAIN.ts` and `<DatasetName>_TEST.ts`.
- Labels/classes, channel count, sequence lengths, missing values, and imbalance are checked.
- Model name is source-supported for `task_name=classification` and required dependencies are installed.
- TRAIN-only preprocessing, augmentation, and fold generation are enforced.
- Accuracy plus imbalance-aware external metrics are reported on held-out folds/test data.
