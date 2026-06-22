# TSLib Classification Model Map

Use this file before selecting or claiming Time-Series-Library classification model support. Names come from the official `main` branch source and scripts inspected on 2026-06-22.

## Source-Supported Classification Models

These models implement a classification path in `models/*.py` and can be selected with `--task_name classification --model <name>`:

- `Autoformer`
- `Crossformer`
- `DLinear`
- `ETSformer`
- `FEDformer`
- `FiLM`
- `Informer`
- `LightTS`
- `MICN`
- `MSGNet`
- `MambaSingleLayer`
- `Nonstationary_Transformer`
- `PatchTST`
- `Pyraformer`
- `Reformer`
- `SegRNN`
- `TimeFilter`
- `TimeMixer`
- `TimesNet`
- `Transformer`
- `iTransformer`

`MambaSingleLayer` is the implementation used by the official `scripts/classification/MambaSL.sh` recipe. The script sets `model_name="MambaSingleLayer"` and uses additional MambaSL-specific arguments such as `tv_dt`, `tv_B`, `tv_C`, `use_D`, `d_conv`, `expand`, and `num_kernels`.

## Official Classification Scripts

The official `scripts/classification/` directory contains bash recipes for:

- `Autoformer`
- `Crossformer`
- `DLinear`
- `ETSformer`
- `FEDformer`
- `FiLM`
- `Informer`
- `LightTS`
- `MICN`
- `MambaSL`
- `PatchTST`
- `Pyraformer`
- `Reformer`
- `TimesNet`
- `Transformer`
- `iTransformer`

Prefer scripted models for benchmark reproduction. For source-supported models without scripts, copy an existing script and verify model-specific CLI arguments against the source.

## Models Not Documented As Classification-Supported

Do not claim classification support for a model just because it exists in `models/`.

Observed caveats in the inspected source:
- `KANAD`, `MultiPatchFormer`, `TiDE`, and `WPMixer` contain explicit `NotImplementedError` classification branches.
- `FreTS`, `Mamba`, `MambaSimple`, `Koopa`, `TSMixer`, `TimeXer`, and most zero-shot/foundation forecasting models are not documented as classification workflows in the inspected source/scripts.
- The README says `models/` are auto-detected, but auto-detection does not mean the model implements every `task_name`.

## Training, Outputs, And Metrics

`Exp_Classification` behavior:
- Builds data twice to set `seq_len`, `enc_in`, and `num_class`.
- Sets `pred_len = 0`.
- Uses `optim.RAdam`.
- Uses `nn.CrossEntropyLoss`.
- Feeds `batch_x`, `padding_mask`, and labels into `model(batch_x, padding_mask, None, None)`.
- Computes `softmax(logits)`, then `argmax`, then accuracy.
- Saves `checkpoint.pth` through early stopping.
- Writes `result_classification.txt` with accuracy.

Only accuracy is native. For F1, balanced accuracy, ROC-AUC, PR-AUC, confusion matrix, or calibration, export predictions/logits or modify `Exp_Classification.test`.

## Project-Level Caveats

- The README news from 2026-04 says maintainers are not actively adding new features and recommend newer benchmarks for current progress; baseline implementations remain correct.
- Scripts use UEA datasets and GPU-specific environment variables; adapt GPU IDs, seeds, batch sizes, and model dimensions per machine.
- Some models are memory/time heavy on long or high-dimensional datasets; MambaSL has special dependency and checkpoint notes.
- `--use_dtw` exists as a metric flag in `run.py`, but classification code uses accuracy; do not describe DTW classification metrics as native.
