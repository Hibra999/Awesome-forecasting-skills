---
name: changepoint-time-series-library
description: Use THUML Time-Series-Library for process-change event detection through its documented anomaly_detection task after validating prepared multivariate time-series data, including supported PSM/MSL/SMAP/SMD/SWAT loaders, run.py scripts, reconstruction-error thresholds, anomaly_ratio, event-level adjustment, Accuracy/Precision/Recall/F-score, and leakage-safe conversion of anomaly intervals to changepoint-like starts or ends.
---

# TSLib Changepoints

Use this skill after time-series data preparation when the user wants THUML Time-Series-Library for changepoint-like process-change events.

Important limitation: official TSLib documentation and code do **not** expose a native changepoint detection task or segmentation API. Use the documented `anomaly_detection` task to detect anomalous event intervals, then derive event starts/ends as changepoint candidates outside TSLib.

## Minimum Install

```bash
git clone https://github.com/thuml/Time-Series-Library.git
cd Time-Series-Library
conda create -n tslib python=3.11
conda activate tslib
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Adjust the PyTorch/CUDA command to the local GPU. Some optional models require extra dependencies, as documented in the upstream README.

## Data Contract

- Supported anomaly loaders are `PSM`, `MSL`, `SMAP`, `SMD`, and `SWAT`.
- Use `--task_name anomaly_detection`, `--features M`, `--seq_len WINDOW`, and `--pred_len 0`.
- Training data should represent normal behavior; test data carries anomaly labels.
- `seq_len` is the sliding reconstruction window. Keep it smaller than train/validation/test lengths.
- `anomaly_ratio` is a prior anomaly percentage used to set the reconstruction-error threshold.
- TSLib returns point anomaly predictions and applies event-level adjustment when labels are available.
- For changepoints, convert contiguous predicted anomaly intervals to starts, ends, or both according to the task definition.

Read `references/tslib-data-workflow.md` before adapting custom datasets or turning anomaly intervals into changepoints.

## Core Pattern

```bash
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model TimesNet \
  --data PSM \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 25 \
  --c_out 25 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3
```

For inference from a saved checkpoint, use the same setting arguments with `--is_training 0`.

## Model Choice

`Exp_Basic` scans all files in `models/`, but official anomaly scripts are dataset-specific. Prefer models with scripts under `scripts/anomaly_detection/<DATASET>/`.

Documented anomaly scripts include:

- `PSM`: `Autoformer`, `DLinear`, `KANAD`, `TimesNet`, `Transformer`.
- `SMAP`: `Autoformer`, `KANAD`, `TimesNet`, `Transformer`.
- `SMD`: `Autoformer`, `KANAD`, `TimesNet`, `Transformer`.
- `SWAT`: `Autoformer`, `KANAD`, `TimesNet`, `Transformer`.
- `MSL`: `Autoformer`, `Crossformer`, `DLinear`, `ETSformer`, `FEDformer`, `FiLM`, `Informer`, `KANAD`, `LightTS`, `MICN`, `Pyraformer`, `Reformer`, `TimesNet`, `Transformer`, `iTransformer`.

Do not claim a model is validated for changepoint detection unless a project-specific experiment proves that anomaly intervals map to the desired process changes.

## Evaluation and Outputs

- `Exp_Anomaly_Detection` trains with MSE reconstruction loss.
- Test scoring uses per-time-step reconstruction error averaged over channels.
- Threshold is `np.percentile(concat(train_energy, test_energy), 100 - anomaly_ratio)`.
- Metrics printed and appended to `result_anomaly_detection.txt`: Accuracy, Precision, Recall, F-score.
- `utils.tools.adjustment(gt, pred)` expands a detected point inside a labeled anomaly segment to the full segment; this is event-level adjustment, not raw point precision.
- No built-in changepoint precision/recall, delay, or segmentation metric is documented.

## Anti-Leakage Rules

- Do not tune `anomaly_ratio`, `seq_len`, model, dimensions, or threshold on final test labels.
- The official threshold combines train and test energy. Treat this as benchmark behavior; for deployment, set thresholds from train/validation or prior normal data only.
- Fit normalization only on training history. Official anomaly loaders use train data for `StandardScaler`.
- Convert anomaly intervals to changepoint timestamps only after model scoring; never use ground-truth event boundaries to postprocess predictions in production.
- Validate with temporal splits or rolling deployment simulation, not random split.

## Common Errors

- Calling TSLib a native changepoint detector; it documents anomaly detection, not changepoint segmentation.
- Using unsupported `--data custom` for anomaly detection without writing a loader.
- Using `--features S` for official anomaly scripts; official examples use multivariate `--features M`.
- Mismatching `enc_in`/`c_out` with the number of channels.
- Setting `seq_len` greater than available train/test rows.
- Forgetting that event-level adjustment can inflate point metrics.
- Treating `test_label` files as available in production.

## References

- Read `references/tslib-data-workflow.md` for loader file formats, intervals, and leakage.
- Read `references/tslib-api-map.md` for `run.py`, anomaly experiment internals, scripts, metrics, and limitations.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_tslib_anomaly_inputs.py` to sanity-check dataset files, labels, dimensions, and key run arguments.

## Ready Checklist

- Task framing says anomaly-event detection or changepoint proxy, not native segmentation.
- Dataset is one of `PSM`, `MSL`, `SMAP`, `SMD`, `SWAT`, or a custom loader has been implemented.
- `seq_len`, `enc_in`, `c_out`, `anomaly_ratio`, model, and script source are documented.
- Metrics distinguish raw point predictions from event-level adjusted scores.
- Thresholding and preprocessing avoid future/test leakage for non-benchmark use.
