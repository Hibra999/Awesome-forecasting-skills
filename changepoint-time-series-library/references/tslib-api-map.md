# TSLib Anomaly API Map

Use exact project entry points and arguments.

## Entry Point

`run.py` dispatches by `--task_name`. Official options include:

- `long_term_forecast`
- `short_term_forecast`
- `imputation`
- `classification`
- `anomaly_detection`
- `zero_shot_forecast`

Use `--task_name anomaly_detection` for process-change event detection.

## Required/Important Arguments

| Argument | Meaning |
| --- | --- |
| `--is_training` | `1` trains and then tests; `0` loads a checkpoint and tests. |
| `--model_id` | Experiment/model identifier, also used in setting/checkpoint names. |
| `--model` | Model class name discovered from `models/`. Prefer names with official anomaly scripts. |
| `--data` | One of official anomaly loaders: `PSM`, `MSL`, `SMAP`, `SMD`, `SWAT`. |
| `--root_path` | Dataset directory. |
| `--features` | Official anomaly scripts use `M`. |
| `--seq_len` | Sliding input/reconstruction window. |
| `--pred_len` | Official anomaly scripts use `0`. |
| `--enc_in` / `--c_out` | Number of input/output channels. |
| `--anomaly_ratio` | Prior anomaly percentage for percentile thresholding. |
| `--batch_size`, `--train_epochs`, `--learning_rate`, `--patience` | Training controls. |

## Experiment Internals

`Exp_Anomaly_Detection`:

1. Builds the model from `self.model_dict[self.args.model](self.args)`.
2. Trains an autoencoding/reconstruction objective with `nn.MSELoss()`.
3. Validates with reconstruction MSE.
4. In `test`, computes per-time-step energy as `torch.mean(MSELoss(reduce=False)(batch_x, outputs), dim=-1)`.
5. Computes threshold as `np.percentile(np.concatenate([train_energy, test_energy]), 100 - anomaly_ratio)`.
6. Builds point predictions where `test_energy > threshold`.
7. Applies `utils.tools.adjustment(gt, pred)`.
8. Reports Accuracy, Precision, Recall, and F-score.

## Official Anomaly Script Coverage

Scripts under `scripts/anomaly_detection` document practical coverage:

- `PSM`: `Autoformer`, `DLinear`, `KANAD`, `TimesNet`, `Transformer`.
- `SMAP`: `Autoformer`, `KANAD`, `TimesNet`, `Transformer`.
- `SMD`: `Autoformer`, `KANAD`, `TimesNet`, `Transformer`.
- `SWAT`: `Autoformer`, `KANAD`, `TimesNet`, `Transformer`.
- `MSL`: `Autoformer`, `Crossformer`, `DLinear`, `ETSformer`, `FEDformer`, `FiLM`, `Informer`, `KANAD`, `LightTS`, `MICN`, `Pyraformer`, `Reformer`, `TimesNet`, `Transformer`, `iTransformer`.

Because `Exp_Basic` auto-scans `models/`, additional models may import and run, but official anomaly-script support should be treated as the documented baseline.

## Limitations

- No official `changepoint_detection` task exists.
- No built-in changepoint precision/recall, Hausdorff distance, delay metric, or segmentation output is documented.
- Official thresholding uses test energy in benchmark code; this is not leakage-safe for deployment threshold selection.
- The official README notes that maintainers are not actively adding new features and recommends newer benchmarks for current progress.
- Plotting for anomaly/changepoint outputs is not documented in `Exp_Anomaly_Detection`.
