---
name: time-series-library-forecasting
description: Use THUML Time-Series-Library (TSLib) for deep-learning time-series forecasting experiments with run.py, long_term_forecast, short_term_forecast, zero_shot_forecast, ETT/custom/M4 loaders, M/S/MS feature modes, seq_len/label_len/pred_len horizons, TimeXer exogenous workflows, deterministic metrics, temporal splits, and leakage-safe validation. Trigger when an agent needs to train, evaluate, or adapt a prepared forecasting dataset with TSLib after applying forecasting-data-prep.
---

# Time-Series-Library Forecasting

Use this skill after `forecasting-data-prep`. TSLib is best for deep-learning research benchmarks, reproducing published scripts, comparing neural architectures, or adapting a prepared CSV to `run.py`. It is not a high-level forecasting API; most work is CLI-driven through repo scripts, experiment classes, and model files.

The official README says maintainers are no longer actively adding new features as of April 2026 and recommends newer benchmarks for current progress claims. Use TSLib for baseline implementations and reproducible experiments, not as proof of state of the art without additional validation.

## Minimum Install

```bash
git clone https://github.com/thuml/Time-Series-Library.git
cd Time-Series-Library
conda create -n tslib python=3.11
conda activate tslib
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Match the Torch CUDA wheel to the machine. Install `mamba_ssm` only for `Mamba.py` workflows and `uni2ts --no-deps` only for `Moirai.py`. Docker compose is documented for containerized runs.

## Data Contract

- Start from a `forecasting-data-prep` contract: sorted time, explicit target, frequency, horizon, train/validation/test cutoffs, covariate availability, and leakage notes.
- Put data under `./dataset` or pass `--root_path` and `--data_path`.
- For `--data custom`, use a CSV with `date`, feature columns, and target; TSLib reorders columns as `date + other features + target`.
- Set `--features S` for univariate-to-univariate, `--features M` for multivariate-to-multivariate, and `--features MS` for multivariate-to-single-target.
- Set `--target` for `S` or `MS`, `--freq` for time encoding, `--seq_len` for lookback, `--label_len` for decoder start tokens, and `--pred_len` for forecast horizon.
- Match `--enc_in`, `--dec_in`, and `--c_out` to the selected feature mode and number of variables.
- TSLib's documented `Dataset_Custom` uses chronological train/validation/test proportions and fits `StandardScaler` on train only. Re-check this if you modify loaders.
- TSLib does not document a generic panel ID column for arbitrary multiple independent series. Run one experiment per series, aggregate externally, or implement and validate a custom loader.

Read `references/time-series-library-data-validation.md` before using custom data, exogenous variables, multiple series, or nonstandard validation.

## Model Selection

Use model names exactly as official files/scripts expose them. Task support is implementation-specific; do not assume every model file supports every task.

- Long/short forecasting benchmark models include `Autoformer`, `Transformer`, `TimesNet`, `DLinear`, `LightTS`, `ETSformer`, `Nonstationary_Transformer`, `FEDformer`, `Pyraformer`, `Informer`, `Reformer`, `PatchTST`, `iTransformer`, `TimeMixer`, `TSMixer`, and `TimeXer`.
- Added forecasting baselines in the official model tree include `Crossformer`, `FiLM`, `FreTS`, `Koopa`, `MICN`, `MSGNet`, `Mamba`, `MambaSimple`, `MambaSingleLayer`, `MultiPatchFormer`, `PAttn`, `SCINet`, `SegRNN`, `TemporalFusionTransformer`, `TiDE`, `TimeFilter`, and `WPMixer`.
- Zero-shot Large Time Series Model workflows are documented for models such as `Chronos`, `Chronos2`, `Moirai`, `Sundial`, `TiRex`, `TimeMoE`, and `TimesFM`. The README also mentions `Toto`, but the browsed official `models/` tree did not show `Toto.py`; do not use `--model Toto` unless the checked-out source contains it.
- `KANAD` is documented as anomaly detection, not a primary forecasting model.

Read `references/time-series-library-model-map.md` before claiming model coverage, exogenous support, LTSM support, or extra dependencies.

## Forecasting Workflow

1. Prepare data with `forecasting-data-prep`; resolve gaps, frequency, target, known-future covariates, and temporal cutoffs before touching TSLib.
2. Choose `--task_name long_term_forecast`, `short_term_forecast`, or `zero_shot_forecast`.
3. Choose `--data ETTh1|ETTh2|ETTm1|ETTm2|custom|m4` for forecasting. Other loaders are for anomaly detection or classification.
4. Choose feature mode, horizon, model, and dimensional args from the prepared data contract.
5. Train with `--is_training 1`; TSLib trains, validates for early stopping, then tests.
6. Predict/evaluate an existing checkpoint with `--is_training 0`.
7. Use saved `./results/<setting>/metrics.npy`, `pred.npy`, and `true.npy`; review `./test_results/<setting>/` plots.

```bash
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/my_data/ --data_path my_series.csv \
  --model_id my_series_96_96 --model DLinear --data custom \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --train_epochs 10 --patience 3 --itr 1
```

For exogenous-variable experiments, prefer the official `scripts/exogenous_forecast/` patterns and `TimeXer`. Only pass future covariates that are known for every forecast timestamp; if a future regressor must itself be forecast, validate that upstream forecast separately.

## Validation, Metrics, and Diagnostics

- Never random split. Use TSLib's chronological train/validation/test loaders or create repeated temporal cutoffs externally for rolling-origin backtests.
- Built-in long-term tests report MAE, MSE, RMSE, MAPE, and MSPE, with optional DTW. Short-term M4 workflows use sMAPE, MAPE, MASE, and OWA.
- Add external WAPE, MASE/RMSSE, bias, or business metrics when needed. Avoid interpreting MAPE/MSPE when actuals can be zero or near zero.
- TSLib saves sample PDF plots during tests and NumPy arrays for custom plotting.
- TSLib does not document a universal residual diagnostics API. Compute validation/test residuals from `pred.npy - true.npy` and diagnose out-of-sample only.
- TSLib forecasting workflows are point-forecast oriented. No common documented probabilistic forecast or prediction-interval API is exposed across models.

## Anti-Leakage Rules

- Do not use random splits, shuffled temporal folds, or future targets in validation.
- Fit scaling, imputation, encoding, feature selection, augmentation choices, and hyperparameter tuning on train only or inside each temporal fold.
- Build lags/rolling features with only history available before each cutoff.
- Use future covariates only when known at prediction time for the full `pred_len`.
- Keep `seq_len`, `label_len`, `pred_len`, `freq`, and split borders aligned with the data contract.
- For rolling-origin validation, rebuild datasets, scalers, model selection, and checkpoints per cutoff.

## Common Errors

- Treating TSLib as a pip-installed API instead of cloning and running the repo.
- Forgetting `date`, `--target`, `--features`, or dimensional args on custom CSVs.
- Assuming arbitrary panel data is supported without a custom loader.
- Using `M` when the task is multivariate input to one target; use `MS` for that pattern.
- Passing observed-only exogenous variables as if their future values were known.
- Comparing against README leaderboards without noting the April 2026 maintenance/benchmark warning.
- Expecting intervals, residual diagnostics, or rolling backtests from a universal built-in API.

## References

- Read `references/time-series-library-model-map.md` for official model names, task caveats, and extra dependencies.
- Read `references/time-series-library-data-validation.md` for accepted data formats, horizons, exogenous variables, validation, metrics, plotting, and diagnostics.
- Read `references/official-sources.md` for official sources consulted.

## Ready Checklist

- `forecasting-data-prep` contract is complete and anti-leakage risks are documented.
- TSLib checkout, Python, Torch/CUDA, and model-specific dependencies match the selected model.
- CSV or benchmark dataset path, `date`, `target`, `freq`, `features`, and dimensions are consistent.
- Chosen model is present in the checked-out official `models/` tree and supports the requested task.
- Validation uses temporal holdout or rolling-origin reruns; no random split.
- Metrics, plots, predictions, true values, and residual checks are produced only from held-out periods.
