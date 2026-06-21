# Time-Series-Library Data, Validation, and Evaluation

Use this reference when preparing a custom dataset or validating a TSLib forecasting run.

## Accepted Forecasting Data

Official forecasting loaders are exposed through `data_provider/data_factory.py`:

- `ETTh1`, `ETTh2`: ETT hourly loader.
- `ETTm1`, `ETTm2`: ETT minute loader.
- `custom`: general CSV loader.
- `m4`: M4 short-term forecasting loader.

Other listed loaders (`PSM`, `MSL`, `SMAP`, `SMD`, `SWAT`, `UEA`) are for anomaly detection or classification workflows, not generic forecasting.

For `Dataset_Custom`, the official loader expects columns arranged logically as:

- `date`
- one or more feature columns
- target column named by `--target`

The loader reads the CSV, removes `date` and target from the feature list, then reorders as `date + feature columns + target`. Keep the timestamp column name as `date` unless you modify and validate the loader.

## Feature Modes

- `--features S`: univariate input and univariate output. The loader uses only `--target`.
- `--features M`: multivariate input and multivariate output. The loader uses all columns after `date`.
- `--features MS`: multivariate input and single-target output. The loader uses all columns after `date`, then evaluation slices the target dimension.

Set dimensions consistently:

- `--enc_in`: number of encoder input variables.
- `--dec_in`: number of decoder input variables.
- `--c_out`: number of output variables. Use `1` for most `MS` single-target workflows unless the official script for the selected model uses a different pattern.

## Time and Horizon Args

- `--freq`: time-feature frequency (`s`, `t`, `h`, `d`, `b`, `w`, `m`) or detailed pandas-style values such as `15min` or `3h`.
- `--seq_len`: lookback/input sequence length.
- `--label_len`: decoder start-token length for encoder-decoder models.
- `--pred_len`: forecast horizon.
- `--seasonal_patterns`: M4 subset selector for short-term forecasting.
- `--inverse`: inverse-transform output data when supported by the loader.

Keep these values synchronized with the data-preparation horizon and validation windows. If the prepared holdout is shorter than `seq_len + pred_len`, the loader cannot create valid windows.

## Splits and Scaling

The documented custom loader creates chronological splits:

- train: first 70 percent
- validation: middle 10 percent, with `seq_len` history overlap for window construction
- test: final 20 percent, with `seq_len` history overlap for window construction

The official custom loader fits `StandardScaler` on the train slice, then transforms the full dataset. This is leakage-safe for scaling, but only if you do not modify the split borders or fit transforms before TSLib. The validation/test overlap contains historical context only; do not let labels from the validation/test horizon enter engineered features.

For repeated backtesting, TSLib does not document one universal rolling-origin API. Create deterministic temporal cutoffs externally, generate a dataset or custom loader per cutoff, and retrain/re-evaluate per fold.

## Exogenous Variables

The README highlights `TimeXer` and `scripts/exogenous_forecast/` for exogenous-variable forecasting. Use those scripts as the canonical pattern.

TSLib's standard custom loader does not expose a Darts-style distinction between target, past covariates, future covariates, and static covariates. With `M`/`MS`, columns are consumed as value sequences. Before using any non-target feature in a forecast horizon:

- classify it with `forecasting-data-prep` as known future, observed past, static, or unknown;
- include it in future decoder windows only if it is truly known at prediction time;
- exclude observed-only future values unless they come from a separately validated upstream forecast;
- document any custom loader changes that separate exogenous inputs from target history.

## Multiple Series and Panels

The official forecasting CSV loader does not document an arbitrary panel ID column. For multiple independent series:

- run one TSLib experiment per series and aggregate metrics externally;
- reshape only when the variables are genuinely simultaneous dimensions of one multivariate series;
- or implement a custom loader with explicit panel semantics, temporal splits per ID, train-only scaling, and leakage tests.

M4 support is benchmark-specific and should not be treated as a generic panel container.

## Training, Prediction, and Outputs

Training path:

```bash
python -u run.py --task_name long_term_forecast --is_training 1 ...
```

Inference/evaluation path:

```bash
python -u run.py --task_name long_term_forecast --is_training 0 ...
```

For long-term forecasting, TSLib saves:

- `./checkpoints/<setting>/checkpoint.pth`
- `./results/<setting>/metrics.npy`
- `./results/<setting>/pred.npy`
- `./results/<setting>/true.npy`
- sample plots under `./test_results/<setting>/`

## Metrics

Official long-term metrics from `utils/metrics.py` are:

- MAE
- MSE
- RMSE
- MAPE
- MSPE

`--use_dtw` optionally computes DTW during test, but this can be slow. Official short-term M4 evaluation uses M4 summary metrics including sMAPE, MAPE, MASE, and OWA.

Recommended additions outside TSLib:

- WAPE for interpretable aggregate relative error.
- MASE or RMSSE for cross-series comparison.
- Mean error or bias for systematic over/under-forecasting.
- Coverage and width only if an explicitly validated probabilistic/interval workflow is added outside the standard point forecast path.

## Plotting and Residuals

TSLib's test code emits periodic PDF plots by joining lookback history with true and predicted horizon values. For custom plots, load `pred.npy` and `true.npy`.

There is no documented universal residual diagnostics API. Compute residuals only from held-out validation/test predictions. Do not inspect residuals on train and then tune a final model against test.

## Limitations to Surface

- TSLib is a research codebase with CLI scripts, not a stable high-level forecasting package API.
- README notes from April 2026 say maintainers will not actively add new features and benchmark relevance may be stale.
- Arbitrary panel data, static covariates, probabilistic intervals, and rolling-origin backtesting are not documented as universal features.
- Feature availability semantics for exogenous variables are the user's responsibility unless using an official script with matching assumptions.
- Many scripts set GPU environment variables; remove or edit them to match local hardware.
