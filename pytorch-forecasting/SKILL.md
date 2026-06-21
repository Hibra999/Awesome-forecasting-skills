---
name: pytorch-forecasting
description: Use sktime/PyTorch Forecasting for neural time-series forecasting with pandas DataFrames, TimeSeriesDataSet, Lightning Trainer, TemporalFusionTransformer, DeepAR/DeepVAR, N-BEATS, N-HiTS, TiDE, TimeXer, xLSTMTime, RecurrentNetwork, DecoderMLP, Baseline, probabilistic or quantile losses, covariates, multiple series via group_ids, multiple targets, temporal validation, plotting, interpretation, and anti-leakage safeguards after applying forecasting-data-prep.
---

# PyTorch Forecasting

Use this skill after `forecasting-data-prep`. PyTorch Forecasting is best for neural, multi-horizon forecasting on pandas data when you need `TimeSeriesDataSet`, covariates, multiple related series, Lightning-based training, probabilistic/quantile outputs, TensorBoard logging, and model interpretation.

Do not use it as a first choice for one very short series or simple statistical baselines; the official docs warn that deep learning needs enough history and related series. Prefer simpler baselines first.

## Minimum Install

```bash
pip install pytorch-forecasting --extra-index-url https://download.pytorch.org/whl/cpu
pip install lightning
```

For Windows or GPU, install the matching PyTorch build first from the official PyTorch instructions. Conda install is documented as `conda install pytorch-forecasting pytorch>=2.0.0 -c pytorch -c conda-forge`. Use `pip install pytorch-forecasting[mqf2]` only when using `MQF2DistributionLoss`.

## Data Contract

- Start from a `forecasting-data-prep` contract: sorted rows, explicit target(s), integer time index, panel IDs, frequency, horizon, covariate availability, temporal cutoffs, and leakage notes.
- Use `TimeSeriesDataSet` with a pandas `DataFrame`. Required roles are `time_idx`, `target`, and `group_ids`; for a single series, create a constant group ID column.
- `time_idx` must be integer-like and increase by 1 for regular observations. If timestamps are real datetimes, map them to a documented integer index and keep the original timestamp for reporting.
- Use `target="y"` for one target or `target=[...]` for multiple targets. Multiple targets require compatible model and loss choices.
- Put calendar and planned regressors in `time_varying_known_*`; put target and observed-only signals in `time_varying_unknown_*`; put static attributes in `static_*`.
- Use `max_encoder_length` for lookback and `max_prediction_length` for forecast horizon.
- Fit encoders, scalers, target normalizers, and categorical mappings on the training `TimeSeriesDataSet`; create validation/test/inference datasets with `TimeSeriesDataSet.from_dataset(..., stop_randomization=True)`.
- `allow_missing_timesteps=True` handles missing integer time steps, not `NA` values. Fill or exclude `NA`s before creating the dataset.
- The official `TimeSeriesDataSet` is in-memory; no built-in large-data rotation exists.

Read `references/pytorch-forecasting-data-validation.md` before using custom panels, missing timesteps, lags, multiple targets, future covariates, or backtests.

## Model Selection

Use public model names exactly as documented. The main documented models are `Baseline`, `DeepAR`, `DecoderMLP`, `NBeats`, `NBeatsKAN`, `NHiTS`, `RecurrentNetwork`, `TemporalFusionTransformer`, `TiDEModel`, `TimeXer`, and `xLSTMTime`.

Use `TemporalFusionTransformer` for rich covariates, multiple related series, quantile forecasts, and interpretation. Use `DeepAR` for autoregressive probabilistic regression and DeepVAR-style multivariate distribution loss. Use `NBeats` only for single-target regression without covariates. Use `NHiTS` for long horizons and covariates. Use `TimeXer` for exogenous-variable forecasting. Use `Baseline` as a last-value benchmark.

The API index also lists v2/source classes such as `DLinear`, `Samformer`, `TFT`, and package-wrapper classes. Do not treat them as top-level stable imports unless the installed version and import path are verified. Read `references/pytorch-forecasting-model-map.md` before claiming support.

## Training Pattern

```python
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

training = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target="y",
    group_ids=["series_id"],
    max_encoder_length=36,
    max_prediction_length=6,
    static_categoricals=["series_id"],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["y"],
)
validation = TimeSeriesDataSet.from_dataset(
    training, full_df, min_prediction_idx=validation_start_idx, stop_randomization=True
)

train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=2)
val_loader = validation.to_dataloader(train=False, batch_size=128, num_workers=2)

model = TemporalFusionTransformer.from_dataset(
    training, loss=QuantileLoss(), learning_rate=0.03, hidden_size=32
)
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    callbacks=[EarlyStopping(monitor="val_loss", mode="min"), LearningRateMonitor()],
)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

Use `Tuner(trainer).lr_find(...)` when tuning the learning rate, then recreate or update the model with the chosen rate.

## Prediction, Evaluation, and Plotting

- Load checkpoints with the model class' `load_from_checkpoint(...)` when needed.
- Predict with `model.predict(data_or_loader, mode="prediction"|"quantiles"|"raw", return_index=True, return_x=True)` as needed; verify mode support for the installed version/model.
- Evaluate on temporal validation/test windows only. Use a final test once; use rolling-origin or expanding-window backtests for model selection.
- Recommended metrics: `MAE`, `RMSE`, `MASE`, `SMAPE`, `MAPE` only when actuals are safely away from zero, `QuantileLoss` for quantiles, and distribution losses for probabilistic models.
- Use `plot_prediction()` for actual-vs-predicted plots. Use `predict_dependency()` and model-specific interpretation only on validation/test or documented sensitivity scenarios.
- For residual diagnostics, compute held-out residuals from predictions and actuals. PyTorch Forecasting does not document one universal residual diagnostics API.

## Anti-Leakage Rules

- Never random split rows or series windows across time. Split by temporal cutoffs, then construct datasets.
- Fit `TimeSeriesDataSet`, normalizers, scalers, categorical encoders, target transformations, hyperparameter tuning, and feature selection on train only or inside each backtest fold.
- Use `from_dataset()` for validation/test so train-fitted transformations are reused forward.
- Use `lags=` or manual `groupby(...).shift(lag)` only with past values; never rolling windows that include the forecast timestamp or future.
- Put future covariates in `time_varying_known_*` only if their values are known for every decoder step at prediction time.
- Keep `max_encoder_length`, `max_prediction_length`, `min_prediction_idx`, gap, frequency, and panel cutoffs aligned with the data-prep contract.
- Disable validation/test randomization with `stop_randomization=True`.

## Common Errors

- Passing datetimes directly as `time_idx` instead of an integer index.
- Building validation with a new `TimeSeriesDataSet(...)` and refitting scalers/encoders on validation/test.
- Putting observed future target drivers in `time_varying_known_*`.
- Using `NBeats` for covariate or multi-target tasks it does not document.
- Expecting neural models to work well on one short series without enough history.
- Treating `allow_missing_timesteps=True` as NA imputation.
- Ignoring version/import differences for v2 or API-indexed classes.

## References

- Read `references/pytorch-forecasting-model-map.md` for supported model names, model capabilities, and API caveats.
- Read `references/pytorch-forecasting-data-validation.md` for data formats, validation, covariates, lags, probabilistic forecasts, plotting, and diagnostics.
- Read `references/official-sources.md` for official sources consulted.

## Ready Checklist

- `forecasting-data-prep` contract is complete and anti-leakage risks are resolved or documented.
- `time_idx`, `target`, `group_ids`, covariate roles, horizon, and cutoffs are explicit.
- Train dataset owns fitted encoders/scalers/normalizers; validation/test are built with `from_dataset()`.
- Model choice supports the required covariates, multiple targets, probabilistic output, and horizon.
- Validation uses temporal cutoffs/backtesting; no random split.
- Metrics, plots, quantiles/intervals, and residual diagnostics are computed on held-out periods only.
