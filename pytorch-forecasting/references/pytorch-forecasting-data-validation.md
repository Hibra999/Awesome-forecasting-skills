# PyTorch Forecasting Data, Validation, and Evaluation

Use this reference when turning a prepared forecasting table into `TimeSeriesDataSet` objects.

## Accepted Data Formats

The primary stable data format is a pandas `DataFrame` passed to `TimeSeriesDataSet`.

Required columns:

- `time_idx`: integer time index used to build sequences. For regular data, it should increase by `+1` per step within each series.
- `target`: one column name or a list of target column names. Targets can be continuous or categorical if the selected model/loss supports them.
- `group_ids`: one or more columns identifying a time series. For a single series, create a constant ID column.

Optional roles:

- `weight`: sample weight column.
- `static_categoricals`, `static_reals`: series-level attributes.
- `time_varying_known_categoricals`, `time_varying_known_reals`: variables known for every forecast horizon step.
- `time_varying_unknown_categoricals`, `time_varying_unknown_reals`: variables observed only up to prediction creation time; include the target here.
- `variable_groups`: related columns encoded/scaled together, for example overlapping holidays.
- `constant_fill_strategy`: constants for generated missing timesteps when `allow_missing_timesteps=True`.

The API also documents `TimeSeries` v2, `EncoderDecoderTimeSeriesDataModule`, and `TslibDataModule`. Use them only when a task explicitly targets those APIs and the local version supports them; the mainstream tutorials use `TimeSeriesDataSet`.

## Missingness and Frequency

- `allow_missing_timesteps=True` fills missing integer time steps on the fly; it does not handle `NA` values.
- Fill or remove `NA` values before creating `TimeSeriesDataSet`.
- If the raw data uses datetime stamps, map them to a regular integer `time_idx` after validating frequency with `forecasting-data-prep`.
- Preserve original timestamps outside the model for reporting, cutoffs, and plots.

## Splits and Transformations

Recommended safe pattern:

1. Split raw prepared data by timestamp cutoff.
2. Create `training = TimeSeriesDataSet(train_df, ...)`.
3. Create validation/test/inference via `TimeSeriesDataSet.from_dataset(training, full_or_future_df, min_prediction_idx=cutoff_idx, stop_randomization=True)`.
4. Convert to dataloaders with `train=True` only for training and `train=False` for validation/test.

This pattern fits target normalizers, categorical encoders, and scalers on train only, then reuses them forward. Do not create validation/test datasets from scratch unless you pass prefitted encoders/scalers explicitly and can prove they were fit on train.

`randomize_length` is training data augmentation. Keep it off for validation/test by using `stop_randomization=True`.

## Lags and Rolling Features

Use the built-in `lags={variable: [lag_steps...]}` only for past lags. Official docs state lagged variables must already appear in time-varying variables and that all series are cut by the largest lag to avoid NA values.

For manual lag features, sort by `group_ids` and `time_idx`, then use group-wise `shift(lag)`. Rolling features must be computed after a shift so the current or future target is not included.

## Future Covariates

Classify all covariates with `forecasting-data-prep` before modeling:

- Known future: calendar flags, holidays, planned prices, scheduled promotions, contractual capacity. Put in `time_varying_known_*`.
- Observed past: realized weather, actual demand drivers, lagged operations, target-derived variables. Put in `time_varying_unknown_*` or lag them explicitly.
- Static: store, SKU, geography, segment. Put in `static_*`.
- Unknown: exclude or ask the user.

If a future covariate must be forecast first, validate that upstream forecast separately and document compounded uncertainty.

## Multiple Series, Multiple Targets, and Panels

Multiple related series are first-class through `group_ids`. A panel should have unique `(group_ids, time_idx)` keys after preparation.

Multiple targets are supported by `target=[...]` and `MultiNormalizer`, but only use them with models/losses that document multiple-target support. The model comparison docs explicitly mention multiple-target support for `TemporalFusionTransformer` and `DeepAR` regression.

For unrelated series or very short panels, compare against simpler methods. The docs note deep models need long histories or related series to be useful.

## Forecast Horizon and Prediction

- `max_encoder_length`: history length.
- `min_encoder_length`: minimum history length when variable histories are allowed.
- `max_prediction_length`: maximum forecast horizon.
- `min_prediction_idx`: first decoder start index for validation/test/inference.
- `predict_mode=True`: creates one prediction sequence per series from the latest provided samples.

Use `model.predict(...)` for inference. Common modes include point predictions, quantiles, and raw outputs depending on model/loss/version. Request `return_index=True` when you need to join predictions back to series IDs and time indexes.

## Probabilistic Forecasting

Documented uncertainty options include:

- `QuantileLoss` for quantile forecasts, commonly with `TemporalFusionTransformer`.
- Distribution losses such as `NormalDistributionLoss`, `NegativeBinomialDistributionLoss`, `LogNormalDistributionLoss`, `BetaDistributionLoss`, `MultivariateNormalDistributionLoss`, `MQF2DistributionLoss`, and `ImplicitQuantileNetworkDistributionLoss`.
- `DeepAR` as a parametric probabilistic model; multivariate distribution loss can account for target correlations in a DeepVAR-like setup.

Do not promise calibrated intervals automatically. Evaluate quantile coverage or distribution calibration on held-out time windows.

## Metrics and Backtesting

Built-in multi-horizon metrics include `MAE`, `RMSE`, `MAPE`, `MASE`, `SMAPE`, `PoissonLoss`, `TweedieLoss`, `QuantileLoss`, distribution losses, `MultiLoss`, and `AggregationMetric`.

Recommended practice:

- Use MAE/RMSE for scale-dependent errors.
- Use MASE/RMSSE externally for cross-series comparison when needed.
- Use SMAPE/WAPE for relative business reporting; avoid MAPE near zero.
- Use QuantileLoss/pinball loss and empirical coverage for quantile forecasts.
- Use distribution negative log likelihood and calibration checks for distribution losses.

PyTorch Forecasting does not provide one universal rolling-origin backtesting helper in the high-level docs. For backtests, loop over temporal cutoffs and recreate the train dataset and model/tuning inside each fold.

## Plotting and Diagnostics

- Use TensorBoard logs for training/validation monitoring.
- Use `plot_prediction()` for actual-vs-predicted plots.
- Use `predict_dependency()` for partial dependency style sensitivity checks when documented for the model.
- Use TFT interpretation functions only with TFT or model-specific documented support.
- Residual diagnostics are manual: join predictions to held-out actuals, compute residuals by horizon, time, group, and covariate slices. Do not tune on final test residuals.

## Documented Limitations to Surface

- `TimeSeriesDataSet` is limited to in-memory operations; no built-in large-data rotation methods are provided.
- `allow_missing_timesteps` does not impute `NA` values.
- Not all models support covariates, multiple targets, classification, or uncertainty.
- Cold-start solely from static covariates is documented only for TFT and is high risk.
- Deep learning can underperform on one or a few short series; compare against non-neural baselines.
- v2/source classes may have different import paths and interfaces than the stable tutorial workflow.
