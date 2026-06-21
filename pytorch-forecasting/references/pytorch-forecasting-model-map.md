# PyTorch Forecasting Model Map

Use this reference when choosing a PyTorch Forecasting model or checking whether a requested capability is officially documented.

## Main Documented Models

The stable docs and public `pytorch_forecasting.models` exports document these primary forecasting models:

- `Baseline`: last-known-target baseline.
- `DeepAR`: autoregressive probabilistic recurrent forecaster; with multivariate distribution loss it becomes DeepVAR-style.
- `DecoderMLP`: simple MLP on decoder inputs; useful as a low-cost baseline when covariates are available.
- `NBeats`: N-BEATS for single-target regression without covariates.
- `NBeatsKAN`: KAN variant of N-BEATS exposed in the API.
- `NHiTS`: long-horizon neural hierarchical interpolation model; documented as covariate-capable and strong for long horizons.
- `RecurrentNetwork`: LSTM/GRU recurrent model wrapper.
- `TemporalFusionTransformer`: covariate-rich multi-horizon model with quantile forecasting and interpretation support.
- `TiDEModel`: TiDE model for long-term time-series forecasting.
- `TimeXer`: time-series forecasting with exogenous variables.
- `xLSTMTime`: long-term forecasting architecture built on xLSTM variants.

Supporting classes such as `BaseModel`, `BaseModelWithCovariates`, `AutoRegressiveBaseModel`, `AutoRegressiveBaseModelWithCovariates`, `GRU`, `LSTM`, `get_rnn`, and `MultiEmbedding` are infrastructure, not standalone forecasting workflows unless a model explicitly uses them.

## API-Indexed and Version-Sensitive Classes

The API index also includes v2/source or package-wrapper classes, including:

- `DLinear` under `pytorch_forecasting.models.dlinear._dlinear_v2`.
- `Samformer` under `pytorch_forecasting.models.samformer._samformer_v2`.
- `TFT` and `TFT_pkg_v2` under temporal fusion transformer v2 modules.
- Package-wrapper classes such as `DeepAR_pkg`, `NHiTS_pkg`, `TimeXer_pkg`, and similar.

These are official API-indexed names, but they are not all shown in the high-level model comparison table or top-level package exports. Before using them, verify the installed version, import path, expected data object, and prediction interface in the local docs/source.

## Capability Guidance

- Covariates: choose `TemporalFusionTransformer`, `DeepAR`, `DecoderMLP`, `NHiTS`, `TiDEModel`, `TimeXer`, `xLSTMTime`, or `RecurrentNetwork` only if the model docs/source for the installed version support the required variable roles. `NBeats` is documented as not supporting covariates.
- Multiple targets: the docs call out that `TemporalFusionTransformer` supports multiple and heterogeneous targets, and `DeepAR` handles multiple targets for regression. Check `target=[...]`, loss, and output sizes for other models.
- Classification: the model comparison table includes classification support for some models, but this skill is for forecasting. Only use categorical targets when the selected model/loss documents it.
- Probabilistic output: `DeepAR` is parametric probabilistic; `TemporalFusionTransformer` can output quantiles through `QuantileLoss`; distribution losses provide parametric probabilistic workflows. Not every model supports uncertainty.
- Inter-series learning: use `group_ids`, static features, and covariates. The docs state only models that process covariates can learn relationships between related series.
- Cold start: the docs state only `TemporalFusionTransformer` supports predictions solely from static covariates, and that it does not work tremendously well. Treat cold-start forecasting as high risk.

## Selection Heuristics

- Always benchmark against `Baseline`.
- Use `TemporalFusionTransformer` when interpretability, rich covariates, known-future variables, or quantile forecasts matter.
- Use `DeepAR` for autoregressive probabilistic baselines, count-like outcomes with distribution losses, or DeepVAR-style correlated targets.
- Use `NHiTS` for long horizon forecasting with covariates and efficient computation.
- Use `NBeats` for univariate or single-target regression without covariates.
- Use `TimeXer` when the explicit task is exogenous-variable forecasting.
- Use `DecoderMLP` or `RecurrentNetwork` as lighter baselines before expensive architectures.

## Implementation Notes

- Prefer each model's `.from_dataset(training, ...)` constructor so dimensions, target normalization, and output parameters are inferred from `TimeSeriesDataSet`.
- Use Lightning `Trainer` for training and checkpoints.
- Use model-specific interpretation methods only where documented. TFT has the strongest documented interpretation story; do not claim universal interpretation for every model.
- If using v2/source classes, pin the package version and include an import smoke test in the user's project, not in this skill.
