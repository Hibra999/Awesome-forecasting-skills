# Merlion Forecasting Model Map

Use this reference before choosing a Merlion forecaster or promising support for exogenous variables, multivariate output, AutoML, or uncertainty.

## Main Forecasting Models

Starter/default:

- `DefaultForecaster`

Univariate models:

- `Arima`
- `Sarima`
- `ETS`
- `Prophet`
- `MSES`

Multivariate models:

- `VectorAR`
- `RandomForestForecaster`
- `ExtraTreesForecaster`
- `LGBMForecaster`

Deep learning models:

- `DeepARForecaster`
- `AutoformerForecaster`
- `ETSformerForecaster`
- `InformerForecaster`
- `TransformerForecaster`

AutoML and seasonality:

- `AutoETS`
- `AutoProphet`
- `AutoSarima`
- `SeasonalityLayer`

Ensembles:

- `ForecasterEnsemble`
- `Mean`
- `Median`
- `Max`
- `ModelSelector`
- `MetricWeightedMean`

## Factory Aliases

`ModelFactory` import aliases include `DefaultForecaster`, `Arima`, `ETS`, `MSES`, `Prophet`, `Sarima`, `VectorAR`, `RandomForestForecaster`, `ExtraTreesForecaster`, `LGBMForecaster`, `TransformerForecaster`, `InformerForecaster`, `AutoformerForecaster`, `ETSformerForecaster`, `DeepARForecaster`, `ForecasterEnsemble`, `AutoETS`, `AutoProphet`, and `AutoSarima`.

Prefer direct imports in examples and `ModelFactory` when loading saved configs or accepting user-provided model names.

## Capability Notes

- All forecasters share `train(train_data, train_config=None, exog_data=None)` and `forecast(time_stamps, time_series_prev=None, exog_data=None, return_iqr=False, return_prev=False)`.
- Forecasting docs group exogenous-regressor support around tree models, `Prophet`, `Sarima`, `VectorAR`, `Arima`, and AutoML variants `AutoSarima` and `AutoProphet`. Check `supports_exog` or the model API before passing `exog_data`.
- `forecast()` returns `(forecast, stderr)`; `stderr` can be `None`. `return_iqr=True` works only for models returning error bars.
- `ForecasterBase.require_univariate` notes that all forecasters can work on multivariate data because they can forecast a single target univariate via `target_seq_index`.
- `support_multivariate_output` indicates whether a model can forecast multivariate output. Do not assume all models output all dimensions.
- Deep-learning models require the `deep-learning` extra and are heavier than statistical defaults. Verify installed dependencies and enough data before use.
- `MSES` and some even-sampling models require `max_forecast_steps` and regular sampling or a resampling transform.
- `Prophet` has its own optional dependency behavior and can be slower or more fragile on some platforms.

## Selection Heuristics

- Start with `DefaultForecaster` or simple statistical models for a baseline.
- Use `Arima`, `Sarima`, `ETS`, `Prophet`, or `MSES` for one target series.
- Use `VectorAR` for joint multivariate dynamics.
- Use tree forecasters when lagged multivariate features and exogenous regressors are central.
- Use AutoML wrappers only inside a train-only validation plan; their selection must not see test data.
- Use `ForecasterEnsemble` with `ModelSelector` or `MetricWeightedMean` only after defining a leakage-safe validation split for combiner training.
- Use deep models only when the user has enough data and explicitly wants neural Merlion forecasters.
