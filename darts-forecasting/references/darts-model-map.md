# Darts Forecasting Model Map

Use this reference to choose a documented Darts model. Verify current capabilities in the official model table and on the model object attributes such as `supports_multivariate`, `supports_past_covariates`, `supports_future_covariates`, `supports_static_covariates`, `supports_probabilistic_prediction`, `supports_sample_weight`, and `supports_transferable_series_prediction`.

## Install-Aware Selection

- `pip install darts` / `conda install u8darts`: core models only, excluding neural networks, Prophet, LightGBM, CatBoost, XGBoost, and StatsForecast.
- `darts[torch]` / `u8darts-torch`: add PyTorch/TorchForecastingModel support.
- `darts[notorch]` / `u8darts-notorch`: add Prophet, LightGBM, CatBoost, XGBoost, and StatsForecast without neural networks.
- `darts[all]` / `u8darts-all`: all available packaged model extras except documented separate dependencies.
- Separately install `neuralforecast>=3.0.0` for `NeuralForecastModel` and `tirex-ts>=1.4.0` for `TiRexModel`. Conda also documents separate `prophet>=1.1.1` for `Prophet`.

## Model Families

### Baseline local models

- `NaiveMean`
- `NaiveSeasonal`
- `NaiveDrift`
- `NaiveMovingAverage`

Use for minimum baselines, sanity checks, and seasonal/drift benchmarks.

### Statistical and classic local models

- `ARIMA`
- `VARIMA`
- `ExponentialSmoothing`
- `Theta`
- `FourTheta`
- `Prophet`
- `FFT`
- `KalmanForecaster`
- `TBATS`
- `Croston`
- `StatsForecastModel`
- `AutoARIMA`
- `AutoETS`
- `AutoCES`
- `AutoMFLES`
- `AutoTBATS`
- `AutoTheta`

Use for interpretable classical forecasting and smaller local-series workflows. Local models generally fit one target series at a time unless documented otherwise.

### Global baseline models

- `GlobalNaiveAggregate`
- `GlobalNaiveDrift`
- `GlobalNaiveSeasonal`

Use as global/multiple-series baselines.

### Regression/global models

- `SKLearnModel`
- `LinearRegressionModel`
- `RandomForestModel`
- `CatBoostModel`
- `LightGBMModel`
- `XGBModel`

Use when converting time series to supervised lagged features. These support lag specifications for target, past covariates, and future covariates, with static covariates where documented.

### Torch forecasting models

- `RNNModel` including RNN, LSTM, and GRU behavior; probabilistic version is equivalent to DeepAR.
- `BlockRNNModel`
- `NBEATSModel`
- `NHiTSModel`
- `TCNModel`
- `TransformerModel`
- `TFTModel`
- `DLinearModel`
- `NLinearModel`
- `TiDEModel`
- `TSMixerModel`
- `NeuralForecastModel`

Use for global learning, larger datasets, GPU training, validation-series training, and probabilistic likelihoods. They use `input_chunk_length`, `output_chunk_length`, and sometimes `output_chunk_shift`.

### Foundation models

- `Chronos2Model`
- `TimesFM2p5Model`
- `TiRexModel`
- `PatchTSTFMModel`

Use only after reading the model-specific docs for dependency, context length, covariate, fine-tuning, and no-training behavior. These are documented as no-training foundation models in the README model table, but examples also cover fine-tuning for some foundation/Torch workflows.

### Ensemble and conformal models

- `NaiveEnsembleModel`
- `RegressionEnsembleModel`
- `ConformalNaiveModel`
- `ConformalQRModel`

Use ensembles to combine forecasts. Use conformal models to generate calibrated quantile intervals for pre-trained global forecasting models; support depends on the wrapped forecasting model.

### Classification forecasting models

- `SKLearnClassifierModel`
- `CatBoostClassifierModel`
- `LightGBMClassifierModel`
- `XGBClassifierModel`

Use only for categorical class-label forecasting, not continuous target forecasting.

## Capability Rules

- Multivariate target: target `TimeSeries` has multiple components. Only use models documented as supporting multivariate targets.
- Multiple series/panel: pass a `Sequence[TimeSeries]`. Use global models or documented multiple-series support.
- Past covariates: use `past_covariates` for observed-past variables; Darts should not use their future values for forecasts.
- Future covariates: use `future_covariates` only when values are known or forecasted into the future for the horizon.
- Static covariates: store on `TimeSeries` and use only when the model supports static covariates.
- Probabilistic forecasts: use `num_samples > 1` when the model supports probabilistic prediction; Torch models can use likelihoods such as `QuantileRegression`, parametric likelihoods, or Monte Carlo dropout where documented.
- Sample weights: global models support sample weights according to the README; still verify model-specific support.

## Documented Limits

- Darts models consume and produce `TimeSeries`; direct pandas/NumPy model input is not the public forecasting pattern.
- `TimeSeries` has a complete sorted time index. Irregular raw data should be regularized before conversion or represented with a RangeIndex only when that is semantically correct.
- Local models do not provide global/multiple-series learning unless documented.
- Missing model dependencies are expected with modular installs.
- Model support can depend on installed optional packages and model configuration; do not infer support from a similar third-party library class.
