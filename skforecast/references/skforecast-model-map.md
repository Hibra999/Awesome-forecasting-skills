# skforecast Forecaster and Model Map

Use this reference before selecting a forecaster or promising a capability.

## Forecaster Classes

Official forecaster types in the 0.22 docs/README:

- `ForecasterRecursive`: single-series recursive multi-step forecasting.
- `ForecasterDirect`: single-series direct multi-step forecasting.
- `ForecasterRecursiveMultiSeries`: global independent multi-series forecasting.
- `ForecasterDirectMultiVariate`: dependent multivariate forecasting.
- `ForecasterRecursiveClassifier`: autoregressive classification.
- `ForecasterRnn`: deep learning recurrent neural networks.
- `ForecasterStats`: statistical model wrapper.
- `ForecasterFoundation`: zero-shot foundation model wrapper.
- `ForecasterEquivalentDate`: equivalent-date baseline.

All forecasters share a broadly similar API, but constructor parameters and available prediction methods differ. Check the class-specific API before using probabilistic methods or special parameters.

## Estimator Families

Machine learning forecasters accept scikit-learn-compatible estimators, including common external libraries such as LightGBM, XGBoost, CatBoost, Keras wrappers, random forests, linear models, and other regressors/classifiers that implement the required sklearn API.

Statistical models documented for `ForecasterStats` include:

- skforecast stats: `Arima`, `Sarimax`, `AutoArima`, `Ets`, `AutoEts`, `Arar`.
- Compatible external statistical estimators from statsmodels, sktime, and aeon where they follow the expected interface.

Deep learning:

- `ForecasterRnn` uses Keras.
- Supported recurrent cells in the user guide: Simple RNN, LSTM, and GRU.
- `create_and_compile_model` helps build Keras models.

Foundation models:

- `FoundationModel` plus `ForecasterFoundation`.
- Official AI context/docs mention Chronos-2, TimesFM 2.5, Moirai-2, and TabICL backends.

## Capability Notes

- `ForecasterRecursive` and `ForecasterDirect` are for one target series; they can use lagged target values, window features, transformations, differentiation, and exogenous variables.
- `ForecasterRecursiveMultiSeries` trains one global model over independent series; past values from one series are not used to predict another unless exogenous/encoding features are designed to do so.
- `ForecasterDirectMultiVariate` is for dependent multivariate forecasting where one target level is forecast from multiple related series.
- `ForecasterStats` is the replacement for deprecated `ForecasterSarimax` since skforecast 0.19.0.
- `ForecasterRnn` requires `skforecast[deeplearning]` and correct Keras backend setup before importing Keras.
- `ForecasterFoundation` is for zero-shot/foundation-model workflows; verify model downloads, API keys, hardware, and package versions.
- `ForecasterEquivalentDate` is a baseline, useful before complex models.

## Probabilistic Support

Methods vary by forecaster and estimator. Official docs cover:

- Bootstrapped residual prediction intervals.
- Conformal prediction and conformal calibration.
- Quantile regression.
- Probabilistic global models.
- CRPS and calibration metrics.

Do not assume every forecaster supports every probabilistic method. Check the selected class' API for `predict_interval`, `predict_quantiles`, `predict_bootstrapping`, and similar methods.

## Selection Heuristics

- Use `ForecasterRecursive` for standard single-series ML forecasting.
- Use `ForecasterDirect` when horizon-specific models matter and the horizon is not too large.
- Use `ForecasterRecursiveMultiSeries` for many related but independent series with shared dynamics.
- Use `ForecasterDirectMultiVariate` when related series are predictors for one target.
- Use `ForecasterStats` for ARIMA/SARIMAX/ETS/ARAR baselines and native intervals.
- Use `ForecasterRnn` for large datasets where recurrent deep learning is justified.
- Use `ForecasterFoundation` for zero-shot experiments and compare against simpler baselines.
- Always include `ForecasterEquivalentDate`, naive, or seasonal naive style baselines when possible.
