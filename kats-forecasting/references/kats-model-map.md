# Kats Forecasting Model Map

Use this reference to choose a documented Kats model. Kats documentation is partially stale: the Sphinx API lists older modules, while GitHub `main` includes additional source-only forecasting modules. Treat source-only classes as supported only after checking dependencies and tests in the target environment.

## Public API Model Modules

- `kats.models.prophet.ProphetModel`, `ProphetParams`: Prophet wrapper. `predict(steps, include_history=False, **kwargs)` returns documented columns `time`, `fcst`, `fcst_lower`, `fcst_upper`.
- `kats.models.arima.ARIMAModel`, `ARIMAParams`: statsmodels ARIMA wrapper; params include `p`, `d`, `q` and optional `exog`, `dates`, `freq`.
- `kats.models.sarima.SARIMAModel`, `SARIMAParams`: seasonal ARIMA wrapper.
- `kats.models.holtwinters.HoltWintersModel`, `HoltWintersParams`: Holt-Winters/exponential smoothing wrapper.
- `kats.models.theta.ThetaModel`, `ThetaParams`: theta model described as simple exponential smoothing with drift.
- `kats.models.stlf.STLFModel`, `STLFParams`: STL decomposition plus a forecasting method on the de-seasonalized component. Docs say methods currently include `prophet`, `linear`, `quadratic`, and `theta`.
- `kats.models.linear_model.LinearModel`, `LinearModelParams`: linear trend model.
- `kats.models.quadratic_model.QuadraticModel`, `QuadraticModelParams`: quadratic trend model with an `alpha` confidence interval parameter.
- `kats.models.harmonic_regression.HarmonicRegressionModel`, `HarmonicRegressionParams`: Fourier/harmonic seasonal regression; `predict(dates=...)` rather than `steps`.
- `kats.models.lstm.LSTMModel`, `LSTMParams`: PyTorch LSTM forecaster with `hidden_size`, `time_window`, and `num_epochs`.
- `kats.models.var.VARModel`, `VARParams`: vector autoregression for multivariate `TimeSeriesData`.
- `kats.models.bayesian_var.BayesianVAR`, `BayesianVARParams`: Bayesian VAR with Minnesota prior; docs state confidence intervals are not yet implemented.

## Source-Visible Forecasting Modules

- `kats.models.ml_ar.MLARModel`, `MLARParams`: LightGBM autoregressive/global model source. Supports target variables, multiple horizons, input windows, historical covariate windows, future covariate windows, categorical features, calendar/Fourier features, and LightGBM parameters. The API is source-documented, not in the public Sphinx model index.
- `kats.models.globalmodel`: global neural/network-style forecasting package added in v0.2.0 with an official tutorial. Use only after reading the source/tutorial for required data layout.
- `kats.models.neuralprophet.NeuralProphetModel`: source-visible wrapper; use only after checking installed dependencies and current source.
- `kats.models.simple_heuristic_model.SimpleHeuristicModel`: source-visible heuristic model; use only after checking source behavior.

## Ensembles and Reconciliation

- `MedianEnsembleModel`: takes the median forecast from individual models.
- `WeightedAvgEnsemble`: learns weights from corresponding backtesting results; do not fit weights on test data.
- `kats.models.ensemble.kats_ensemble`: broader Kats ensemble helper module.
- `TemporalHierarchicalModel`: combines base models across temporal aggregation levels and reconciles forecasts. Documented reconciliation methods include `bu`, `median`, `struc`, `svar`, `hvar`, `mint_shrink`, and `mint_sample`.

## Hyperparameter and Meta-Learning

- Many model classes implement `get_parameter_search_space()`.
- The `kats.models.metalearner` package includes metadata, hyperparameter tuning, model selection, and predictability modules.
- The README cites a self-supervised framework for hyperparameter tuning. Use these utilities only with temporal validation; do not tune on test periods.

## Capability Rules

- Univariate local forecasting: prefer Prophet, ARIMA/SARIMA, Holt-Winters, Theta, STLF, linear/quadratic, harmonic regression, or LSTM when dependencies fit.
- Multivariate endogenous forecasting: use `VARModel` or `BayesianVAR`; do not use them for unrelated panel series.
- Multiple independent series/panel: use per-series loops unless using `MLARModel` or documented global-model tutorials.
- Exogenous variables: Kats does not provide one uniform `X` API. ARIMAParams exposes optional `exog`; `MLARParams` exposes historical and future covariate window dictionaries. For other models, do not claim exogenous support unless the model docs/source show it.
- Probabilistic/interval output: many forecast dataframes include `fcst_lower` and `fcst_upper`, but interval semantics are model-specific. Bayesian VAR docs explicitly say confidence intervals are not implemented.

## Documented Limits

- Public docs are for Kats 0.0.1 while PyPI latest is 0.2.0; source and docs can disagree.
- PyPI 0.2.0 is old, alpha-classified, and declares Python 3.7/3.8 classifiers.
- Minimal install disables many functions.
- Kats does not document a universal residual diagnostics API, a universal panel data API, or a consistent exogenous-regressor interface across all forecasters.
