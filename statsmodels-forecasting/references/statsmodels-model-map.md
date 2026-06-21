# Statsmodels Forecasting Model Map

Use this reference when selecting a documented Statsmodels forecasting API. Prefer the simplest model family that satisfies the data contract and validation plan.

## Univariate Models

- `statsmodels.tsa.ar_model.AutoReg`: autoregressive models with configurable lags, seasonal dummies, deterministic terms, and optional `exog`. Use for transparent AR baselines and lag-order selection via `ar_select_order`.
- `statsmodels.tsa.arima.model.ARIMA`: AR, MA, ARMA, ARIMA, SARIMA, and regression with ARIMA errors. Accepts `endog`, optional `exog`, `order`, `seasonal_order`, `trend`, `dates`, `freq`, and `missing`.
- `statsmodels.tsa.statespace.sarimax.SARIMAX`: state-space seasonal ARIMA with exogenous regressors. Use when needing Kalman-filter state-space features, measurement error, time-varying regression, missing-data handling, or prediction intervals from state-space results.
- `statsmodels.tsa.holtwinters.SimpleExpSmoothing`, `Holt`, `ExponentialSmoothing`: classical exponential smoothing and Holt-Winters. Supports additive and multiplicative trend/seasonal forms. Do not assume confidence intervals from the Holt-Winters implementation.
- `statsmodels.tsa.exponential_smoothing.ets.ETSModel`: ETS models with additive/multiplicative error, trend, damping, and seasonality. Use when ETS likelihood and prediction results are needed.
- `statsmodels.tsa.statespace.exponential_smoothing.ExponentialSmoothing`: linear exponential smoothing in state-space form. It can produce confidence intervals under Gaussian errors and supports state-space diagnostics; it does not cover every nonlinear/multiplicative Holt-Winters case.
- `statsmodels.tsa.statespace.structural.UnobservedComponents`: univariate structural models such as local level, trend, seasonal, cycle, and regression components.
- `statsmodels.tsa.ardl.ARDL`: autoregressive distributed lag models for one dependent series with lagged endogenous variables, lagged exogenous variables, fixed regressors, trends, and seasonality. `causal=True` excludes lag 0 of exog.
- `statsmodels.tsa.ardl.UECM`: unrestricted error correction model related to ARDL workflows.
- `statsmodels.tsa.forecasting.theta.ThetaModel`: Theta forecasting procedure for a single series.
- `statsmodels.tsa.forecasting.stl.STLForecast`: decomposes with STL, models the deseasonalized series with a supplied non-seasonal model, and combines forecasts with seasonal extrapolation.
- `statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression` and `markov_regression.MarkovRegression`: regime-switching models. Use only when regime behavior is part of the modeling assumption.

## Multivariate Endogenous Systems

- `statsmodels.tsa.vector_ar.var_model.VAR`: VAR(p) models for 2-dimensional endogenous data; includes lag-order selection and in-sample/out-of-sample prediction.
- `statsmodels.tsa.vector_ar.svar_model.SVAR`: structural VAR models for systems with structural assumptions.
- `statsmodels.tsa.vector_ar.vecm.VECM`: vector error correction models for cointegrated multivariate systems.
- `statsmodels.tsa.statespace.varmax.VARMAX`: multivariate state-space VARMA/VARMAX with optional `exog`.
- `statsmodels.tsa.statespace.dynamic_factor.DynamicFactor`: dynamic factor models for multiple observed series; supports exogenous variables.
- `statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ`: dynamic factor model with mixed-frequency support and large observed systems. Official docs note it is generally recommended for dynamic factor models, but it does not support exogenous variables.

## Supporting Tools

- Decomposition: `seasonal_decompose`, `STL`, `MSTL` help diagnostics and feature understanding. They are not standalone forecasting models except through `STLForecast`.
- Stationarity and dependence tests: `adfuller`, `kpss`, `zivot_andrews`, `acf`, `pacf`, `q_stat`, `ccf`, and cointegration tests support diagnosis and model selection.
- Order selectors: `ar_select_order`, `arma_order_select_ic`, `ardl_select_order`, and VAR `select_order` are documented helpers. Statsmodels does not document a general `auto_arima` estimator.
- Custom state space: subclass or configure `MLEModel` only when built-in state-space models cannot express the needed structure.

## Panel Guidance

Statsmodels does not provide native global panel forecasting over independent series IDs. Use one model per `series_id` when each ID is independent, or reshape to a multivariate model only when the columns are jointly endogenous variables observed at the same timestamps.

## Exogenous Regressors

- `ARIMA`, `SARIMAX`, `AutoReg`, `VAR`, `VARMAX`, `DynamicFactor`, and `ARDL` have documented exogenous-regressor support.
- Future exog must be known at prediction time and supplied for every out-of-sample step.
- Do not use contemporaneous future values unless the operational process truly knows them at the forecast cutoff.

## Intervals and Probabilistic Output

Statsmodels commonly exposes prediction intervals through prediction result objects such as `get_prediction` and `get_forecast`, especially for state-space results. Interval availability and assumptions are model-specific, so verify the chosen results class. Treat intervals as model-based uncertainty, not distribution-free guarantees.
