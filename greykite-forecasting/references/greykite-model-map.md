# Greykite Model Map

Use this reference before selecting a Greykite model template or claiming support.

## Forecasting Models and Templates

Core model families:

- `Silverkite`: LinkedIn's flagship model for interpretable forecasts.
- `Prophet`: Facebook/Meta Prophet through Greykite's `ProphetEstimator`.
- `Auto-ARIMA`: pmdarima Auto-ARIMA through `AutoArimaEstimator`.
- Lag-based: simple baselines such as week-over-week through `LagBasedEstimator`.
- Multistage: sequential residual models through `MultistageForecastTemplate`.

`ModelTemplateEnum` names in the official API/source:

- `AUTO`
- `SILVERKITE`
- `SILVERKITE_DAILY_1_CONFIG_1`
- `SILVERKITE_DAILY_1_CONFIG_2`
- `SILVERKITE_DAILY_1_CONFIG_3`
- `SILVERKITE_DAILY_1`
- `SILVERKITE_DAILY_90`
- `SILVERKITE_WEEKLY`
- `SILVERKITE_MONTHLY`
- `SILVERKITE_HOURLY_1`
- `SILVERKITE_HOURLY_24`
- `SILVERKITE_HOURLY_168`
- `SILVERKITE_HOURLY_336`
- `SILVERKITE_EMPTY`
- `SK`
- `PROPHET`
- `AUTO_ARIMA`
- `SILVERKITE_TWO_STAGE`
- `MULTISTAGE_EMPTY`
- `LAG_BASED`
- `SILVERKITE_WOW`

`SimpleSilverkiteTemplateOptions` can define additional valid generic Silverkite templates not enumerated in `ModelTemplateEnum`.

## Estimators and Low-Level Components

Documented/source estimator classes include:

- `SimpleSilverkiteEstimator`
- `SilverkiteEstimator`
- `ProphetEstimator`
- `AutoArimaEstimator`
- `LagBasedEstimator`
- `MultistageForecastEstimator`

Notable low-level components:

- `SimpleSilverkiteForecast`
- `SilverkiteForecast`
- `ModelSummary`
- `ChangepointDetector`
- `ReconcileAdditiveForecasts`

Use the high-level `Forecaster.run_forecast_config` path unless a task explicitly needs low-level control.

## Capability Notes

- Silverkite supports growth, seasonality, holidays/events, trend changepoints, seasonality changepoints, regressors, lagged regressors, autoregression, interaction terms, MSE, quantile loss, and custom fit algorithms.
- Prophet supports growth, seasonality, holidays, regressors, trend changepoints, and prediction intervals; docs explicitly say lagged regressors are not supported with Prophet.
- Auto-ARIMA is useful as a classic baseline and supports fit/prediction intervals through the template.
- `LAG_BASED` is a baseline for past-observation forecasts, for example week-over-week or week-over-3-week median.
- Multistage templates fit multiple models sequentially on residuals and sum predictions.
- `SILVERKITE_WOW` first models long-term effects with Silverkite, then estimates residuals using week-over-week logic.
- `ReconcileAdditiveForecasts` adjusts a set of base forecasts to satisfy additivity constraints; it is post-processing, not a model that learns multiple targets directly.

## Selection Heuristics

- Start with `AUTO` for most prepared datasets.
- Use `SILVERKITE` or tuned `SILVERKITE_*` templates for fast, interpretable, business-facing forecasts.
- Use `PROPHET` when logistic growth or Prophet-specific behavior is required and speed is less important.
- Use `AUTO_ARIMA` as a classic baseline or when interpretability is less important.
- Use `LAG_BASED` to establish simple week-over-week/past-observation baselines.
- Use `SILVERKITE_TWO_STAGE` or `SILVERKITE_WOW` for granular or residual-style multistage workflows.
- Use `SK` only for advanced low-level Silverkite tuning; it is not intended as a good out-of-the-box default.
