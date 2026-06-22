# Prophet Changepoint Data Workflow

Use Prophet changepoints only after the base forecasting prep contract exists: entity id if applicable, ordered `ds`, numeric `y`, frequency/sampling policy, train/validation/test periods, missing/outlier policy, regressors, holidays/events, and leakage notes.

## Input Shape

- Python API expects a pandas dataframe with columns `ds` and `y`.
- `ds` should be a timezone-naive date or timestamp parseable by pandas.
- `y` is one numeric target. Prophet supports optional additional regressors, but not multiple target columns in one model.
- Prophet sorts rows internally, but the prepared data contract should be sorted before fitting so cutoffs and diagnostics are auditable.
- Missing dates are allowed. The outlier guide says Prophet has no problem with missing data and recommends setting outlier `y` values to missing in some cases.

## Multiple Series

Prophet is not documented as a global multi-series/panel forecaster. For panels:

1. Split by entity and time.
2. Fit one Prophet object per entity.
3. Tune hyperparameters without using future periods from that entity.
4. Store changepoint outputs with entity id and timestamp.

Do not pool rows from different entities into one `ds/y` series unless the target is intentionally aggregated.

## Logistic and Flat Trends

- `growth="linear"` is the common default.
- `growth="logistic"` requires `cap` in training and future dataframes. If a floor is used, include `floor` too.
- `growth="flat"` is documented in Prophet's constructor and additional topics; use it when trend growth should be suppressed and explanatory regressors/seasonality carry the forecast.

## Regressors and Known Future Values

Extra regressors must be added with `add_regressor` before fitting and must be present in both fitting and prediction dataframes. The docs state another time series can be used as a regressor only if future values are known.

For leakage safety:

- Derive regressor transformations inside each train fold.
- Do not use validation/test `y` to build regressors.
- If future regressor values are forecasts, document that the Prophet forecast is conditional on those regressor forecasts.

## Shocks and Events

Use holidays/events for known one-off shocks or recurring events when they are part of the modeling assumption. The handling-shocks guide demonstrates COVID-style event windows and seasonality changes.

Do not define shock windows after looking at validation/test residuals unless the task is explicitly retrospective.

## Changepoint Outputs

Useful outputs:

- `m.changepoints`: potential changepoint dates selected automatically or supplied manually.
- `m.params["delta"]`: fitted trend rate changes at those potential changepoints.
- `forecast["trend"]`: fitted/predicted trend component.
- `add_changepoints_to_plot(...)`: visual overlay of changepoints whose absolute mean delta exceeds `threshold`.

Map changepoints to timestamps, not row numbers, when reporting results.
