---
name: prophet-forecasting
description: Use Prophet for univariate forecasting with trend, seasonality, holidays, extra regressors, uncertainty intervals, temporal cross-validation, and diagnostic plots. Trigger this skill when an agent needs to model a prepared time-series dataset with the official Python or R Prophet library, after first applying forecasting-data-prep to validate frequency, horizon, covariate availability, temporal splits, and anti-leakage safeguards.
---

# Prophet Forecasting

Use this skill after `forecasting-data-prep`. Prophet is best for one target series at a time with interpretable trend, multiple seasonalities, holidays/events, and optional known-future regressors. It is not a native panel, hierarchical, ARIMA, ETS, or neural forecasting library.

## Minimum Install

Python:

```bash
python -m pip install prophet
```

R:

```r
install.packages("prophet")
```

Python package name is `prophet`; older `fbprophet` imports are pre-v1.0.

## Data Contract

- Required columns: `ds` datestamp and numeric `y`.
- `ds` must parse as date/timestamp; Python Prophet does not support timezone-aware `ds`, so convert to one timezone and remove tz before fitting.
- One model forecasts one target. For multiple series, fit separate Prophet models per series or aggregate before modeling.
- For `growth="logistic"`, both history and future dataframes require `cap`; if using a saturating minimum, also provide `floor`.
- Every extra regressor and conditional seasonality column must be present in both fit and predict dataframes.

Run the bundled schema checker when converting prepared data:

```bash
python prophet-forecasting/scripts/prophet_contract_check.py data.csv \
  --regressors promo,price \
  --growth logistic \
  --freq D
```

## Supported Model Choices

- `Prophet(growth="linear")`: default piecewise linear trend.
- `Prophet(growth="logistic")`: saturating growth with `cap`, optional `floor`.
- `Prophet(growth="flat")`: flat trend for strong-seasonality or known-regressor/counterfactual use cases.
- Built-in yearly, weekly, and daily seasonality can be `auto`, enabled/disabled, or set to a Fourier order.
- Use `add_seasonality(name, period, fourier_order, ...)` for documented custom seasonalities.
- Use `seasonality_mode="additive"` or `"multiplicative"`; individual seasonalities/regressors can override `mode`.
- Custom trend functions beyond linear/logistic/flat require modifying Prophet source; do not invent a constructor option for them.

## Modeling Workflow

1. Prepare data with `forecasting-data-prep`; preserve `freq`, horizon, cutoffs, known-future covariates, and excluded leaky columns.
2. Rename selected time and target columns to `ds` and `y`; sort by `ds`; remove duplicate `ds` rows.
3. Split by temporal cutoff before fitting transformations. Never random split.
4. Add holidays/events only if their future dates are known or intentionally omitted for one-off shocks.
5. Add regressors only when future values are available for the prediction dataframe. Use lagged/rolling regressors only if built from past data before the cutoff.
6. Fit: `m = Prophet(...); m.add_regressor(...); m.fit(train_df)`.
7. Predict with either `m.make_future_dataframe(periods=horizon, freq=freq)` plus required future columns, or a hand-built future dataframe matching the forecast dates.
8. Evaluate with temporal holdout or Prophet `cross_validation`; use test only once for final reporting.

## Python Pattern

```python
from prophet import Prophet

train = train.rename(columns={time_col: "ds", target_col: "y"})
future = future.rename(columns={time_col: "ds"})

m = Prophet(
    growth="linear",
    seasonality_mode="additive",
    interval_width=0.8,
)
for col in known_future_regressors:
    m.add_regressor(col)

m.fit(train[["ds", "y", *known_future_regressors]])
forecast = m.predict(future[["ds", *known_future_regressors]])
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
```

Use `make_future_dataframe(periods=horizon, freq=freq)` only when its generated dates exactly match the required horizon and valid timestamps. For monthly data, forecast monthly with a pandas offset such as `MS`, not daily.

## Validation and Diagnostics

- Prophet cross-validation: `from prophet.diagnostics import cross_validation, performance_metrics`.
- Set `initial`, `period`, and `horizon` as pandas Timedelta strings in Python, or pass explicit `cutoffs`.
- Official metrics include MSE, RMSE, MAE, MAPE, MDAPE, sMAPE, and interval coverage; add MASE/WAPE externally when scale comparison or demand planning needs it.
- Plots: `m.plot(forecast)`, `m.plot_components(forecast)`, `plot_plotly`, `plot_components_plotly`, and `add_changepoints_to_plot`.
- Inspect uncertainty: default intervals cover trend uncertainty and observation noise; use `mcmc_samples > 0` for full Bayesian sampling and seasonality uncertainty when justified by runtime.
- For debugging Python preprocessing, use `m.preprocess(df)` and `m.calculate_initial_params()` when available in the installed version.

## Anti-Leakage Rules

- Never random split.
- Fit scalers, imputers, encoders, outlier thresholds, and target transforms on train only.
- Recompute lags/rolling features per cutoff using only past information.
- Do not use future regressors unless they are actually known at prediction time for every horizon step.
- Respect `freq`, horizon, valid timestamp windows, and any gap between train end and forecast start.
- During backtesting, rebuild the future dataframe and all regressors from each fold cutoff.

## Common Errors

- Passing non-`ds`/`y` column names directly to Prophet.
- Leaving timezone-aware timestamps in Python `ds`.
- Asking for daily forecasts from monthly data or for timestamps outside regular observed windows.
- Adding a regressor after fitting, omitting it from `future`, or leaving nulls in regressor columns.
- Using `growth="logistic"` without `cap` in both history and future, or with `cap <= floor`.
- Treating Prophet as native multivariate/panel forecasting; use one model per target series.
- Pickling Python models; use Prophet JSON serialization instead.

## References

- Read `references/prophet-data-and-features.md` for data schema, holidays, regressors, growth, and frequency details.
- Read `references/prophet-validation-diagnostics.md` for cross-validation, metrics, uncertainty, plotting, outliers, and limitations.
- Read `references/official-sources.md` for official documentation sources consulted.

## Ready Checklist

- `forecasting-data-prep` contract exists and all leakage risks are resolved or documented.
- Prophet dataframe has `ds`, `y`, one row per timestamp, no timezone-aware `ds`, numeric finite `y`.
- Frequency and horizon match Prophet future dates.
- Holidays/events and regressors are known for the forecast horizon or excluded.
- Validation uses temporal cutoffs/backtesting, not random split.
- Forecast output includes `yhat`, intervals, component plots, and appropriate temporal metrics.
