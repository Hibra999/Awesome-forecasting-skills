# Prophet Changepoint API Map

Use exact documented Python names.

## Core Classes and Functions

- `from prophet import Prophet`
- `m.fit(df)`
- `m.predict(future)`
- `m.make_future_dataframe(periods=..., freq=..., include_history=True)`
- `m.plot(forecast)`
- `m.plot_components(forecast)`
- `from prophet.plot import add_changepoints_to_plot`
- `from prophet.diagnostics import cross_validation, performance_metrics`
- `from prophet.plot import plot_cross_validation_metric`

## Constructor Parameters Relevant to Changepoints

| Parameter | Documented Use |
| --- | --- |
| `growth` | `"linear"`, `"logistic"`, or `"flat"` trend. |
| `changepoints` | Explicit dates where potential trend slope changes are allowed. |
| `n_changepoints` | Number of automatic potential changepoints; not used if `changepoints` is supplied. |
| `changepoint_range` | Proportion of history used for automatic potential changepoints. Defaults to `0.8`. |
| `changepoint_prior_scale` | Sparse-prior strength controlling trend flexibility. Defaults to `0.05`. |
| `interval_width` | Width of forecast intervals. Defaults to `0.80`. |
| `mcmc_samples` | Full Bayesian sampling count; `0` uses MAP estimation. |
| `uncertainty_samples` | Simulated draws for uncertainty intervals; `0` disables uncertainty estimation. |
| `scaling` | `"absmax"` or `"minmax"` scaling. |

## Automatic Changepoint Mechanics

Prophet specifies many potential changepoints and puts a sparse prior on rate-change magnitudes. By default it uses 25 potential changepoints, uniformly placed in the first 80% of the history. Significant changepoints are inferred from non-negligible fitted deltas, not from the potential grid alone.

## Manual Changepoints

Use:

```python
m = Prophet(changepoints=["2020-03-21", "2020-06-06"])
```

Manual changepoints constrain where slope changes are allowed. They must fall within the training data.

## Plotting Significant Changepoints

Use:

```python
from prophet.plot import add_changepoints_to_plot

fig = m.plot(forecast)
add_changepoints_to_plot(fig.gca(), m, forecast, threshold=0.01)
```

The plotting function thresholds `abs(mean(m.params["delta"], axis=0))`.

## Cross-Validation and Metrics

Use simulated historical forecasts:

```python
from prophet.diagnostics import cross_validation, performance_metrics

df_cv = cross_validation(m, initial="730 days", period="180 days", horizon="365 days")
df_p = performance_metrics(df_cv)
```

Documented metrics include `mse`, `rmse`, `mae`, `mape`, `mdape`, `smape`, and `coverage`. `plot_cross_validation_metric` supports these metrics.

## Regressors, Holidays, and Seasonality

- `m.add_regressor("name")` before fitting.
- Regressor columns must exist in training and future dataframes.
- Regressors cannot be constant throughout history.
- Holidays use a dataframe with `holiday` and `ds`, with optional `lower_window`, `upper_window`, and `prior_scale`.
- `add_country_holidays(country_name=...)` is supported for built-in holidays.

## Documented Limitations

- Prophet is a forecasting procedure, not a generic changepoint segmentation package.
- Official docs do not document online minimal-delay changepoint detection.
- Prophet models one target series per model.
- Trend uncertainty assumes future trend changes resemble historical changes; docs warn coverage may not be accurate.
- Outliers can be fit as trend changes and inflate uncertainty.
- Timezone-aware `ds` values are not supported in the Python implementation.
