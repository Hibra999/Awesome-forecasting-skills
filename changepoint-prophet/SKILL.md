---
name: changepoint-prophet
description: Use Prophet for trend changepoint analysis inside Prophet forecasting workflows after validating prepared time-series data, including ds/y pandas inputs, automatic or manual potential changepoints, changepoint prior tuning, trend deltas, forecast intervals, cross-validation, plotting significant changepoints, regressors, holidays/shocks, and leakage-safe retrospective or forecasting use.
---

# Prophet Changepoints

Use this skill after forecasting data preparation when the goal is to understand or control **trend rate changes** in a Prophet model.

Do not use Prophet as a general-purpose offline segmentation library or online minimal-delay changepoint detector. Official docs describe automatic trend changepoints inside a forecasting model.

## Minimum Install

```bash
python -m pip install prophet
```

PyPI lists `prophet 1.3.0` as the latest release, published January 27, 2026. The README also documents conda-forge installation.

## Data Contract

- Use a pandas dataframe with `ds` and `y`.
- `ds` must be parseable by pandas, preferably `YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`; timezone-aware datetimes are not supported.
- `y` is a numeric single target. Prophet is univariate with optional regressors, holidays, capacities, floors, and seasonalities.
- For logistic growth, include `cap`; include `floor` only when using a saturating minimum.
- For multiple entities/panels, fit one Prophet model per entity. Prophet is not documented as a global panel model.
- Add each extra regressor with `m.add_regressor(name)` before `fit`; the column must exist in both training and prediction dataframes, and future values must be known.

Read `references/prophet-data-workflow.md` before adapting panels, manual changepoints, shocks, or operational forecasting workflows.

## Core Pattern

```python
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

m = Prophet(changepoint_prior_scale=0.05, n_changepoints=25, changepoint_range=0.8)
m.fit(df[["ds", "y"]])
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)

fig = m.plot(forecast)
add_changepoints_to_plot(fig.gca(), m, forecast)
```

## Changepoint Controls

- `n_changepoints`: number of potential changepoints if `changepoints` is not supplied; default is 25.
- `changepoint_range`: proportion of history used for automatic potential changepoints; default is 0.8.
- `changepoint_prior_scale`: sparse-prior strength for trend flexibility; default is 0.05. Larger values allow more/larger rate changes; smaller values dampen them.
- `changepoints`: explicit dates where slope changes are allowed; all must fall within training data.
- `growth`: documented trend choices are `"linear"`, `"logistic"`, and `"flat"`.

Prophet places many possible changepoints and uses sparse regularization; do not interpret every potential changepoint as a detected process change. Inspect significant deltas.

## Training, Prediction, and Inspection

- Fit with `m.fit(train_df)`; a Prophet object can only be fit once.
- Predict with `m.predict(future_df)`; output includes `yhat`, uncertainty columns, `trend`, and components.
- Inspect potential changepoint dates in `m.changepoints`.
- Inspect fitted rate changes in `m.params["delta"]`; average across draws before thresholding when MCMC is used.
- Plot significant changepoints with `add_changepoints_to_plot(fig.gca(), m, forecast, threshold=0.01)`.
- Use `m.plot_components(forecast)` to separate trend, seasonality, holidays, and extra regressors.

## Validation and Metrics

- Use `prophet.diagnostics.cross_validation` with temporal cutoffs; never random split.
- Use `performance_metrics` and choose forecast metrics such as RMSE, MAE, MAPE, MDAPE, SMAPE, and coverage.
- Tune `changepoint_prior_scale`, `n_changepoints`, `changepoint_range`, seasonalities, holidays, and regressors only on train/validation cutoffs.
- If true changepoint labels exist, compare detected significant changepoint dates against labels with a task-specific date tolerance outside Prophet; Prophet does not document a built-in changepoint precision/recall metric.

## Uncertainty

- Prophet returns `yhat_lower` and `yhat_upper` by default.
- `interval_width` controls interval width.
- Trend uncertainty assumes future trend changes follow the historical frequency and magnitude; docs warn not to expect calibrated coverage in all cases.
- Use `mcmc_samples > 0` only when full Bayesian sampling and seasonality uncertainty are needed.

## Anti-Leakage Rules

- Split by time before outlier handling, scaling, regressor engineering, manual changepoint selection, hyperparameter tuning, or shock labeling.
- For forecasting, fit Prophet only on data available at each cutoff. Do not use full-history changepoints to justify past predictions.
- If adding regressors, future values must be known at prediction time.
- If modeling shocks as holidays/events, define event windows from information available at that cutoff.
- Keep validation/test changepoint labels out of train-time selection of `changepoint_prior_scale`, `changepoints`, and `changepoint_range`.

## Common Errors

- Treating Prophet changepoints as generic distributional segmentation.
- Using Prophet for online minimal-delay detection; this is not documented as a core capability.
- Passing timezone-aware `ds` values.
- Forgetting `cap` for logistic growth.
- Adding regressors after `fit`, omitting them from `future`, using constant history regressors, or leaking unknown future regressor values.
- Overfitting outliers as trend changes; official docs recommend removing outliers by setting affected `y` values to missing when appropriate.
- Increasing `changepoint_range` near the series end without validating forecast stability.

## References

- Read `references/prophet-data-workflow.md` for data shape, panels, regressors, shocks, and leakage.
- Read `references/prophet-api-map.md` for changepoint controls, plotting, diagnostics, and limitations.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_prophet_changepoints.py` to sanity-check CSV inputs and manual changepoint dates.

## Ready Checklist

- Data has sorted, timezone-naive `ds` and numeric `y`.
- The task is trend-changepoint analysis inside Prophet, not generic segmentation.
- `growth`, `changepoint_prior_scale`, `n_changepoints`/`changepoints`, and `changepoint_range` are documented.
- Evaluation uses temporal cutoffs or fixed retrospective periods.
- All regressors, holidays, shocks, and manual changepoints obey train-only selection rules.
