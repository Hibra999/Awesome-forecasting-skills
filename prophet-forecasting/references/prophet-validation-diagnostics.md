# Prophet Validation and Diagnostics

Load this reference when evaluating Prophet models, tuning parameters, handling outliers, or explaining uncertainty.

## Temporal Validation

Use Prophet's simulated historical forecasts or explicit temporal holdout. Never random split.

Python:

```python
from prophet.diagnostics import cross_validation, performance_metrics

df_cv = cross_validation(
    m,
    initial="730 days",
    period="180 days",
    horizon="365 days",
)
df_metrics = performance_metrics(df_cv)
```

Notes:

- In Python, `initial`, `period`, and `horizon` strings use pandas Timedelta units of days or shorter.
- Explicit `cutoffs` can be passed when business cutoffs, reporting gaps, or custom folds matter.
- Default initial window is three times the horizon and default cutoff spacing is half the horizon.
- Rebuild fold-specific regressors and future data from each cutoff. Do not reuse future values not available at the cutoff.

## Metrics

Official `performance_metrics` reports MSE, RMSE, MAE, MAPE, MDAPE, sMAPE, and coverage for prediction intervals.

Add external metrics when needed:

- MASE/RMSSE for comparing many series with different scales.
- WAPE for aggregate demand planning.
- Bias/mean error for systematic over- or under-forecasting.
- Pinball loss only if evaluating quantiles derived from Prophet intervals/samples with a documented method.

## Uncertainty

- Prophet returns `yhat_lower` and `yhat_upper` by default.
- `interval_width` controls interval width.
- With `mcmc_samples=0`, intervals mainly reflect trend uncertainty and observation noise from the MAP estimate.
- Use `mcmc_samples > 0` for full Bayesian sampling and seasonality uncertainty; expect materially slower fitting.
- `uncertainty_samples=0` disables uncertainty estimation and can speed prediction.
- `predictive_samples(future)` returns raw posterior predictive samples in Python.
- Do not claim calibrated coverage without validating coverage on temporal backtests.

## Plotting and Inspection

- `m.plot(forecast)`: observed history, forecast, and uncertainty.
- `m.plot_components(forecast)`: trend, seasonalities, holidays, and extra regressors.
- `plot_plotly` and `plot_components_plotly`: interactive plots; require plotly dependencies.
- `add_changepoints_to_plot(fig.gca(), m, forecast)`: overlay detected changepoints.
- `plot_cross_validation_metric(df_cv, metric="mape")`: visualize metric by horizon.
- `m.preprocess(df)` and `m.calculate_initial_params()` help inspect transformed Python model inputs in versions that include them.

## Outliers and Shocks

- Prophet can fit through missing `y`, so known outlier targets can be set to missing after documenting the rule.
- Large outliers may inflate future uncertainty because the trend uncertainty model can treat them as possible future trend changes.
- For shocks that should not repeat, model them as one-off holidays/events without future dates.
- For lasting behavior changes, inspect trend changepoints and consider conditional seasonalities or adjusted changepoints with temporal validation.

## Tuning Parameters

Tune with cross-validation, not in-sample plots:

- `changepoint_prior_scale`: trend flexibility; larger values fit more trend changes.
- `seasonality_prior_scale`: seasonality flexibility.
- `holidays_prior_scale`: holiday/event flexibility.
- Fourier orders for built-in or custom seasonalities.
- `seasonality_mode`: additive vs multiplicative.
- `changepoint_range` or explicit `changepoints` when recent changes need special treatment.

## Operational Notes

- A fitted Prophet model can only be fit once; refit a new model when new data arrives.
- Warm-starting is documented for small updates, but can be worse when many new observations or large changes occur.
- Python models should be saved with `prophet.serialize.model_to_json` and loaded with `model_from_json`, not pickle.
- Official docs state there are no plans to further develop the underlying Prophet model and recommend other libraries when state-of-the-art accuracy is the primary goal.
- Prophet can be robust to missing data and trend shifts, but this does not replace a documented data-preparation audit.
