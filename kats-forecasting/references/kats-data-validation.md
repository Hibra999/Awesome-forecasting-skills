# Kats Data, Validation, and Diagnostics

Use this reference before implementing Kats data conversion, covariates, temporal validation, intervals, plotting, or diagnostics.

## TimeSeriesData

- `TimeSeriesData` is the fundamental Kats data structure.
- It can be initialized from pandas `DataFrame`, `Series`, or separate `time` and `value` objects.
- Default time column is `time`; pass `time_col_name` if the prepared dataframe uses another name.
- `value` can be a pandas `Series` for univariate data or `DataFrame` for multivariate data.
- Useful methods/properties include `time`, `value`, `to_dataframe()`, `to_array()`, `time_to_index()`, `infer_freq_robust()`, `freq_to_timedelta()`, `is_univariate()`, `is_data_missing()`, `validate_data()`, `interpolate()`, and `plot(cols=[...])`.
- `validate_data(validate_frequency=True, validate_dimension=True)` raises when frequency or dimensions are invalid.
- `interpolate(freq=..., method=..., remove_duplicate_time=...)` can fill missing timestamps/data with linear/backward/forward options. Apply it only to train data or inside each fold.

## Data Shapes

- Univariate: one target value column.
- Multivariate: multiple value columns inside one `TimeSeriesData`; use only with models that document multivariate behavior.
- Multiple series/panel: no generic public panel container is documented. Use one `TimeSeriesData` per ID, or documented list/dict inputs for global model source APIs.
- Hierarchy: `TemporalHierarchicalModel` is for temporal aggregation reconciliation, not arbitrary cross-sectional hierarchy.

## Covariates and Regressors

- Kats does not provide a unified `X`/covariate API across all forecasting models.
- `ARIMAParams` documents optional `exog`, `dates`, and `freq`.
- `MLARParams` documents `cov_history_input_windows`, `cov_future_input_windows`, `categoricals`, calendar features, and Fourier features.
- Future covariates must be known at prediction time for every horizon step. Historical covariates and lag windows must use only values available before the forecast cutoff.
- If using Prophet holidays or growth caps/floors through `ProphetParams`, construct future inputs only from known future information.

## Horizon and Frequency

- Most Kats local models use `predict(steps=..., include_history=False, **kwargs)`.
- Prophet examples pass `freq` to `predict`, e.g. monthly start frequency.
- Harmonic regression predicts for explicit future `dates`.
- Temporal hierarchical reconciliation uses `predict(steps=..., method=..., freq=...)`.
- Always keep `steps`, `freq`, seasonal periods, and validation windows aligned with the prepared data frequency.

## Backtesting and Metrics

- The homepage says Kats includes backtesting and hyperparameter tuning, and PyPI notes consolidated backtesting APIs in v0.2.0. The public Sphinx docs do not provide a complete universal backtesting reference.
- When API details are unclear, implement explicit rolling or expanding temporal cutoffs manually:
  1. Slice train up to cutoff.
  2. Fit preprocessing and model on that slice only.
  3. Predict `steps=horizon`.
  4. Compare against actuals after the cutoff.
- Recommended metrics: MAE, RMSE, MASE/RMSSE, WAPE, bias/OPE, interval coverage, and interval width. Use MAPE/sMAPE only when actuals are safely nonzero.
- Kats source includes `smape`, `sbias`, and exceedance-style metrics in some model code, but do not rely on undocumented metric functions without checking installed source.

## Prediction Intervals

- Many local model `predict()` docs say output includes `fcst_lower` and `fcst_upper`.
- Prophet intervals follow Prophet parameters such as `interval_width`, `mcmc_samples`, and `uncertainty_samples`.
- Quadratic model has `alpha` for confidence intervals.
- Bayesian VAR documentation says confidence intervals are not yet implemented even though output column descriptions mention lower/upper fields.
- Always validate empirical coverage on temporal holdout when intervals matter.

## Plotting and Diagnostics

- Plot source data with `TimeSeriesData.plot(cols=[...])`.
- Plot forecasts with `model.plot()` where implemented, or `Model.plot(data, forecast_df, include_history=True)`.
- Ensemble base class docs note plotting is not implemented in the base class; check concrete ensemble behavior.
- For residual diagnostics, compute `actual - fcst` on temporal holdout or historical backtest outputs. Inspect residual time plots, residual ACF/PACF externally, bias, and horizon-by-horizon error.
- Kats does not document a uniform residual diagnostics API across forecasters.

## Anti-Leakage Checks

- Do not random split rows.
- Do not call `interpolate()` across validation/test target periods before splitting.
- Do not compute normalization, categorical encodings, Fourier/calendar selection, model choice, hyperparameter search, or ensemble weights on full data.
- Do not let future covariates include values not known at the forecast creation timestamp.
- Do not mix independent panel IDs into a multivariate VAR unless they are truly jointly endogenous variables.
- Refit transformations and models inside each temporal cutoff during validation.
