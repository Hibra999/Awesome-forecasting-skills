# Temporal Validation and Leakage

Load this reference when designing splits, backtests, feature windows, or preprocessing pipelines.

## Split Patterns

- **Single cutoff holdout**: train before cutoff, validate/test after cutoff. Use for quick diagnostics or final test.
- **Three-way temporal split**: train, validation, then test. Use validation for model and feature selection; reserve test for final reporting.
- **Rolling-origin backtest**: move the cutoff forward by a fixed step. Use when recent performance stability matters.
- **Expanding-window backtest**: training begins at the first observation and expands each fold. Good default for many forecasting tasks.
- **Sliding-window backtest**: fixed-length training window moves forward. Use when old history becomes irrelevant.
- **Gap-aware split**: leave a gap between train end and forecast start when labels, reporting delays, or feature availability require it.

## Required Parameters

- `forecast_horizon`: number of periods predicted per forecast origin.
- `gap`: number of periods unavailable between train end and forecast start.
- `step`: distance between backtest origins.
- `initial_window`: minimum train history for the first fold.
- `cutoff_timestamp`: last timestamp visible to the training process for a fold.
- `frequency`: fixed temporal grain used to interpret horizon and gap.

## Anti-Leakage Rules

- Never use random split for forecasting evaluation.
- Split by time before fitting preprocessing or target transformations.
- Fit scalers, imputers, encoders, feature selectors, anomaly thresholds, and decomposition parameters on train only.
- Create lag features with an explicit shift so the current target is not included.
- Create rolling or expanding features only from timestamps at or before the forecast creation cutoff.
- Do not interpolate target values across validation/test cutoffs.
- Do not use realized future covariates unless the model scenario explicitly has them at prediction time.
- Recompute features independently inside each backtest fold.
- Keep final test data untouched until the modeling workflow is selected.

## Feature Timing Checklist

For every feature, answer:

- Is this value known at forecast creation time?
- If known in the future, is it known for every horizon step?
- If observed in the past, what reporting delay applies?
- Does a rolling window include the target at time `t` when predicting `t`?
- Was any statistic fit using validation or test rows?
- Does panel normalization use future rows from the same series or other series?

## Metric Selection

- Use MAE for interpretable average absolute error.
- Use RMSE when large errors should be penalized more strongly.
- Use MASE or RMSSE to compare across series with different scales.
- Use WAPE for aggregate demand planning, but be careful when actual totals are near zero.
- Use sMAPE only after checking behavior near zero.
- Use pinball loss for quantile forecasts.
- Use CRPS for full probabilistic forecast distributions when supported.
- Always include bias or mean error when systematic over- or under-forecasting matters.
