# Greykite Changepoint Data Workflow

Use this reference before converting prepared time-series data into Greykite changepoint workflows.

## Accepted Data

- Input is a pandas `DataFrame`.
- Required columns are a parseable time column and one numeric value column.
- Greykite examples use `time_col="ts"` and `value_col="y"`, but names are configurable.
- Timestamps can be any format accepted by `pandas.to_datetime`.
- Keep entity IDs outside the detector call; loop per entity if needed.

## Temporal Split

Create chronological splits before:

- anomaly/outlier adjustment
- level-shift detection
- manual changepoint selection
- `regularization_strength` tuning
- potential changepoint spacing/count selection
- seasonality component selection
- forecast template/model selection

For retrospective analysis, document whether detected changepoints were found on full history or only on the training window. For operational claims, rerun detection at each historical cutoff.

## Custom Dates

Manual `dates` must be known at the training cutoff. Greykite maps changepoint dates to the closest time on or after the requested date within each training split and deduplicates dates.

Do not add dates because they explain final test residuals unless those dates were known business events before the forecast/detection cutoff.

## End Buffers

Avoid changepoints near the end of the training data. Use:

- `no_changepoint_proportion_from_end`
- `no_changepoint_distance_from_end`
- Prophet-through-Greykite: `changepoint_range`

The docs advise reserving enough data after the last changepoint to validate forecast accuracy in a backtest.

## Panels and Multiple Metrics

For independent series:

```python
for entity_id, frame in df.groupby("entity_id"):
    detector = ChangepointDetector()
    result = detector.find_trend_changepoints(frame, time_col="ts", value_col="y")
```

Do not pool entities unless the pooled target is the actual process being analyzed.

## Labels and Evaluation

If labels are exact changepoints, compare detected timestamps to labels with a tolerance window. If labels are intervals, count an alarm as true positive when a detected changepoint lands in or near the interval according to the project rule.

Report:

- precision, recall, F1
- false positives
- false negatives
- absolute delay to nearest label
- median/mean delay
- forecast metric deltas if changepoints are used for forecasting

## Forecast Integration

When using Silverkite:

- Provide `MetadataParam(time_col=..., value_col=..., freq=...)`.
- Provide `ModelComponentsParam(changepoints=...)`.
- Use `EvaluationPeriodParam` for temporal backtest and rolling CV.
- Inspect `result.backtest`, `result.forecast`, `result.grid_search`, `plot_components`, and changepoint detection plots.

Forecast regressors/events must be known for future timestamps. Changepoint detection does not make unknown future regressors valid.
