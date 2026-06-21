---
name: forecasting-data-prep
description: Prepare, validate, diagnose, and split time-series datasets before forecasting with Prophet, Darts, StatsForecast, NeuralForecast, sktime, Nixtla, GluonTS, or similar libraries. Use this skill whenever an agent must identify time and target columns, validate frequency and panel structure, handle gaps/missing values/outliers/exogenous variables, create temporal splits/backtests, and prevent data leakage before modeling.
---

# Forecasting Data Prep

Use this skill before any forecasting-model skill. The output should be a clean data contract, diagnostics, temporal splits, and explicit leakage notes. Do not start modeling until the readiness checklist passes or the remaining risks are accepted.

## Default Workflow

1. **Inspect schema and intent**
   - Identify the timestamp column, target column, optional series ID columns, and candidate exogenous variables.
   - Prefer explicit user-provided column names. If inferring, state confidence and ambiguous alternatives.
   - Confirm the business meaning of one row: event, transaction, snapshot, aggregate period, or already-resampled observation.

2. **Normalize time**
   - Parse timestamps with an explicit format when known.
   - Use one canonical timezone. Prefer UTC for storage and convert only for reporting/calendar features.
   - Sort by `series_id` columns and timestamp. Require monotonic time within each series.
   - Decide whether timestamps represent period start, period end, or instant observations.

3. **Validate grain and frequency**
   - Infer frequency per series where possible; compare with the required modeling frequency.
   - Check for mixed granularities, irregular intervals, daylight-saving artifacts, and partial periods.
   - If resampling is needed, define aggregation rules per column: target sum/mean/last as appropriate, known-future flags by max/any, prices by last/mean, static fields by first after consistency checks.

4. **Audit data quality**
   - Detect duplicate `(series_id, timestamp)` keys and resolve them before splitting.
   - Quantify missing timestamps, missing target values, and missing exogenous values separately.
   - Treat temporal gaps differently from null targets: a missing timestamp means the row is absent; a null target means the row exists but target is unknown.
   - Flag outliers with robust rules, then decide whether to keep, cap, correct, or annotate. Do not remove outliers automatically.

5. **Handle panels and exogenous variables**
   - For panel data, validate each series independently: start/end dates, frequency, length, missingness, and enough history for the requested horizon.
   - Classify exogenous variables as:
     - `known_future`: calendar, holidays, scheduled promotions, planned prices, contractual capacity.
     - `observed_past`: weather observations, realized demand drivers, lagged operations, sensor values.
     - `static`: store, item, geography, category, segment.
     - `unknown`: requires user confirmation before modeling.
   - Never use a variable at prediction time unless it will truly be available for the full forecast horizon.

6. **Split before fitting transformations**
   - Create train/validation/test by timestamp cutoffs, not random split.
   - Fit imputers, scalers, encoders, target transforms, anomaly thresholds, feature selection, and PCA only on train.
   - Apply fitted transforms forward to validation/test. Refit only inside a documented rolling-origin or expanding-window backtest fold.

7. **Create features without leakage**
   - Create lag and rolling features with `shift(1)` or a horizon-aware offset before rolling/expanding calculations.
   - For direct multi-horizon models, ensure each feature uses only data available at the forecast creation timestamp.
   - Do not compute global statistics, encodings, interpolation, or calendar/event joins using validation/test targets.

8. **Set validation and metrics**
   - Define `forecast_horizon`, optional `gap`, step size, number of folds, and minimum training window.
   - Use holdout test only once for final evaluation. Use validation/backtesting for model selection.
   - Recommended metrics: MAE/RMSE for scale-dependent error, MASE/RMSSE for cross-series comparison, sMAPE/WAPE when appropriate, pinball loss or CRPS for probabilistic forecasts, and bias/mean error for systematic over/under-forecasting.

9. **Produce minimum diagnostics**
   - Plot target over time for representative series and aggregate total if panel data.
   - Plot missingness/gaps by time and by series.
   - Plot train/validation/test cutoffs on the target chart.
   - Plot residual-like anomalies only after defining a leakage-safe baseline or robust threshold from train.

## Deterministic Audit Script

For tabular files, run the bundled audit before manual cleanup. It requires `pandas` in the active Python environment:

```bash
python forecasting-data-prep/scripts/forecasting_data_audit.py data.csv \
  --time-col ds \
  --target-col y \
  --id-cols unique_id \
  --freq D \
  --horizon 14
```

The script emits JSON diagnostics for inferred columns, duplicates, gaps, missingness, frequency, panel length, exogenous candidates, outlier counts, and split readiness. Treat it as a first pass; domain decisions still require human or task context.

## References

- Read `references/data-contract.md` when column roles, grain, panel structure, or resampling rules are unclear.
- Read `references/validation-and-leakage.md` when designing temporal splits, rolling-origin backtests, feature windows, or transformation pipelines.
- Read `references/library-selection.md` when choosing preparation libraries and dependencies.

## Ready for Modeling Checklist

- Timestamp and target columns are explicit and parsed correctly.
- Data is sorted and unique at `(series_id, timestamp)` or `(timestamp)` for single-series data.
- Timezone, timestamp semantics, frequency, and period boundaries are documented.
- Gaps, duplicates, missing targets, and outliers are quantified and handled or accepted.
- Each panel series has enough history for the horizon, validation plan, and required lags.
- Exogenous variables are classified as known future, observed past, static, or excluded.
- Train/validation/test or backtest folds use temporal cutoffs with any required gap.
- All transformations are fit on train only and applied forward.
- Lag, rolling, expanding, interpolation, and encodings use only past information.
- Metrics match the business objective and support comparison across series when needed.
