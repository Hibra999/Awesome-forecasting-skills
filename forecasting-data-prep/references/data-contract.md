# Forecasting Data Contract

Load this reference when a dataset's structure, grain, or variable roles are ambiguous.

## Required Decisions

- `time_col`: one timestamp or period column. Prefer explicit user input over inference.
- `target_col`: numeric value to forecast. For counts or demand, document whether zero means true zero or missing activity.
- `id_cols`: zero or more columns that define independent series. A panel key is `id_cols + time_col`.
- `freq`: modeling frequency such as hourly, daily, weekly, monthly, or business day.
- `horizon`: number of future periods predicted from a forecast creation timestamp.
- `timestamp_semantics`: period start, period end, or instant observation.
- `timezone`: canonical timezone used for joins, features, and cutoffs.

## Column Detection Heuristics

- Time candidates usually contain names like `date`, `ds`, `time`, `timestamp`, `period`, `week`, `month`, or `datetime`, or parse cleanly as datetimes.
- Target candidates are numeric and often named `y`, `target`, `value`, `sales`, `demand`, `count`, `qty`, `revenue`, `load`, or `volume`.
- Series ID candidates are stable categorical keys such as `unique_id`, `series_id`, `store_id`, `item_id`, `sku`, `location`, or `meter_id`.
- Static covariates do not change within a series. Validate this before treating them as static.
- Exogenous candidates are all non-time, non-target, non-id columns. They must be classified by future availability before modeling.

## Grain and Resampling Rules

Before resampling, establish what each row measures. Do not aggregate transactions, snapshots, and already-aggregated observations with the same rule.

Common rules:

- Additive targets: use `sum` over the destination period, for example sales units or incidents.
- Average-state targets: use `mean`, for example temperature or utilization rate.
- End-of-period state: use `last`, for example inventory or account balance.
- Binary event flags: use `max` or `any`.
- Prices and rates: use `last`, `mean`, or a domain-specific weighted average.
- Static attributes: use `first` only after verifying one unique value per series.

After resampling, mark created rows and imputed targets so downstream models can use missingness indicators when appropriate.

## Quality Checks

- Duplicate keys: duplicates at `(id_cols, time_col)` must be aggregated, deduplicated, or rejected.
- Missing timestamps: compare actual timestamps with the expected date range per series.
- Missing targets: decide whether to impute, drop, mark unavailable, or leave for a model that supports missing values.
- Mixed frequency: flag series whose modal interval differs from the requested frequency.
- Short series: flag series with fewer observations than `minimum_train_window + horizon + gap`.
- Constant targets: keep only if the business process can be constant; otherwise verify data extraction.
- Negative values: validate against domain constraints for demand, counts, prices, and inventory.
- Outliers: flag robustly per series; do not remove without domain justification.

## Minimal Output Contract

Return or document:

```yaml
time_col: "ds"
target_col: "y"
id_cols: ["unique_id"]
freq: "D"
horizon: 14
timezone: "UTC"
timestamp_semantics: "period_start"
known_future_covariates: ["holiday", "planned_promo"]
observed_past_covariates: ["temperature_observed"]
static_covariates: ["store_region"]
excluded_columns: ["leaky_future_sales_total"]
splits:
  train_end: "2025-09-30"
  validation_end: "2025-10-31"
  test_end: "2025-11-14"
leakage_controls:
  random_split: false
  transforms_fit_on_train_only: true
  horizon_gap_respected: true
```
