---
name: aggregation-kats
description: Use Facebook/Meta Kats TSFeatures for time-series aggregation and feature extraction after validating data, including TimeSeriesData inputs, univariate and multivariate feature dictionaries, multiple independent series loops, feature groups, opt-in/opt-out feature selection, tabular ML matrices, plotting context, and leakage-safe train/fold feature generation.
---

# Kats Aggregation

Use this skill after the time-series data-prep step. Kats `TsFeatures` is best for legacy or existing Kats projects that need interpretable statistical time-series features as tabular inputs for downstream classification, regression, clustering, anomaly workflows, meta-learning, or exploratory profiling.

Do not treat `TsFeatures` as a forecaster, classifier, clusterer, or supervised feature selector. It transforms `TimeSeriesData` into feature dictionaries; downstream modeling is separate.

## Minimum Install

```bash
python -m pip install --upgrade pip
python -m pip install kats
MINIMAL_KATS=1 python -m pip install kats
```

`MINIMAL_KATS=1` omits many dependencies and disables functionality. Kats 0.2.0 on PyPI is alpha-era software released in 2022 and declares Python 3.7/3.8 classifiers; check environment compatibility before promising execution.

## Data Contract

- Require a prepared contract: series/entity id, sorted timestamp column, numeric value column(s), train/validation/test split, window policy, target alignment if downstream supervised learning is used, and leakage notes.
- Convert each series to `kats.consts.TimeSeriesData`. Initialize from a pandas `DataFrame`, `Series`, or `DatetimeIndex` plus value data.
- Use the default `time` column or pass `time_col_name`; values can be a pandas `Series` for univariate data or a `DataFrame` for multivariate data.
- `TsFeatures().transform(ts)` returns `{feature: value}` for univariate input and `list[dict]` for multivariate input.
- Kats does not document one panel/long container for many independent series. Loop over ids and build one feature row per id/window.
- Source code raises `ValueError` for series shorter than 5 points; some feature groups require longer windows or seasonal periods.

Read `references/kats-tsfeatures-data.md` before adapting univariate, multivariate, multiple-series, or rolling-window data.

## Extraction Pattern

```python
import pandas as pd
from kats.consts import TimeSeriesData
from kats.tsfeatures.tsfeatures import TsFeatures

ts = TimeSeriesData(df[["time", "value"]], time_col_name="time")
features = TsFeatures().transform(ts)

row = {"series_id": series_id, **features}
```

For multiple independent series:

```python
rows = []
for series_id, group in train_df.groupby("series_id", sort=False):
    ts = TimeSeriesData(group[["time", "value"]], time_col_name="time")
    rows.append({"series_id": series_id, **TsFeatures().transform(ts)})

X_features = pd.DataFrame.from_records(rows).set_index("series_id")
```

For multivariate `TimeSeriesData`, handle the documented `list[dict]` output explicitly and name rows by component/column.

## Feature Configuration

- Default feature groups in current source include `statistics`, `stl_features`, `level_shift_features`, `acfpacf_features`, `special_ac`, `holt_params`, and `hw_params`.
- Optional groups default off in source: `cusum_detector`, `robust_stat_detector`, `bocp_detector`, `outlier_detector`, `trend_detector`, `nowcasting`, `seasonalities`, and `time`.
- Use `selected_features=[...]` to opt into named features or feature groups.
- Use keyword switches such as `stl_features=False` or `time=True` to opt out/in documented groups.
- Official docs/pages disagree on fixed feature counts; use group/feature names from the installed version.

Read `references/kats-tsfeatures-api-map.md` before selecting groups, parameters, or feature names.

## Validation and Metrics

- Validate the feature extraction plus downstream model with the split strategy required by the task: temporal splits for forecasting-like labels, group-aware splits for entity leakage, and stratified splits for imbalanced classification labels.
- For classification, report F1-macro, balanced accuracy, ROC-AUC/PR-AUC where probabilities exist, and confusion matrices.
- For regression, report MAE/RMSE plus scale-aware metrics when useful.
- For clustering, evaluate feature scaling, cluster stability, silhouette/task-specific metrics, and feature distribution sanity checks.
- Use `TimeSeriesData.plot(cols=[...])` or pandas/matplotlib to inspect raw series and windows; Kats does not document a dedicated `TsFeatures` plotting API.

## Anti-Leakage Rules

- Split ids, entities, windows, or temporal periods before imputing, scaling, PCA, feature selection, or fitting downstream models.
- Extract features separately inside each train fold when feature windows are derived from longer chronological series.
- Rolling/windowed feature rows must use only observations available at the row's decision timestamp.
- If feature group/parameter choices are selected from labels or validation metrics, select them inside train/validation folds only.
- Fit `StandardScaler`, PCA, supervised selectors, classifiers, regressors, and cluster postprocessing only on train folds.
- Keep timestamp cutoff, window length, seasonal period, frequency, entity id, and target alignment explicit in the data-prep contract.

## Common Errors

- Expecting a DataFrame output; `TsFeatures.transform()` returns a dict or list of dicts.
- Passing many independent series as one multivariate `TimeSeriesData` when one row per entity is required.
- Using series shorter than 5 points or too short for `window_size`, `stl_period`, `nbins`, or lag-based features.
- Assuming documented supervised feature selection or sklearn transformers exist in Kats TSFeatures; they are not documented.
- Forgetting optional detector, nowcasting, seasonality, and time groups are off by default in current source.
- Scaling/PCA on all feature rows before splitting.
- Relying on one fixed TSFeatures count despite official count inconsistencies across homepage, tutorial, and source.

## References

- Read `references/kats-tsfeatures-data.md` for `TimeSeriesData` contracts and tabular-row construction.
- Read `references/kats-tsfeatures-api-map.md` for feature groups, parameters, defaults, and documented gaps.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_kats_tsfeatures_csv.py` to sanity-check single-series or panel CSVs before creating `TimeSeriesData`.

## Ready Checklist

- Data-prep contract defines id, time, value columns, windows, split strategy, and leakage risks.
- Each feature row maps to one entity/component/window with no future observations.
- Kats/Python/dependency compatibility is checked for the target environment.
- Feature groups and parameters are documented from installed Kats source/docs.
- Scaling, PCA, selection, and downstream modeling are fit only on train folds.
- Metrics and plots are computed on held-out or validation data appropriate to the downstream task.
