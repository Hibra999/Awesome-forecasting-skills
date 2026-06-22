# Kats TSFeatures Data Contract

Use this reference when turning prepared time-series data into tabular rows with Kats `TsFeatures`.

## Core Container

Kats uses `kats.consts.TimeSeriesData` as its main data structure. Official API docs say it can be initialized from:

- `pandas.DataFrame`
- `pandas.Series`
- `pandas.DatetimeIndex` plus value data

Typical input uses a time column named `time`, or a custom column passed with `time_col_name`.

```python
from kats.consts import TimeSeriesData

ts = TimeSeriesData(df[["time", "value"]], time_col_name="time")
```

`TimeSeriesData` stores:

- `time`: pandas `Series` of time values.
- `value`: pandas `Series` for univariate data or pandas `DataFrame` for multivariate data.

Useful methods documented in the API include `validate_data()`, `is_data_missing()`, `interpolate()`, `infer_freq_robust()`, `to_dataframe()`, `to_array()`, and `plot(cols=[...])`.

## Univariate Series

For one value column, `TsFeatures().transform(ts)` returns one dictionary:

```python
{"length": 90.0, "mean": 123.4}
```

Convert it to one tabular row by adding the entity/window id:

```python
row = {"series_id": series_id, **TsFeatures().transform(ts)}
```

## Multivariate Series

The official `transform()` docs state that multivariate input returns a list of feature maps. Treat each dictionary as a component/column output and add explicit identifiers before building a tabular matrix.

Do not assume a single merged row unless you define and test a naming convention such as `"{column}__{feature}"`.

## Multiple Independent Series

Kats does not document one long/panel container for independent ids. Use a loop:

```python
rows = []
for series_id, group in df.groupby("series_id", sort=False):
    ts = TimeSeriesData(group[["time", "value"]], time_col_name="time")
    rows.append({"series_id": series_id, **TsFeatures().transform(ts)})
```

For rolling windows, group by `(series_id, window_end)` or an equivalent window id after constructing past-only windows.

## Minimum Length and Missing Data

Current source raises a `ValueError` when `len(ts) < 5`. Several feature groups can still return `NaN` or fail/skip useful values when the series is too short for `window_size`, `stl_period`, `nbins`, `lag_size`, or `acfpacf_lag`.

Validate before extraction:

- Time values are present and sortable.
- No duplicate time values inside a series/window unless intentionally handled.
- Value columns are numeric and finite.
- Series/window length is at least 5 and preferably long enough for selected groups.

## Leakage-Safe Rows

When building feature rows for supervised learning:

- Split before computing scalers, PCA, selectors, or downstream models.
- If labels are tied to future outcomes, each feature window must end before the label horizon starts.
- Do not let validation/test rows influence imputation, feature-group choice, scaling, PCA components, or model selection.
