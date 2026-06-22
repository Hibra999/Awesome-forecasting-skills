# tsfresh Data Formats

Use this reference when converting prepared time-series data into the formats accepted by `tsfresh.extract_features()`.

## Supported Containers

`extract_features()` accepts:

- A pandas `DataFrame` in flat/wide format.
- A pandas `DataFrame` in stacked/long format.
- A dictionary of flat pandas `DataFrame` objects keyed by kind.

The output is a pandas `DataFrame` with one row per `column_id` value and one column per extracted feature.

## Flat/Wide DataFrame

Use this when each time-series kind is already a separate value column.

Required structure:

- `column_id`: identifies the entity, sample, segment, or rolled window.
- `column_sort`: optional time/order column; pass it unless row order is already guaranteed.
- value columns: all other time-series columns, unless excluded before extraction.
- `column_kind=None` and `column_value=None`.

Example schema:

| id | time | temperature | pressure |
| --- | --- | --- | --- |
| a | 1 | 20.1 | 3.0 |
| a | 2 | 20.4 | 3.2 |
| b | 1 | 18.9 | 2.9 |

Call pattern:

```python
extract_features(df, column_id="id", column_sort="time")
```

## Stacked/Long DataFrame

Use this when the time-series kind/name lives in a column and values live in one numeric value column.

Required structure:

- `column_id`: identifies the entity/sample/window.
- `column_sort`: optional time/order column.
- `column_kind`: identifies the signal/kind.
- `column_value`: contains numeric time-series values.

Example schema:

| id | time | kind | value |
| --- | --- | --- | --- |
| a | 1 | temperature | 20.1 |
| a | 1 | pressure | 3.0 |
| a | 2 | temperature | 20.4 |

Call pattern:

```python
extract_features(
    df,
    column_id="id",
    column_sort="time",
    column_kind="kind",
    column_value="value",
)
```

## Dictionary of Flat DataFrames

Use this when each kind is stored separately. The dictionary key is the kind name and each value is a flat DataFrame with the same id/sort semantics.

```python
timeseries = {
    "temperature": temperature_df,
    "pressure": pressure_df,
}
extract_features(timeseries, column_id="id", column_sort="time")
```

## Required Data Quality

- `column_id`, `column_sort`, `column_kind`, and `column_value` may not contain `NaN`, `Inf`, or `-Inf`.
- If `column_sort` is omitted, tsfresh assumes the current row order is already correct.
- Equidistant timestamps are not required globally, but some calculators are meaningful only for equidistant series.
- Keep identifiers stable and align the target `y` index with extracted feature row ids before supervised selection.

## Rolling and Forecasting-Style Tables

`roll_time_series()` creates shifted/rolled windows and encodes ids such as `(original_id, shift_time)`. Features for a positive-direction rolling window should only use observations up to the window endpoint.

`make_forecasting_frame()` creates a supervised forecasting-style feature/target frame, but the official docs state it works only for one-dimensional time series of one id and one kind.

## Large Data Notes

For large data, tsfresh documents:

- `n_jobs` and `chunksize` in `extract_features()`.
- Dask DataFrame support with `pip install tsfresh[dask]`.
- `pivot=False` to avoid materializing a huge wide final matrix too early.
- `dask_feature_extraction_on_chunk` and `spark_feature_extraction_on_chunk` for lower-level grouped workflows.
