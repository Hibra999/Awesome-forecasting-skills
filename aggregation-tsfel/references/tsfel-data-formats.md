# TSFEL Data Formats

Use this reference when turning prepared time-series data into tabular feature matrices with TSFEL.

## In-Memory Inputs

`tsfel.time_series_features_extractor()` accepts:

- Python `list`
- `numpy.ndarray`
- pandas `Series`
- pandas `DataFrame`

Univariate data is one column/vector over time. Multivariate data must store dimensions in separate columns. TSFEL docs assume all dimensions share the same sampling frequency.

```python
cfg = tsfel.get_features_by_domain()
X = tsfel.time_series_features_extractor(cfg, data, fs=100)
```

## Windowing

Without `window_size`, TSFEL extracts one feature row for the full input series.

With `window_size`, TSFEL splits the signal into fixed-size windows and returns one row per window.

```python
X = tsfel.time_series_features_extractor(
    cfg,
    data,
    fs=100,
    window_size=100,
    overlap=0.0,
)
```

For supervised tasks, generate windows after train/validation/test splitting unless you can prove windows do not cross split boundaries or target horizons.

## Output

TSFEL always returns a pandas `DataFrame`.

- Columns are extracted feature names.
- Rows are full-series outputs or window outputs.
- For multivariate input, features from all dimensions are horizontally stacked.
- If input columns have names, those names prefix feature columns; otherwise TSFEL uses numeric prefixes such as `0_`.

## Sampling Requirements

TSFEL does not handle unevenly sampled data by default. The FAQ recommends transforming uneven series into equally spaced observations, for example with linear interpolation, before extraction.

Pass `fs` when extracting spectral features. The FAQ says the default sampling rate is 100 Hz, but recommends passing the true value and avoiding spectral features if the sampling rate is unavailable.

## Dataset Files

`dataset_features_extractor()` supports file-based datasets. Official docs describe these assumptions:

- Time series are stored in different file locations.
- Files use a delimited format.
- The first column contains timestamps.
- Following columns contain time-series values.
- Files might be unsynchronized in time; TSFEL can linearly interpolate/resample at a user-provided frequency.

Call pattern:

```python
data = tsfel.dataset_features_extractor(
    main_directory,
    tsfel.get_features_by_domain(),
    search_criteria=["Accelerometer.txt"],
    time_unit=1e-9,
    resample_rate=100,
    window_size=250,
    output_directory=output_directory,
)
```

The Get Started page shows `search_criteria` as a string, while the API reference
types it as a list and the source iterates over it. Prefer a list of filenames.

## Multiple Independent Series

TSFEL's in-memory extractor handles one signal/window set at a time, not a long panel with an id column. For independent ids, loop over groups and add explicit ids to the output rows.

```python
rows = []
for entity_id, group in df.groupby("entity_id", sort=False):
    X_i = tsfel.time_series_features_extractor(cfg, group[value_cols], fs=fs)
    X_i.insert(0, "entity_id", entity_id)
    rows.append(X_i)
X = pd.concat(rows, ignore_index=True)
```

## Data Quality Checks

- Value columns should be numeric and finite.
- Time/order should be sorted inside each entity/window.
- Sampling interval should be constant or intentionally resampled.
- `window_size` should be less than or equal to each train segment length.
- `overlap` should be `>= 0` and `< 1`.
