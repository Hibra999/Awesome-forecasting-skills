# tsflex Data Formats

Use this reference when converting prepared time-series data into tsflex inputs.

## Accepted Inputs

Official docs define `data` as:

```python
import pandas as pd
from typing import List, Union

data: Union[pd.Series, pd.DataFrame, List[Union[pd.Series, pd.DataFrame]]]
```

`FeatureCollection.calculate()` also documents pandas `DataFrameGroupBy` support.

Each Series or DataFrame column is treated as a time series. Names matter:

- A Series should have a unique `name`.
- DataFrame columns become series names.
- No duplicate series names are allowed.

## Index Requirements

- Indexes must be sortable and monotonically increasing for chunking.
- All series used together must have compatible index dtypes.
- Time-string windows/strides such as `"15min"` require a time index.
- Numeric windows/strides can be sample/range based, but official docs caution against integer sample windows on time-indexed data unless fixed-frequency assumptions are valid.

## Wide vs Long

tsflex is built for:

- Wide DataFrames: one shared index, one column per modality/series.
- Series lists: independent named Series/DataFrames, useful for asynchronous or sparse modalities.

Official docs discourage blind `long -> wide` conversion when it introduces many `NaN` values. Prefer converting long data to a list of named Series:

```python
def long_dataframe_to_series_list(long_df, time_col, value_col, kind_col):
    series_list = []
    for name, group in long_df.groupby(kind_col, sort=False):
        series_list.append(
            pd.Series(group[value_col].to_numpy(), index=group[time_col], name=name)
        )
    return series_list
```

## Output

`FeatureCollection.calculate()` returns a pandas DataFrame or list of DataFrames.

Feature columns use:

```text
<SERIES-NAME>__<FEAT-NAME>__w=<WINDOW>__s=<STRIDE>
```

For grouped/consecutive extraction, the API documents returned group columns and `__start`/`__end` fields.

## Asynchronous and Irregular Series

tsflex supports unevenly sampled and asynchronous data. Consequences:

- Windows may contain different numbers of samples per series.
- Some windows may be empty.
- Many-to-one or many-to-many feature functions must handle different lengths after index alignment.
- `.calculate(..., approve_sparsity=True)` explicitly acknowledges sparse/irregular data.

## Chunking

Use `tsflex.chunking.chunk_data()` for gaps and continuous segments. It accepts pandas Series/DataFrame lists or dicts of DataFrames, optional `fs_dict`, `chunk_range_margin`, `min_chunk_dur`, `max_chunk_dur`, and `sub_chunk_overlap`.

Anti-leakage note: split before chunking or ensure chunks/sub-chunks cannot cross split boundaries or target horizons.

## Unsupported or Limited

Official docs state:

- Multi-indexes are not supported.
- Multi-columns are not supported.
- Duplicate series names are not allowed.
- All series used together need the same sequence-index dtype.
