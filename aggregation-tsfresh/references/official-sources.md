# Official Sources Consulted

Consulted for the `aggregation-tsfresh` skill:

- tsfresh GitHub repository: https://github.com/blue-yonder/tsfresh
- tsfresh documentation home: https://tsfresh.readthedocs.io/en/latest/
- Quick Start: https://tsfresh.readthedocs.io/en/latest/text/quick_start.html
- Data Formats: https://tsfresh.readthedocs.io/en/latest/text/data_formats.html
- sklearn Transformers: https://tsfresh.readthedocs.io/en/latest/text/sklearn_transformers.html
- Transformer API: https://tsfresh.readthedocs.io/en/latest/api/tsfresh.transformers.html
- Feature Extraction Settings: https://tsfresh.readthedocs.io/en/latest/text/feature_extraction_settings.html
- Feature Filtering: https://tsfresh.readthedocs.io/en/latest/text/feature_filtering.html
- Feature Calculators list: https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
- Feature extraction API: https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html
- Convenience API: https://tsfresh.readthedocs.io/en/latest/api/tsfresh.convenience.html
- Large Data: https://tsfresh.readthedocs.io/en/latest/text/large_data.html
- Forecasting / rolling windows: https://tsfresh.readthedocs.io/en/latest/text/forecasting.html

Notes captured from official docs:

- `pip install tsfresh` is the minimum install; `tsfresh[dask]` is documented for Dask workflows.
- The docs identify flat/wide DataFrames, stacked/long DataFrames, and dictionaries of flat DataFrames as accepted formats.
- Special columns used for id, sort, kind, and value must not contain `NaN`, `Inf`, or `-Inf`.
- `extract_features()` returns one feature row per id.
- `extract_relevant_features()` and selection APIs are supervised and must be fit only on train/fold data.
- `make_forecasting_frame()` is documented as limited to one-dimensional time series of one id and one kind.
