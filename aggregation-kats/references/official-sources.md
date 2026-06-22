# Official Sources Consulted

Consulted for the `aggregation-kats` skill:

- Kats GitHub repository: https://github.com/facebookresearch/Kats
- Kats homepage: https://facebookresearch.github.io/Kats/
- Kats API index: https://facebookresearch.github.io/Kats/api/
- `kats.tsfeatures.tsfeatures` API: https://facebookresearch.github.io/Kats/api/kats.tsfeatures.tsfeatures.html
- `kats.consts.TimeSeriesData` API: https://facebookresearch.github.io/Kats/api/kats.consts.html
- Kats TSFeatures tutorial notebook: https://github.com/facebookresearch/Kats/blob/main/tutorials/kats_203_tsfeatures.ipynb
- Kats source for TSFeatures: https://github.com/facebookresearch/Kats/blob/main/kats/tsfeatures/tsfeatures.py
- PyPI package page: https://pypi.org/project/kats/

Notes captured from official sources:

- Kats is documented as a time-series analysis toolkit covering forecasting, detection, feature extraction/embedding, and multivariate analysis.
- The homepage says the TSFeature extraction module can produce 65 features for ML models.
- The current source maps default and optional groups to explicit feature names, including time-based features added in version 0.2.0.
- The tutorial introduces `TsFeatures`, applications with multiple time series, and opt-in/opt-out feature calculation.
- `TimeSeriesData` is the fundamental Kats data structure and can be initialized from pandas `DataFrame`, `Series`, or `DatetimeIndex` plus values.
- `TsFeatures.transform()` returns a dict for univariate input and a list of dicts for multivariate input.
- PyPI latest release is `kats 0.2.0`, released March 15, 2022, with Python 3.7/3.8 classifiers and alpha development status.
