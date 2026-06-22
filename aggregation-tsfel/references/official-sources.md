# Official Sources Consulted

Consulted for the `aggregation-tsfel` skill:

- TSFEL GitHub repository: https://github.com/fraunhoferportugal/tsfel
- TSFEL documentation home: https://tsfel.readthedocs.io/en/latest/
- Get Started: https://tsfel.readthedocs.io/en/latest/descriptions/get_started.html
- Feature List: https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html
- Feature extraction API: https://tsfel.readthedocs.io/en/latest/descriptions/modules/tsfel.feature_extraction.html
- Personalised Features: https://tsfel.readthedocs.io/en/latest/descriptions/personal.html
- FAQ: https://tsfel.readthedocs.io/en/latest/descriptions/faq.html
- PyPI package page: https://pypi.org/project/tsfel/
- Source for `calc_features.py`: https://github.com/fraunhoferportugal/tsfel/blob/master/tsfel/feature_extraction/calc_features.py

Notes captured from official sources:

- TSFEL is documented as a time-series feature extraction library with statistical, temporal, spectral, and fractal domains.
- Docs and README state TSFEL extracts more than 65 distinct features.
- Installation is `pip install tsfel`; PyPI latest release is 0.2.0 from August 20, 2025.
- Supported in-memory inputs are list, ndarray, pandas Series, and pandas DataFrame.
- The extractor returns a pandas DataFrame with feature columns and one row per full signal or extracted window.
- Multivariate dimensions are separate columns and should share the same sampling frequency.
- TSFEL does not handle unevenly sampled data by default; resampling/interpolation is required before direct extraction.
- The documented/source core function is `time_series_features_extractor`.
