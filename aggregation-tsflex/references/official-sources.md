# Official Sources Consulted

Consulted for the `aggregation-tsflex` skill:

- tsflex GitHub repository: https://github.com/predict-idlab/tsflex
- tsflex README: https://github.com/predict-idlab/tsflex/blob/main/README.md
- tsflex documentation home: https://predict-idlab.github.io/tsflex/
- Feature extraction API: https://predict-idlab.github.io/tsflex/features/index.html
- Feature collection API: https://predict-idlab.github.io/tsflex/features/feature_collection.html
- Integration wrappers API: https://predict-idlab.github.io/tsflex/features/integrations.html
- Processing API: https://predict-idlab.github.io/tsflex/processing/index.html
- Chunking API: https://predict-idlab.github.io/tsflex/chunking/index.html
- PyPI package page: https://pypi.org/project/tsflex/
- Official example notebooks folder: https://github.com/predict-idlab/tsflex/tree/main/examples

Notes captured from official sources:

- tsflex is documented as a flexible time-series processing and feature extraction toolkit.
- Installation is `pip install tsflex` or `conda install -c conda-forge tsflex`.
- Latest PyPI release is 0.4.1 from September 6, 2024; docs show `__version__ = "0.4.1"`.
- Accepted data is based on pandas Series/DataFrame objects, lists of those objects, and documented GroupBy support in `FeatureCollection.calculate`.
- tsflex supports multivariate, multimodal, unevenly sampled, and asynchronous data, but makes the user responsible for robust feature functions.
- Feature extraction is configured with `FeatureCollection`, `FeatureDescriptor`, `MultipleFeatureDescriptors`, and `FuncWrapper`.
- tsflex uses callable feature functions and wrappers for external packages rather than a fixed built-in feature catalog.
- Official limitations include no multi-index/multi-column support, unique series-name requirements, compatible index dtypes, and no multiprocessing support on Windows.
