# Official Sources Consulted

Consulted on 2026-06-22.

- sktime GitHub README: https://github.com/sktime/sktime
- sktime stable documentation index: https://www.sktime.net/en/stable/
- Installation guide: https://www.sktime.net/en/stable/installation.html
- Time-series classification example notebook: https://www.sktime.net/en/stable/examples/02_classification.html
- In-memory data representations and data loading: https://www.sktime.net/en/stable/examples/AA_datatypes_and_datasets.html
- Time-series classification API reference: https://www.sktime.net/en/stable/api_reference/classification.html
- Data format specifications/API reference: https://www.sktime.net/en/stable/api_reference.html

Notes:
- The stable API reference lists classifier classes and meta-estimators under "Time series classification".
- The tutorial notes that classifiers use estimator tags such as `capability:multivariate`, `capability:unequal_length`, `capability:missing_values`, and `capability:predict_proba`; always check tags in the installed version.
- The data-format tutorial says sktime provides no unequal-length classification example data directly, but UCR/UEA datasets may be unequal length/index and can be loaded through `load_UCR_UEA_dataset`.
