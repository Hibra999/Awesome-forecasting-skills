# Official Sources Consulted

- User-provided archived GitHub repository: https://github.com/tinkoff-ai/etna
- Current ETNA documentation: https://docs.etna.ai/stable/
- Installation docs: https://docs.etna.ai/stable/installation.html
- Classification tutorial: https://docs.etna.ai/stable/tutorials/305-classification.html
- API reference index: https://docs.etna.ai/stable/api_reference.html
- Current GitHub repository linked by docs: https://github.com/etna-team/etna
- `TimeSeriesBinaryClassifier` source: https://github.com/etna-team/etna/blob/master/etna/experimental/classification/classification.py
- `PredictabilityAnalyzer` source: https://github.com/etna-team/etna/blob/master/etna/experimental/classification/predictability.py
- Feature extraction source directory: https://github.com/etna-team/etna/tree/master/etna/experimental/classification/feature_extraction
- `TSFreshFeatureExtractor` source: https://github.com/etna-team/etna/blob/master/etna/experimental/classification/feature_extraction/tsfresh.py
- `WEASELFeatureExtractor` source: https://github.com/etna-team/etna/blob/master/etna/experimental/classification/feature_extraction/weasel.py
- `pyproject.toml` extras: https://github.com/etna-team/etna/blob/master/pyproject.toml

## Important Documentation Notes

- ETNA docs stable version reviewed: 3.0.0.
- The user link points to `tinkoff-ai/etna`, which is archived on GitHub; current docs link to `etna-team/etna`.
- Classification is documented as experimental.
- The installation page spells the classification extra as `classiciation`, but the tutorial uses `etna[classification]` and current `pyproject.toml` defines `classification = ["pyts", "tsfresh"]`.
