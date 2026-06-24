# Official Sources Consulted

- Luminaire GitHub repository: <https://github.com/zillow/luminaire/>
- Luminaire README / PyPI project description: <https://github.com/zillow/luminaire/blob/master/README.md>
- Luminaire documentation home: <https://zillow.github.io/luminaire/>
- Data Profiling tutorial: <https://zillow.github.io/luminaire/tutorial/dataprofiling.html>
- Configuration Optimization tutorial: <https://zillow.github.io/luminaire/tutorial/optimization.html>
- Outlier Detection tutorial: <https://zillow.github.io/luminaire/tutorial/outlier_batch.html>
- Streaming anomaly detection tutorial: <https://zillow.github.io/luminaire/tutorial/streaming.html>
- Batch model API reference: <https://zillow.github.io/luminaire/api_reference/batch_models.html>
- Data exploration API reference: <https://zillow.github.io/luminaire/api_reference/exploration.html>
- Configuration optimization API reference: <https://zillow.github.io/luminaire/api_reference/optimization.html>
- Streaming/window density API reference: <https://zillow.github.io/luminaire/api_reference/streaming.html>
- Luminaire PyPI package: <https://pypi.org/project/luminaire/>
- Source `setup.py`: <https://github.com/zillow/luminaire/blob/master/setup.py>
- Source `requirements.txt`: <https://github.com/zillow/luminaire/blob/master/requirements.txt>
- Source package tree for model/exploration/optimization modules via GitHub API.
- Luminaire paper: <https://arxiv.org/abs/2011.05047>

Consultation notes:

- README and PyPI describe Luminaire as a hands-off anomaly detection library for monitoring time-series data.
- Official docs document data profiling, batch outlier detection, configuration optimization, and streaming/window anomaly detection.
- The explicit model classes documented are `LADStructuralModel`, `LADFilteringModel`, and `WindowDensityModel`, with `DataExploration` and `HyperparameterOptimization` as workflow components.
- No dedicated plotting API or multivariate panel API is documented in the official pages reviewed.
