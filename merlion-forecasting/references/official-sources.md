# Official Sources Consulted

Checked on 2026-06-21.

- Salesforce Merlion GitHub README/repo: https://github.com/salesforce/Merlion
- Raw README: https://raw.githubusercontent.com/salesforce/Merlion/main/README.md
- Documentation home/API index: https://opensource.salesforce.com/Merlion/latest/merlion.html
- Forecast models API: https://opensource.salesforce.com/Merlion/latest/merlion.models.forecast.html
- AutoML API: https://opensource.salesforce.com/Merlion/latest/merlion.models.automl.html
- Ensemble API: https://opensource.salesforce.com/Merlion/latest/merlion.models.ensemble.html
- Evaluation API: https://opensource.salesforce.com/Merlion/latest/merlion.evaluate.html
- Intro forecasting tutorial: https://opensource.salesforce.com/Merlion/latest/tutorials/forecast/0_ForecastIntro.html
- Forecaster features tutorial: https://opensource.salesforce.com/Merlion/latest/tutorials/forecast/1_ForecastFeatures.html
- Multivariate forecasting tutorial: https://opensource.salesforce.com/Merlion/latest/tutorials/forecast/2_ForecastMultivariate.html
- Exogenous regressors tutorial: https://opensource.salesforce.com/Merlion/latest/tutorials/forecast/3_ForecastExogenous.html
- Examples tree and notebooks: https://github.com/salesforce/Merlion/tree/main/examples
- Model factory source aliases: https://github.com/salesforce/Merlion/blob/main/merlion/models/factory.py

Documentation notes:

- GitHub indicates the repository was archived on March 11, 2026 and is read-only.
- Latest GitHub release observed: v2.0.2 on February 15, 2023.
- The official docs are Sphinx pages under `opensource.salesforce.com/Merlion/latest`; version links show v2.0.2 as the latest documented release.
- No official generic panel forecasting container was identified; use loops or external wrappers for multiple independent series.
- No universal residual diagnostics API was identified; compute residual diagnostics from held-out `TimeSeries` predictions.
