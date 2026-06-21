# Official Sources Consulted

Checked on 2026-06-21.

- Nixtla StatsForecast GitHub README: https://github.com/Nixtla/statsforecast
- Raw README: https://raw.githubusercontent.com/Nixtla/statsforecast/main/README.md
- Documentation home/index: https://nixtlaverse.nixtla.io/statsforecast/
- Installation guide: https://nixtlaverse.nixtla.io/statsforecast/docs/getting-started/installation.html
- Quick start: https://nixtlaverse.nixtla.io/statsforecast/docs/getting-started/getting_started_short.html
- End-to-end walkthrough: https://nixtlaverse.nixtla.io/statsforecast/docs/getting-started/getting_started_complete.html
- Polars walkthrough: https://nixtlaverse.nixtla.io/statsforecast/docs/getting-started/getting_started_complete_polars.html
- Cross-validation tutorial: https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/crossvalidation.html
- Exogenous regressors guide: https://nixtlaverse.nixtla.io/statsforecast/docs/how-to-guides/exogenous.html
- Generating features guide: https://nixtlaverse.nixtla.io/statsforecast/docs/how-to-guides/generating_features.html
- Conformal prediction tutorial: https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/conformalprediction.html
- Probabilistic forecasting tutorial: https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/probabilisticforecasting.html
- Core methods API: https://nixtlaverse.nixtla.io/statsforecast/src/core/core.html
- Models API/reference: https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html
- Official model source exports: https://raw.githubusercontent.com/Nixtla/statsforecast/main/python/statsforecast/models.py
- Distributed docs: Dask, Ray, and Spark pages under https://nixtlaverse.nixtla.io/statsforecast/docs/distributed/

Documentation notes:

- Latest GitHub release observed: v2.0.3 on 2025-10-29.
- README/docs and source exports are not identical. The skill treats `UCM`, `SklearnModel`, `ConstantModel`, `ZeroModel`, and `NaNModel` as source-exported names that require local verification.
- No universal residual diagnostics API was identified in the consulted docs; diagnostics should be computed from held-out forecasts, fitted values, or cross-validation output.
- StatsForecast is documented as local univariate modeling over many series, with exogenous regressors and static covariates where supported, not as a joint multivariate forecasting library.
