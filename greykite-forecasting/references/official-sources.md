# Official Sources Consulted

Checked on 2026-06-21.

- LinkedIn Greykite GitHub README/repo: https://github.com/linkedin/greykite
- Raw README: https://raw.githubusercontent.com/linkedin/greykite/master/README.rst
- Setup metadata: https://raw.githubusercontent.com/linkedin/greykite/master/setup.py
- Documentation home: https://linkedin.github.io/greykite/docs/1.0.0/html/index.html
- Forecast model overview: https://linkedin.github.io/greykite/docs/1.0.0/html/pages/overview/100_forecast_intro.html
- Choose a model: https://linkedin.github.io/greykite/docs/1.0.0/html/pages/stepbystep/0100_choose_model.html
- Choose a model template: https://linkedin.github.io/greykite/docs/1.0.0/html/pages/stepbystep/0200_choose_template.html
- Configure a forecast: https://linkedin.github.io/greykite/docs/1.0.0/html/pages/stepbystep/0400_configuration.html
- Check forecast result: https://linkedin.github.io/greykite/docs/1.0.0/html/pages/stepbystep/0500_output.html
- Model components overview: https://linkedin.github.io/greykite/docs/1.0.0/html/pages/model_components/0100_introduction.html
- Regressors: https://linkedin.github.io/greykite/docs/1.0.0/html/pages/model_components/0700_regressors.html
- Auto-regression: https://linkedin.github.io/greykite/docs/1.0.0/html/pages/model_components/0800_autoregression.html
- Uncertainty intervals: https://linkedin.github.io/greykite/docs/1.0.0/html/pages/model_components/0900_uncertainty.html
- Benchmarking and rolling CV: https://linkedin.github.io/greykite/docs/1.0.0/html/pages/benchmarking/benchmarking.html
- Model templates source: https://raw.githubusercontent.com/linkedin/greykite/master/greykite/framework/templates/model_templates.py
- Quickstart, tutorials, and model template galleries under https://linkedin.github.io/greykite/docs/1.0.0/html/gallery/

Documentation notes:

- GitHub release page showed Greykite 1.0.0 as latest release on January 18, 2024, while current `master` setup metadata declares version 1.1.0. Pin and verify the installed package version.
- The official docs are generated for 1.0.0 and include API reference, tutorials, quickstarts, and examples.
- No official generic panel forecasting container was identified; the main `Forecaster` workflow uses one `value_col`.
- No broad multivariate endogenous forecast API was identified; reconciliation is post-processing for multiple base forecasts.
