# Awesome Forecasting Skills

Skills for AI agents working on forecasting, time-series data preparation, and time-series classification workflows.

## Skills

Use the foundational data-preparation skill before any library-specific modeling skill.

| Skill | Purpose |
| --- | --- |
| `forecasting-data-prep` | Foundational forecasting skill for preparing, validating, splitting, and diagnosing time-series data before modeling. |
| `prophet-forecasting` | Prophet forecasting with regressors, holidays, temporal validation, intervals, diagnostics, and leakage checks. |
| `statsmodels-forecasting` | Statsmodels classical/econometric forecasting, including ARIMA/SARIMAX/ETS/state-space models, exogenous regressors, intervals, and diagnostics. |
| `sktime-forecasting` | sktime forecasting workflows, including forecasters, transformations, exogenous variables, backtesting, metrics, and probabilistic outputs where supported. |
| `darts-forecasting` | Darts forecasting with `TimeSeries`, statistical/ML/deep models, covariates, backtesting, probabilistic forecasts, and multiple-series workflows. |
| `kats-forecasting` | Kats forecasting with `TimeSeriesData`, model families, backtesting, intervals where supported, diagnostics, and documented caveats. |
| `time-series-library-forecasting` | THUML Time-Series-Library long-term forecasting experiments, model scripts, dataset formats, metrics, and anti-leakage setup. |
| `pytorch-forecasting` | PyTorch Forecasting workflows with `TimeSeriesDataSet`, group IDs, known/unknown covariates, deep models, Lightning training, and validation. |
| `statsforecast` | Nixtla StatsForecast fast statistical forecasting for many series, exogenous regressors, intervals, cross-validation, and distributed backends. |
| `merlion-forecasting` | Salesforce Merlion forecasting with `TimeSeries`, model pipelines, transforms, evaluation, visualization, and model-selection workflows. |
| `greykite-forecasting` | LinkedIn Greykite forecasting with Silverkite, regressors, events/holidays, uncertainty, diagnostics, and temporal evaluation. |
| `skforecast` | skforecast recursive/direct forecasters, sklearn-style regressors, statistical/foundation/deep integrations, backtesting, tuning, and intervals. |
| `etna-forecasting` | ETNA multi-segment forecasting with `TSDataset`, transforms, exogenous features, pipelines, backtesting, intervals, and diagnostics. |
| `fedot-forecasting` | FEDOT AutoML time-series forecasting with `InputData`, `TsForecastingParams`, operation repositories, validation, and exogenous workflows. |
| `classification-sktime` | sktime time-series classification after `ts-classification-data-prep`, including panel formats, classifier families, stratified CV, and anti-leakage validation. |
| `classification-time-series-library` | THUML Time-Series-Library deep learning classification with UEA `.ts` files, padding masks, official scripts, model support checks, and leakage-safe evaluation. |

## Workflow Notes

- Forecasting skills depend conceptually on `forecasting-data-prep`: prepare and validate data first, then model.
- Classification skills depend conceptually on `ts-classification-data-prep`: prepare, pad/truncate, validate labels and tensor/panel shapes first, then classify.
- Each skill keeps `SKILL.md` concise and uses `references/` for longer model maps, data format notes, official sources, and limitations.
- Scripts are included only when they add repeatable validation.
