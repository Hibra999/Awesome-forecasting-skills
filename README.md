# Awesome Forecasting Skills

Skills for AI agents working on forecasting, time-series data preparation, time-series classification, time-series pattern discovery, time-series feature aggregation, and change point detection workflows.

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
| `classification-tslearn` | tslearn time-series classification with 3D arrays, DTW/GAK KNN/SVC, shapelets, MLP, early classification, stratified evaluation, and leakage-safe preprocessing. |
| `classification-pyts` | pyts time-series classification with fixed-length 2D/3D arrays, SAXVSM, BOSSVS, KNN/DTW, shapelets, interval forests, TSBF, WEASEL/MUSE pipelines, and leakage-safe validation. |
| `classification-dl-4-tsc` | hfawaz dl-4-tsc research-code classification with UCR TSV/MTS NPY inputs, TensorFlow/Keras DNNs, fixed 3D tensors, CLI runs, saved metrics, and leakage-aware validation. |
| `classification-etna` | ETNA experimental binary time-series classification with TSFresh/WEASEL feature extraction, sklearn classifiers, fold masks, predictability analysis, and leakage-safe validation. |
| `clustering-stumpy` | STUMPY matrix-profile pattern discovery with query matching, motifs/discords, multidimensional motifs, MPdist similarity, snippets, segmentation, and leakage-aware validation. |
| `aggregation-tsfresh` | tsfresh feature aggregation for transforming time series into tabular ML matrices, with flat/stacked/dict formats, feature settings, relevance filtering, sklearn transformers, Dask/Spark scaling, rolling windows, and anti-leakage validation. |
| `aggregation-kats` | Kats TSFeatures aggregation for converting TimeSeriesData into tabular feature rows, with univariate/multivariate outputs, multiple-series loops, feature groups, opt-in/out switches, and leakage-safe downstream ML validation. |
| `aggregation-tsfel` | TSFEL feature aggregation for tabular ML matrices, with ndarray/Series/DataFrame inputs, univariate/multivariate signals, windowing, statistical/temporal/spectral/fractal domains, dataset extraction, custom configs, and anti-leakage validation. |
| `aggregation-tsflex` | tsflex flexible feature aggregation with pandas Series/DataFrame/list inputs, asynchronous multivariate data, irregular sampling, custom callable features, strided windows, chunking, external feature wrappers, and anti-leakage validation. |
| `changepoint-ruptures` | ruptures offline change point detection with exact/approximate search methods, cost functions, penalties, breakpoint-count tuning, segmentation metrics, plotting, custom costs, and anti-leakage validation. |
| `changepoint-prophet` | Prophet trend changepoint analysis inside forecasting workflows, with ds/y validation, automatic/manual changepoints, prior-scale tuning, significant-delta plotting, cross-validation, intervals, regressors, shocks, and anti-leakage safeguards. |
| `changepoint-kats` | Kats changepoint and process-change detection with TimeSeriesData, CUSUM, BOCPD, robust statistical detection, rolling CUSUM models, related trend/statistical-change detectors, evaluation, plotting, and anti-leakage validation. |
| `changepoint-time-series-library` | THUML Time-Series-Library process-change event detection via the documented anomaly_detection task, with PSM/MSL/SMAP/SMD/SWAT loaders, reconstruction-error thresholds, anomaly_ratio, event-level adjustment, metrics, and leakage-safe anomaly-to-changepoint conversion. |
| `changepoint-merlion` | Salesforce Merlion native BOCPD changepoint detection with TimeSeries inputs, LevelShift/TrendChange/Auto change kinds, online updates, z-score probability scoring, thresholded alarms, TSAD metrics, plotting, and anti-leakage validation. |

## Workflow Notes

- Forecasting skills depend conceptually on `forecasting-data-prep`: prepare and validate data first, then model.
- Classification skills depend conceptually on `ts-classification-data-prep`: prepare, pad/truncate, validate labels and tensor/panel shapes first, then classify.
- Pattern discovery skills require ordered, numeric time-series data with documented window length, normalization, split, and validation policy before mining motifs or matches.
- Aggregation skills depend conceptually on time-series data preparation: define id/window/time/value fields first, then extract features and fit supervised selectors only on train folds.
- Change point detection skills require ordered numeric signals, explicit offline vs online assumptions, method/cost/stopping-rule selection, and validation that does not leak future segments into operational predictions.
- Each skill keeps `SKILL.md` concise and uses `references/` for longer model maps, data format notes, official sources, and limitations.
- Scripts are included only when they add repeatable validation.
