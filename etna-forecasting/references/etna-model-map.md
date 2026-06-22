# ETNA Model And Capability Map

Use this file before selecting or claiming ETNA model support. Names below come from ETNA 3.0 stable API reference and README. Do not add aliases without checking current official docs or local installed source.

## Forecasting Models

Baselines and moving averages:
- `NaiveModel`
- `MovingAverageModel`
- `SeasonalMovingAverageModel`
- `DeadlineMovingAverageModel`

Classical and smoothing models:
- `AutoARIMAModel`
- `SARIMAXModel`
- `SimpleExpSmoothingModel`
- `HoltModel`
- `HoltWintersModel`
- `BATSModel`
- `TBATSModel`
- `ProphetModel`

StatsForecast-backed wrappers:
- `StatsForecastARIMAModel`
- `StatsForecastAutoARIMAModel`
- `StatsForecastAutoCESModel`
- `StatsForecastAutoETSModel`
- `StatsForecastAutoThetaModel`

Regression and sklearn-style wrappers:
- `CatBoostMultiSegmentModel`
- `CatBoostPerSegmentModel`
- `ElasticMultiSegmentModel`
- `ElasticPerSegmentModel`
- `LinearMultiSegmentModel`
- `LinearPerSegmentModel`
- `SklearnMultiSegmentModel`
- `SklearnPerSegmentModel`

Neural and pretrained models:
- `etna.models.nn.RNNModel`
- `etna.models.nn.MLPModel`
- `etna.models.nn.DeepStateModel`
- `etna.models.nn.NBeatsGenericModel`
- `etna.models.nn.NBeatsInterpretableModel`
- `etna.models.nn.PatchTSTModel`
- `etna.models.nn.DeepARModel`
- `etna.models.nn.TFTModel`
- `etna.models.nn.ChronosModel`
- `etna.models.nn.ChronosBoltModel`
- `etna.models.nn.TimesFMModel`

The API reference also lists abstract model interfaces and DeepState state-space helpers such as `CompositeSSM`, `LevelSSM`, `LevelTrendSSM`, `SeasonalitySSM`, `DailySeasonalitySSM`, and `YearlySeasonalitySSM`. Treat them as model API components, not first-choice forecasting estimators, unless the user asks specifically for DeepState customization and official docs are loaded.

## Pipelines And Ensembles

Forecasting pipelines:
- `Pipeline`: applies transforms, fits one ETNA model, and forecasts the next `horizon` timestamps in one iteration.
- `AutoRegressivePipeline`: makes regressive models autoregressive; repeatedly applies transforms and predicts `step` values until `horizon` is covered.
- `HierarchicalPipeline`: documented in the API for hierarchical workflows.
- `FoldMask`: explicit temporal fold masks for backtesting.
- `assemble_pipelines`: helper listed in the API reference.

Ensembles:
- `DirectEnsemble`
- `StackingEnsemble`
- `VotingEnsemble`

Do not assume every model supports every pipeline, interval, exogenous, or saved-inference workflow. Check the model's API page for special constraints.

## Optional Dependency Extras

Minimum install is `pip install etna`. Optional extras documented by ETNA include:
- `prophet`: Prophet model.
- `torch`: neural network models.
- `statsforecast`: StatsForecast models.
- `auto`: AutoML utilities.
- `chronos`: Chronos-like pretrained models.
- `timesfm`: TimesFM pretrained models.
- `wandb`: Weights & Biases logger.
- `clearml`: ClearML logger with TensorBoard support.
- `classiciation`: time-series classification functionality; this spelling appears in the official installation page. It is outside this forecasting skill unless classification is explicitly requested.

## Prediction Intervals

Official interval wrappers:
- `NaiveVariancePredictionIntervals`: estimates quantiles from k-fold backtest residual variance.
- `ConformalPredictionIntervals`: conformal interval wrapper listed in the API and tutorial.
- `EmpiricalPredictionIntervals`: estimates intervals from empirical historical residuals.

Use `forecast(prediction_interval=True, quantiles=(...), n_folds=...)` where supported. Evaluate with `Coverage` and `Width` on backtest windows.

## Metrics

Official metric classes include:
- Error metrics: `MAE`, `MSE`, `RMSE`, `MSLE`, `MAPE`, `SMAPE`, `WAPE`, `MedAE`, `MaxDeviation`, `R2`, `Sign`, `MissingCounter`.
- Interval metrics: `Coverage`, `Width`.
- Helpers and enums: `Metric`, `MetricWithMissingHandling`, `MetricAggregationMode`, `MetricMissingMode`, `compute_metrics`.

Recommended defaults:
- Use `MAE` or `RMSE` for scale-dependent evaluation.
- Use `SMAPE` or `WAPE` for reporting across segments; still inspect low/zero actuals.
- Use `Coverage` and `Width` together for prediction intervals.
- Avoid `MAPE` when actual values can be zero or near zero.

## Plotting, Analysis, And Diagnostics

Useful documented analysis functions:
- Data/EDA: `ts.plot()`, `seasonal_plot`, `stl_plot`, `acf_plot`, `cross_corr_plot`, `distribution_plot`, `plot_periodogram`, `plot_correlation_matrix`, `plot_holidays`, `plot_imputation`.
- Forecast/backtest: `plot_forecast`, `plot_backtest`, `plot_backtest_interactive`, `plot_metric_per_segment`, `metric_per_segment_distribution_plot`.
- Diagnostics: `get_residuals`, `plot_residuals`, `prediction_actual_scatter_plot`, `qq_plot`.
- Interpretation: `plot_forecast_decomposition`, `plot_feature_relevance`, `get_model_relevance_table`, `get_statistics_relevance_table`.
- Anomaly tools are present in `etna.analysis`, but anomaly detection is outside this forecasting skill unless requested.

## Documented Limitations And Caveats

- ETNA's core forecasting container expects a `target` feature; multi-target endogenous forecasting beyond one target per segment is not documented in the browsed official docs.
- The regular `Pipeline` forecasts the next `horizon` points; if target lags are used as direct features, choose lags that are actually known for every future step. The get-started tutorial explicitly warns that with regular `Pipeline` lags should be `>= HORIZON`.
- `AutoRegressivePipeline` is documented for iterative recursive forecasting with smaller-step lags.
- Optional models require extras; missing packages raise import errors.
- Saved pipeline loading uses `dill`; official API warns that loading untrusted artifacts can execute arbitrary code.
- Apple M1/ARM install limitations are documented for CatBoost and numba/llvmlite dependencies.
- The stable docs consulted are ETNA 3.0.0. Re-check docs/source when working with a different installed version.
