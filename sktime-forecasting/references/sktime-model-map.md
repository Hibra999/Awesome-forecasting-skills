# sktime Forecasting Model Map

Use this reference when selecting a forecaster or verifying whether a model name is officially documented. The installed environment can differ from the docs because many forecasters require soft dependencies; verify with:

```python
from sktime.registry import all_estimators, all_tags

all_estimators("forecaster", as_dataframe=True)
all_estimators(
    "forecaster",
    filter_tags={"capability:pred_int": True},
    as_dataframe=True,
)
all_tags(estimator_types="forecaster")
```

Important tags include `scitype:y`, `y_inner_mtype`, `X_inner_mtype`, `ignores-exogeneous-X`, `X-y-must-have-same-index`, `requires-fh-in-fit`, `capability:pred_int`, and `capability:pred_var`.

## Composition, Reduction, and Utilities

- Base/API: `BaseForecaster`, `ForecastingHorizon`.
- Pipelines/composition: `make_pipeline`, `TransformedTargetForecaster`, `ForecastingPipeline`, `ColumnEnsembleForecaster`, `MultiplexForecaster`, `ForecastX`, `ForecastByLevel`, `TransformSelectForecaster`, `GroupbyCategoryForecaster`, `HierarchyEnsembleForecaster`, `Permute`, `FhPlexForecaster`, `IgnoreX`, `FallbackForecaster`, `YfromX`.
- Reduction to regression: `make_reduction`, `DirectTabularRegressionForecaster`, `DirectTimeSeriesRegressionForecaster`, `MultioutputTabularRegressionForecaster`, `MultioutputTimeSeriesRegressionForecaster`, `RecursiveTabularRegressionForecaster`, `RecursiveTimeSeriesRegressionForecaster`, `DirRecTabularRegressionForecaster`, `DirRecTimeSeriesRegressionForecaster`, `DirectReductionForecaster`, `RecursiveReductionForecaster`, `SkforecastAutoreg`, `SkforecastRecursive`, `DartsRegressionModel`, `DartsLinearRegressionModel`, `DartsXGBModel`.
- Model selection/tuning: `ForecastingGridSearchCV`, `ForecastingRandomizedSearchCV`, `ForecastingOptCV`, `ForecastingSkoptSearchCV`, `ForecastingOptunaSearchCV`.
- Evaluation: `evaluate`.

## Baselines, Intervals, Calibration

- Baselines/known values: `NaiveForecaster`, `ForecastKnownValues`, `DummyGlobalForecaster`.
- Interval/probability wrappers and bootstrap-style helpers: `SquaringResiduals`, `NaiveVariance`, `ConformalIntervals`, `BaggingForecaster`, `EnbPIForecaster`.
- Bias adjustment: `BoxCoxBiasAdjustedForecaster`.

## Classical and Statistical Forecasters

- Trend: `TrendForecaster`, `PolynomialTrendForecaster`, `STLForecaster`, `CurveFitForecaster`, `ProphetPiecewiseLinearTrendForecaster`, `SplineTrendForecaster`, `StatsForecastMSTL`.
- Exponential smoothing/theta: `ExponentialSmoothing`, `AutoETS`, `StatsForecastAutoETS`, `StatsForecastAutoCES`, `ThetaForecaster`, `ThetaModularForecaster`, `StatsForecastAutoTheta`.
- AR/MA and multivariate systems: `AutoREG`, `ARIMA`, `StatsModelsARIMA`, `SARIMAX`, `VAR`, `VARReduce`, `VARMAX`, `VECM`.
- Auto-ARIMA and ARAR: `AutoARIMA`, `StatsForecastAutoARIMA`, `ARARForecaster`.
- ARCH/ARDL family: `StatsForecastARCH`, `StatsForecastGARCH`, `ARCH`, `ARDL`.
- Structural and external package wrappers: `BATS`, `TBATS`, `StatsForecastAutoTBATS`, `Prophet`, `Prophetverse`, `HierarchicalProphet`, `UnobservedComponents`, `DynamicFactor`, `GreykiteForecaster`.
- Intermittent demand: `Croston`, `StatsForecastADIDA`, `TSB`.

## Deep, Foundation, and Large Model Forecasters

- Deep learning: `LTSFLinearForecaster`, `LTSFDLinearForecaster`, `LTSFNLinearForecaster`, `LTSFTransformerForecaster`, `XLSTMForecaster`, `SCINetForecaster`, `ConvTimeNetForecaster`, `CINNForecaster`, `NeuralForecastRNN`, `NeuralForecastLSTM`, `NeuralForecastTCN`, `NeuralForecastGRU`, `NeuralForecastDilatedRNN`, `PytorchForecastingTFT`, `PytorchForecastingDeepAR`, `PytorchForecastingNHiTS`, `PytorchForecastingNBeats`, `PyKANForecaster`, `RBFForecaster`, `ESRNNForecaster`.
- Pre-trained/foundation: `ChronosForecaster`, `Chronos2Forecaster`, `FalconTSTForecaster`, `FlowStateForecaster`, `HFTransformersForecaster`, `KronosForecaster`, `LagLlamaForecaster`, `MantisForecaster`, `MOIRAIForecaster`, `MomentFMForecaster`, `PatchTSMixerForecaster`, `PatchTSTForecaster`, `TimeLLMForecaster`, `TimeMoEForecaster`, `TimerForecaster`, `TimerS1Forecaster`, `TimesFMForecaster`, `TimesFM2Forecaster`, `TinyTimeMixerForecaster`, `TotoForecaster`.

## Ensembles, Hierarchical, Causal, Online, Adapters

- Ensembles/stacking: `EnsembleForecaster`, `AutoEnsembleForecaster`, `StackingForecaster`, `ResidualBoostingForecaster`, `AutoTS`, `MAPAForecaster`.
- Causal forecasting: `DoubleMLForecaster`.
- Hierarchical reconciliation: `ReconcilerForecaster`.
- Online/stream: `OnlineEnsembleForecaster`, `NormalHedgeEnsemble`, `NNLSEnsemble`, `UpdateEvery`, `UpdateRefitsEvery`, `DontUpdate`.
- Framework adapter: `HCrystalBallAdapter`.
- Agentic: `AutoResearchForecaster`.

## Selection Rules

- Choose a simple baseline first; complex models must beat it under the same temporal validation.
- Prefer native statistical models when the data is a small/medium univariate series with explainable trend/seasonality.
- Prefer reduction forecasters when a tabular regressor, lag-window setup, or many engineered covariates is the main modeling idea.
- Prefer panel/global/hierarchical forecasters only when the model or wrapper documents support for the relevant `scitype:y`; otherwise sktime may vectorize and fit one forecaster per instance/variable.
- Prefer probabilistic methods only when tags or docs confirm support. Positive tags on composites mean the logic is implemented when the relevant components support it.
- Do not claim a model is available merely because it exists in a third-party library; use the documented sktime class or adapter name.

## Documented Limits

- `sktime` is designed for in-memory computation on a single machine, not distributed training.
- Core install excludes many soft dependencies. Missing dependency errors are expected when using wrappers without installing the wrapped library.
- Some capabilities depend on estimator hyperparameters and installed extras; tags can reflect general potential support rather than a specific configured instance.
- Residual diagnostics are not uniform across all forecasters; use documented wrapped-model APIs or holdout residual analysis.
