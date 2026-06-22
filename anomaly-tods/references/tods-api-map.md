# TODS API Map

TODS is organized as a D3M-style primitive pipeline system. The high-level modules documented officially are:

- `tods.data_processing`
- `tods.timeseries_processing`
- `tods.feature_analysis`
- `tods.detection_algorithm`
- `tods.reinforcement`
- evaluation/common primitives registered in `tods/resources/.entry_points.ini`

## High-level helpers

README examples use:

- `generate_dataset(df, target_index=...)`
- `generate_problem(dataset, metric)`
- `evaluate_pipeline(dataset, pipeline, metric)`
- `schemas.load_default_pipeline()`
- `BruteForceSearch(problem_description=..., backend=...)`
- `search.search_fit(input_data=[dataset], time_limit=...)`
- `search.evaluate(best_pipeline).scores`

Hosted docs also show an older helper path `tods.utils.generate_dataset_problem`; verify installed version before using that import.

## D3M primitive workflow

Manual pipeline examples use:

- `d3m.metadata.pipeline.Pipeline`
- `PrimitiveStep`
- `d3m.index.get_primitive("d3m.primitives...")`
- `add_argument`, `add_output`, `add_hyperparameter`
- semantic types including `Attribute` and `TrueTarget`
- final `construct_predictions`

Detector primitives follow the D3M style:

- `set_training_data(inputs=...)`
- `fit(timeout=None, iterations=None)`
- `produce(inputs=...)` returns binary labels: `1` outlier, `0` normal.
- `produce_score(inputs=...)` returns outlier scores where implemented.
- fitted attributes often include `decision_scores_`, `threshold_`, and `labels_`.

## Data processing primitives

Official docs and entry points list:

- `DatasetToDataframe`
- `TimeIntervalTransform`
- `CategoricalToBinary`
- `ColumnFilter`
- `TimeStampValidation`
- `DuplicationValidation`
- `ContinuityValidation`
- `SKImputer`
- `ColumnParser`
- `ExtractColumnsBySemanticTypes`
- `ConstructPredictions`

Use these only inside train/validation folds when they learn or alter data.

## Time-series processing primitives

Official docs and entry points list:

- `SKAxiswiseScaler`
- `SKStandardScaler`
- `SKPowerTransformer`
- `SKQuantileTransformer`
- `MovingAverageTransform`
- `SimpleExponentialSmoothing`
- `HoltSmoothing`
- `HoltWintersExponentialSmoothing`
- `TimeSeriesSeasonalityTrendDecomposition`
- `SubsequenceSegmentation` in entry points

## Feature analysis primitives

Official docs and entry points list:

- `AutoCorrelation`
- `StatisticalMean`, `StatisticalMedian`, `StatisticalGmean`, `StatisticalHmean`
- `StatisticalAbsEnergy`, `StatisticalAbsSum`, `StatisticalMaximum`, `StatisticalMinimum`
- `StatisticalMeanAbs`, `StatisticalMeanAbsTemporalDerivative`, `StatisticalMeanTemporalDerivative`
- `StatisticalMedianAbsoluteDeviation`, `StatisticalKurtosis`, `StatisticalSkew`
- `StatisticalStd`, `StatisticalVar`, `StatisticalVariation`, `StatisticalVecSum`
- `StatisticalWillisonAmplitude`, `StatisticalZeroCrossing`
- `SpectralResidualTransform`, `FastFourierTransform`, `DiscreteCosineTransform`, `WaveletTransform`
- `NonNegativeMatrixFactorization`, `BKFilter`, `HPFilter`, `SKTruncatedSVD`, `TRMF`

## Detection primitives

Official hosted detection docs list:

- `AutoRegODetect`
- `DeepLog`
- `KDiscordODetect`
- `LSTMODetect`
- `MatrixProfile`
- `PCAODetect`
- `PyodABOD`
- `PyodAE`
- `PyodCBLOF`
- `PyodCOF`
- `PyodHBOS`
- `PyodIsolationForest`
- `PyodKNN`
- `PyodLODA`
- `PyodLOF`
- `PyodMoGaal`
- `PyodOCSVM`
- `PyodSOD`
- `PyodSoGaal`
- `PyodVAE`
- `Telemanom`
- `UODBasePrimitive`

The official entry point file additionally registers:

- `DAGMM`
- `SystemWiseDetection`
- `Ensemble`

README says TODS includes all point-wise detection algorithms supported by PyOD, but the explicit public primitive list above is the safer support claim.

## Reinforcement and evaluation primitives

- `RuleBasedFilter` is the documented reinforcement primitive.
- Evaluation/common entry points include `FixedSplit`, `KFoldSplit`, `KFoldSplitTimeseries`, `NoSplit`, `TrainScoreSplit`, `RedactColumns`, `CSVReader`, and `Denormalize`.

## Documented gaps and caveats

- Hosted docs are `0.0.1`; PyPI is `0.0.2`; README examples have newer import paths than the hosted docs.
- PyPI package is old and sparse; README recommends local source install.
- Modern Python/TensorFlow compatibility is not documented. Source `setup.py` pins legacy dependencies including `tensorflow==2.4`, `keras==2.4.0`, `numpy<=1.21.2`, `pyod==1.0.5`, and `statsmodels==0.11.1`.
- TODS is focused on time-series outlier detection pipelines; it does not provide forecasting outputs.
