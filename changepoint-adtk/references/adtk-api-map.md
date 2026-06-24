# ADTK API Map

Official scope: ADTK is an unsupervised/rule-based anomaly detection toolkit for time series. Use it for changepoint work only when anomalous time points or event intervals are acceptable changepoint candidates.

## Detector inventory

Univariate or per-column detectors:

- `ThresholdAD(low=None, high=None)`: fixed value limits.
- `QuantileAD(low=None, high=None)`: learn lower/upper quantiles from historical data.
- `InterQuartileRangeAD(c=3.0)`: learn IQR-based outlier limits.
- `GeneralizedESDTestAD(alpha=0.05)`: generalized ESD outlier test.
- `PersistAD(window=1, c=3.0, side="both", min_periods=None, agg="median")`: persistent deviation from near past; useful for temporary abrupt changes or spikes.
- `LevelShiftAD(window, c=6.0, side="both", min_periods=None)`: compares medians in adjacent windows and marks the boundary as a level-shift point.
- `VolatilityShiftAD(window, c=6.0, side="both", min_periods=None, agg="std")`: compares volatility measures in adjacent windows.
- `SeasonalAD(freq=None, side="both", c=3.0, trend=False)`: detects seasonal-pattern violations after classical seasonal decomposition.
- `AutoregressionAD(n_steps=1, step_size=1, regressor=None, c=3.0, side="both")`: detects anomalous autoregression relationships.
- `CustomizedDetector1D`: wraps a custom one-dimensional detector function.

Multivariate detectors:

- `MinClusterDetector(model)`: anomaly based on clustering historical data.
- `OutlierDetector(model)`: wraps outlier models over multivariate historical data.
- `RegressionAD(regressor, target, c=3.0, side="both")`: detects anomalous inter-series relationship around one target column.
- `PcaAD(k=1, c=5.0)`: PCA reconstruction style multivariate anomaly detection.
- `CustomizedDetectorHD`: wraps a custom high-dimensional detector function.

Transformers and aggregators that matter for changepoints:

- `RollingAggregate`: sliding-window statistics for pattern features.
- `DoubleRollingAggregate`: side-by-side window statistics used by persistence, level-shift, and volatility-shift workflows.
- `ClassicSeasonalDecomposition`: seasonal residual extraction.
- `Retrospect`: lagged/retrospective feature construction.
- `RegressionResidual`, `PcaProjection`, `PcaReconstruction`, `PcaReconstructionError`: multivariate residual/projection features.
- `OrAggregator`, `AndAggregator`: combine anomaly labels or events.
- `Pipeline`, `Pipenet`: connect transformers, detectors, and aggregators into models.

## Unified methods and outputs

- `fit(ts)` trains a detector on a `pandas.Series` or `pandas.DataFrame`. For many univariate detectors, a `DataFrame` means one detector per column.
- `detect(ts, return_list=False)` returns binary anomaly labels by default; `return_list=True` returns events.
- `fit_detect(ts, return_list=False)` trains and detects on the same data. Use only for training diagnostics or unsupervised full-history exploration, not final validation.
- `fit_predict` and `predict` are aliases available in the detector APIs.
- `score(ts, anomaly_true, scoring="recall", **kwargs)` detects and scores against true anomalies. Supported scoring values are `recall`, `precision`, `f1`, and `iou`.

When `return_list=True`, a univariate result is a list where each event is either a `pandas.Timestamp` or a `(Timestamp, Timestamp)` interval. A `DataFrame` may return a dict of event lists keyed by column.

## Parameter notes

- `window` may be an integer count or a time offset string where documented. For changepoint candidates, record whether it is a sample count or time duration.
- `side` is documented as `"both"`, `"positive"`, or `"negative"` for detectors that distinguish direction.
- `c` controls the outlier threshold scale inside several detectors. Tune it on validation data only.
- `agg` for `PersistAD` is documented as mean or median. `agg` for `VolatilityShiftAD` is documented as `std`, `iqr`, or `idr`.
- `SeasonalAD(freq=None)` can be left to infer in simple cases, but production skills should document the intended period.
- `RegressionAD` needs a target column name and a compatible regressor object.

## Metrics and plotting

ADTK metrics accept binary label `Series`/`DataFrame`, event lists, or dicts of event lists:

- `recall(y_true, y_pred, thresh=0.5)`
- `precision(y_true, y_pred, thresh=0.5)`
- `f1_score(y_true, y_pred, recall_thresh=0.5, precision_thresh=0.5)`
- `iou(y_true, y_pred)`

`adtk.visualization.plot` plots a `Series`/`DataFrame` with anomaly labels, event lists, or nested dicts. Use `anomaly_tag="span"` for intervals and `"marker"` for point candidates.

## Documented gaps

- No native optimal offline segmentation API like dynamic programming over a cost function.
- No documented supervised anomaly/changepoint training workflow; docs say alternative tools are needed for supervised formulations.
- No forecasting horizon or forecasting-style exogenous regressor interface. Multivariate columns are simultaneous signals.
- Online minimal-delay changepoint guarantees are not documented.
