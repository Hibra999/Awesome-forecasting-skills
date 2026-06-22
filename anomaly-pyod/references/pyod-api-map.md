# PyOD API Map

PyOD 3 exposes three use layers:

- Classic API: choose a detector class and call `fit`, `decision_function`, `predict`, `predict_proba`, and optional confidence/rejection methods.
- `ADEngine`: lifecycle orchestration for profiling, detector selection, comparison, running, analysis, and explanation.
- Agentic/MCP path: optional `od-expert` skill and MCP tools. Use only if the environment is configured for those tools.

## Shared detector API

For classic detectors:

- `fit(X, y=None)`: fit the detector. In unsupervised methods, `y` is ignored.
- `decision_function(X)`: raw anomaly scores; higher means more abnormal in PyOD convention.
- `predict(X, return_confidence=False)`: binary labels, where `0` is inlier and `1` is outlier.
- `predict_proba(X, method="linear", return_confidence=False)`: outlier probabilities from scores.
- `predict_confidence(X)`: per-sample confidence where supported.
- `predict_with_rejection(...)`: output `-2` for rejected uncertain predictions where supported.
- Fitted attributes include `decision_scores_`, `labels_`, and usually `threshold_`.

## Official detector inventory

Tabular and multimodal detectors/classes listed by official README/API pages:

- Probabilistic/statistical: `ECOD`, `ABOD`, `FastABOD`, `COPOD`, `MAD`, `SOS`, `QMCD`, `KDE`, `Sampling`, `GMM`, `HBOS`.
- Linear/subspace/covariance: `PCA`, `KPCA`, `MCD`, `CD`, `OCSVM`, `RGraph`, `ROD`.
- Distance, density, clustering, local: `KNN`, `LOF`, `CBLOF`, `COF`, `HDBSCAN`, `SOD`, `LOCI`, `LMDD`.
- Isolation/ensemble/scalable: `IForest`, `INNE`, `FeatureBagging`, `LODA`, `LSCP`, `SUOD`, `DIF`, `XGBOD`.
- Deep learning: `AutoEncoder`, `VAE`, `DeepSVDD`, `DevNet`, `LUNAR`, `DIF`, `MO_GAAL`, `SO_GAAL`.
- Embedding/multimodal: `EmbeddingOD`, `MultiModalOD`.

Time-series detectors:

- `TimeSeriesOD`: bridge that applies PyOD detectors to sliding windows.
- `MatrixProfile`: matrix profile via STOMP; official docs call it transductive.
- `SpectralResidual`: FFT saliency-style detector.
- `KShape`: k-Shape clustering detector.
- `SAND`: streaming subsequence detector with drift adaptation, marked experimental in the README.
- `LSTMAD`: LSTM prediction-error detector with Mahalanobis scoring.
- `AnomalyTransformer`: transformer detector, marked experimental in the README.

Graph detectors with `pyod[graph]`:

- `SCAN`, `Radar`, `ANOMALOUS`, `DOMINANT`, `AnomalyDAE`, `CoLA`, `CONAD`, `GUIDE`.

Score-combination utilities in `pyod.models.combination` include `average`, `maximization`, `median`, `majority_vote`, `aom`, and `moa`.

Threshold functions in `pyod.models.thresholds` include `AUCP`, `BOOT`, `CHAU`, `CLF`, `CLUST`, `CPD`, `DECOMP`, `DSN`, `EB`, `FGD`, `FILTER`, `FWFM`, `GESD`, `HIST`, `IQR`, `KARCH`, `MAD`, `MCST`, `META`, `MOLL`, `MTT`, `OCSVM`, `QMCD`, `REGR`, `VAE`, `WIND`, `YJ`, and `ZSCORE`.

## Inputs by modality

- Classic tabular: numeric array-like `X` of shape `(n_samples, n_features)`.
- Time series: numeric array-like shape `(n_timestamps,)` or `(n_timestamps, n_channels)`; pandas DataFrames and lists are documented as auto-converted.
- Graph: PyG `Data` object with `x` node features and `edge_index`, except `SCAN` can work without features.
- Text/image/audio: use `EmbeddingOD` or related multimodal APIs with optional dependencies and documented encoders.

## Metrics and utilities

- `pyod.utils.data.evaluate_print` prints ROC-AUC and precision @ rank n in examples.
- `pyod.utils.data.precision_n_scores` calculates Precision @ Rank n.
- `pyod.utils.data.generate_data` and `generate_data_clusters` create synthetic data for examples.
- Model persistence is documented with joblib or pickle.
- SUOD accelerates large heterogeneous detector ensembles.

## Documented limitations and caveats

- Labels are not used by unsupervised detectors during `fit`.
- `contamination` defines the threshold from training scores; choose it before final testing.
- `MatrixProfile` is transductive in the official time-series docs; use `decision_scores_` and `labels_` after `fit`.
- Graph detectors are described as transductive in v1 of the graph section.
- Optional modalities require optional extras and external dependencies.
- PyOD provides outlier scores and point labels. It does not replace forecasting, event segmentation, root-cause analysis, or business alert triage.
