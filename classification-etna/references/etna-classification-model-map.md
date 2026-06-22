# ETNA Classification Model Map

ETNA classification is in `etna.experimental.classification`; the docs warn that the architecture and APIs can change.

## Official Classes

| Component | Import | Role |
| --- | --- | --- |
| `TimeSeriesBinaryClassifier` | `from etna.experimental.classification import TimeSeriesBinaryClassifier` | Binary classifier wrapper around a feature extractor and sklearn-compatible classifier. |
| `TSFreshFeatureExtractor` | `from etna.experimental.classification.feature_extraction import TSFreshFeatureExtractor` | Extracts tsfresh features from each 1D series. |
| `WEASELFeatureExtractor` | `from etna.experimental.classification.feature_extraction import WEASELFeatureExtractor` | Extracts WEASEL symbolic bag-of-pattern features using pyts internals. |
| `PredictabilityAnalyzer` | `from etna.experimental.classification import PredictabilityAnalyzer` | Specialized binary classifier for whether ETNA segments are forecastable under a quality threshold. |

## Classifier Behavior

- `TimeSeriesBinaryClassifier` accepts labels `0` and `1` only.
- It calls `feature_extractor.fit_transform(x, y)` during `fit`, then `classifier.fit(features, y)`.
- `predict(x)` thresholds the positive-class probability.
- `predict_proba(x)` returns only the positive-class probability vector.
- `masked_crossval_score(x, y, mask)` refits the extractor and classifier for each unique fold value in `mask`.
- Built-in CV metrics are macro precision, macro recall, macro fscore, and ROC-AUC.

## Feature Extractors

- `TSFreshFeatureExtractor(default_fc_parameters=None, fill_na_value=-100, n_jobs=1, **kwargs)` passes parameters to `tsfresh.extract_features` and fills generated NaNs.
- `WEASELFeatureExtractor(padding_value, word_size=4, ngram_range=(1, 2), n_bins=4, window_sizes=None, window_steps=None, anova=True, drop_sum=True, norm_mean=True, norm_std=True, strategy="entropy", chi2_threshold=2, sparse=True, alphabet=None)` uses symbolic Fourier approximation, count vectorization, and chi-square feature selection.
- `WEASELFeatureExtractor` handles shorter transform-time series by padding according to train-fitted window settings; `padding_value` can be numeric or `"back_fill"`.

## Predictability Analyzer

- `PredictabilityAnalyzer.get_series_from_dataset(ts)` extracts sorted segment target arrays from an ETNA `TSDataset` and crops NaNs.
- `analyze_predictability(ts)` returns `{segment: 0_or_1}`.
- `get_available_models()` returns `["weasel", "tsfresh", "tsfresh_min"]`.
- `download_model(model_name, dataset_freq, path)` downloads a pickled pretrained analyzer from the official ETNA bucket.
- `load(path)` uses pickle; never load untrusted or tampered files.

## Documented Gaps

- No documented multiclass wrapper.
- No documented native multivariate or channel-wise input.
- No documented built-in distance, shapelet, ROCKET, deep-learning, or ensemble TSC classifier families in `etna.experimental.classification`.
- No documented sklearn `Pipeline` integration for ETNA feature extractors; use `masked_crossval_score` or explicit fold loops to avoid leakage.
