# TSFEL API Map

Use exact API/source names. The core extractor is plural: `time_series_features_extractor`.

## Core Functions

| API | Purpose | Notes |
| --- | --- | --- |
| `tsfel.get_features_by_domain()` | Load predefined feature configuration. | Default docs describe temporal, statistical, and spectral sets. |
| `tsfel.get_features_by_domain("statistical")` | Load statistical domain config. | Descriptive statistics and distribution summaries. |
| `tsfel.get_features_by_domain("temporal")` | Load temporal domain config. | Time-order-sensitive features. |
| `tsfel.get_features_by_domain("spectral")` | Load spectral domain config. | Requires correct sampling frequency for meaningful values. |
| `tsfel.get_features_by_domain("fractal")` | Load fractal domain config. | Usually for longer signals; disabled in default config files per README/docs. |
| `tsfel.time_series_features_extractor(config, timeseries, fs=None, window_size=None, overlap=0, verbose=1, **kwargs)` | Extract features from univariate or multivariate in-memory time series. | Returns pandas `DataFrame`. |
| `tsfel.dataset_features_extractor(main_directory, feat_dict, verbose=1, **kwargs)` | Extract features from files under a dataset root. | Saves feature CSV files. |
| `tsfel.feature_extraction.calc_features.calc_window_features(config, window, fs, ...)` | Extract features from one univariate or multivariate window. | Lower-level utility. |
| `load_combined_feature_modules(features_path=None)` | Load TSFEL feature functions plus optional user module. | User-defined functions can override same-name functions. |

## Important Parameters

- `fs`: sampling frequency. Use the real value for spectral features.
- `window_size`: number of samples per window.
- `overlap`: percentage overlap between consecutive windows, between 0 and 1.
- `n_jobs`: parallelism; `None` means one job unless inside a joblib backend, and `-1` uses all processors.
- `verbose`: `0` silent, `1` progress bar.
- `features_path`: path to custom feature module.
- `header_names`: names for each signal column/window.
- `search_criteria`, `time_unit`, `resample_rate`, `pre_process`, `output_directory`: dataset extraction parameters.

Note: the API reference names `resampling_rate`, but the Get Started example and source use `resample_rate`.

## Feature Domains

Official README/docs list more than 65 features across:

- Statistical: examples include absolute energy, entropy, histogram, interquartile range, kurtosis, max, mean, median, min, RMS, skewness, standard deviation, variance.
- Temporal: examples include area under the curve, autocorrelation, centroid, mean/median differences, turning points, signal distance, slope, zero crossing rate, neighbourhood peaks.
- Spectral: examples include FFT/spectrogram coefficients, fundamental frequency, LPCC, MFCC, max power spectrum, median frequency, spectral centroid/entropy/kurtosis/skewness/slope/spread, wavelet features.
- Fractal: DFA, Higuchi fractal dimension, Hurst exponent, maximum fractal length, multiscale entropy, Petrosian fractal dimension.

Use the official feature list for complete installed-version names.

## Config Files

Feature extraction is configured via dictionary/JSON. Individual features can be enabled or disabled with `use: yes/no`; some features include a `parameters` field for settings such as histogram bins or wavelet parameters.

Save the config used for experiments so feature matrices are reproducible.

## Custom Features

Personalised features require:

- A Python file containing feature functions.
- `@set_domain("domain", "...")` annotation using available domains.
- Metadata added to a JSON feature file.
- Passing `features_path` to extraction.

## Documented Gaps

- TSFEL is not a classifier, regressor, forecaster, or clustering model.
- TSFEL does not document supervised feature selection as a core API.
- TSFEL does not document sklearn transformer wrappers as core API.
- TSFEL does not handle unevenly sampled data by default.
- TSFEL does not document a dedicated plotting API for extracted features.
