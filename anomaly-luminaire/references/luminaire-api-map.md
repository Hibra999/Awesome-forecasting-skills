# Luminaire API Map

Use this reference when selecting Luminaire classes and building code from official docs.

## Installation and package metadata

- PyPI package: `luminaire`
- Latest PyPI version observed: `0.4.3`, released January 31, 2024.
- Python requirement from PyPI/setup.py: `>=3.7`; classifiers list Python 3.7, 3.8, 3.9, and 3.10.
- Source `requirements.txt` includes bounded dependencies: `numpy>=1.17.5, <=1.22.4`, `pandas>=0.25.3, <=2.0.3`, `statsmodels>=0.13.0, <=0.13.5`, plus `bayescd`, `changepy`, `hyperopt`, `pykalman`, `scipy`, `scikit-learn`, and `decorator`.

## Data exploration and profiling

Import:

```python
from luminaire.exploration.data_exploration import DataExploration
```

Documented constructor:

```python
DataExploration(
    freq="D",
    min_ts_mean=None,
    fill_rate=None,
    sig_level=0.05,
    min_ts_length=None,
    max_ts_length=None,
    is_log_transformed=None,
    data_shift_truncate=True,
    min_changepoint_padding_length=None,
    change_point_threshold=2,
    window_length=None,
    *args,
    **kwargs,
)
```

Methods documented:

- `add_missing_index(df=None, freq=None)`: reindexes missing dates; docs note duplicate dates are averaged.
- `kf_naive_outlier_detection(input_series, idx_position)`: returns an anomaly flag for an index position.
- `profile(df, impute_only=False, **kwargs)`: performs preprocessing/profiling before optimization or model training; returns `(preprocessed_dataframe, summary_dict)`.
- `stream_profile(df, ...)`: prepares streaming/window data; examples use `stream_profile(df=data)` and `stream_profile(df=scoring_data, impute_only=True, impute_zero=True)`.

Profiling output may include `success`, `trend_change_list`, `change_point_list`, `is_log_transformed`, `min_ts_mean`, `ts_start`, and `ts_end`.

## Configuration optimization

Import:

```python
from luminaire.optimization.hyperparameter_optimization import HyperparameterOptimization
```

Documented constructor:

```python
HyperparameterOptimization(
    freq,
    detection_type="OutlierDetection",
    min_ts_mean=None,
    max_ts_length=None,
    min_ts_length=None,
    scoring_length=None,
    random_state=None,
    **kwargs,
)
```

Docs state only batch `OutlierDetection` is currently supported for `detection_type`.

Documented method:

- `run(data, max_evals=50)`: returns optimal hyperparameters. Tutorial examples call `hopt_obj.run(data=data)`.

Common output keys include `LuminaireModel`, `data_shift_truncate`, `fill_rate`, `include_holidays_exog`, `is_log_transformed`, `max_ft_freq`, `p`, and `q`.

## Batch outlier models

Structural model import:

```python
from luminaire.model.lad_structural import LADStructuralHyperParams, LADStructuralModel
```

Documented hyperparameters:

- `include_holidays_exog`: include holidays as exogenous variables in regression.
- `p`: AR component order.
- `q`: MA component order.
- `is_log_transformed`: log-transform input unless negatives are present.
- `max_ft_freq`: maximum Fourier frequency order.

`LADStructuralModel(hyper_params, freq, min_ts_length=None, max_ts_length=None, min_ts_mean=None, min_ts_mean_window=None, **kwargs)` is for periodic/correlated series and can run validation during training by setting `validation=True` in `train(...)`; docs say this validation flag is only available for `LADStructuralModel`.

Filtering model import:

```python
from luminaire.model.lad_filtering import LADFilteringHyperParams, LADFilteringModel
```

Documented hyperparameter:

- `is_log_transformed`

`LADFilteringModel(hyper_params, freq, min_ts_length=None, max_ts_length=None, **kwargs)` is a Markovian state-space/Kalman-filter model for noisy series. Its `score(...)` returns both a score dictionary and an updated model object for sequential scoring.

## Holiday exogenous features

Import:

```python
from luminaire.model.model_utils import LADHolidays
```

Docs describe `LADHolidays(name=None, holiday_rules=None)` as generating holiday calendars for batch outlier detection. README/tutorial says holiday exogenous features are currently supported for daily data only.

## Streaming/window density model

Import:

```python
from luminaire.model.window_density import WindowDensityHyperParams, WindowDensityModel
```

Documented hyperparameters:

- `freq=None`
- `max_missing_train_prop=0.1`
- `is_log_transformed=False`
- `baseline_type="aggregated"` or `"last_window"`
- `detection_method=None`, `"kldiv"`, or `"sign_test"`
- `min_window_length=None`
- `max_window_length=None`
- `window_length=None`
- `detrend_method="modeling"`, `"ma"`, or `"diff"`

Docs recommend `kldiv` for high-frequency data such as seconds/minutes and `sign_test` for lower frequency such as hourly/daily. Default configuration is documented for `S`, `T`, `15T`, `H`, and `D`; other frequency types should be specified as `custom` and configured manually.

`WindowDensityModel.train(data)` returns `(success, training_timestamp, model)`. `WindowDensityModel.score(data)` returns an anomaly summary; examples show `score, scored_window = model.score(processed_data)`.

## Scoring outputs

Batch structural examples include:

- `Success`
- `IsLogTransformed`
- `AdjustedActual`
- `Prediction`
- `StdErr` or `PredStdErr`
- `CILower`
- `CIUpper`
- `ConfLevel`
- `ExogenousHolidays`
- `IsAnomaly`
- `IsAnomalyExtreme`
- `AnomalyProbability`
- `DownAnomalyProbability`
- `UpAnomalyProbability`
- `ModelFreshness`

Window density examples include:

- `Success`
- `ConfLevel`
- `IsAnomaly`
- `AnomalyProbability`

Docs state Luminaire has built-in thresholds at `0.9` for mild anomalies and `0.999` for extreme anomalies, while users can set their own anomaly threshold using anomaly probabilities.

## Documented limitations

- Luminaire docs focus on univariate time-series monitoring with one `raw` column.
- Batch frequencies are limited to documented pandas frequency aliases.
- Holiday exogenous features are documented only for daily structural models.
- `HyperparameterOptimization.detection_type` currently supports only batch `OutlierDetection`.
- No dedicated plotting API is documented.
- Modern Python versions beyond 3.10 are not advertised in PyPI classifiers.
