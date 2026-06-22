---
name: anomaly-luminaire
description: Use Luminaire for time-series anomaly and outlier detection after validating ordered univariate data, including DataExploration profiling, HyperparameterOptimization, LADStructuralModel, LADFilteringModel, WindowDensityModel streaming/window detection, anomaly probabilities, confidence intervals, model freshness, and anti-leakage safeguards.
---

# Luminaire Anomaly Detection

Use this skill when the task is monitoring a single time series for future point outliers or sustained window anomalies with a mostly hands-off Zillow Luminaire workflow.

Important scope: Luminaire is not a general multivariate detector. Official examples expect a pandas `DataFrame` with a datetime index and a value column named `raw`. Run one model per series/entity unless you build external orchestration.

## Minimum Install

```bash
pip install luminaire
```

PyPI lists Luminaire `0.4.3`, Python `>=3.7`, and classifiers through Python 3.10. Source requirements pin practical ranges, including `numpy<=1.22.4`, `pandas<=2.0.3`, and `statsmodels<=0.13.5`; use a pinned environment for reproducibility.

## Data Contract

- Use one ordered time series at a time: datetime index plus numeric `raw` column.
- Batch outlier detection supports frequencies documented for `DataExploration`/batch models: `H`, `D`, `W`, `W-SUN`, `W-MON`, `W-TUE`, `W-WED`, `W-THU`, `W-FRI`, `W-SAT`.
- Streaming/window density supports default configuration for `S`, `T`, `15T`, `H`, `D`; other frequencies require manual `custom` configuration.
- Luminaire profiling can impute missing timestamps, average duplicate dates, log-transform, detect trend/change points, and truncate training data after recent shifts. Treat all of these as train-only operations.

Read `references/luminaire-data-workflow.md` before adapting CSVs, labels, multiple entities, or temporal validation.

## Core Patterns

Fully automatic batch outlier detection:

```python
import pandas as pd
from luminaire.exploration.data_exploration import DataExploration
from luminaire.optimization.hyperparameter_optimization import HyperparameterOptimization

data = pd.read_csv("series.csv", parse_dates=["time"]).set_index("time")
data = data.rename(columns={"value": "raw"})[["raw"]]

opt_config = HyperparameterOptimization(freq="D").run(data=data)
training_data, pre_prc = DataExploration(freq="D", **opt_config).profile(data)

model_class_name = opt_config["LuminaireModel"]
model_class = getattr(__import__("luminaire.model", fromlist=[""]), model_class_name)
model_object = model_class(hyper_params=opt_config, freq="D")
success, model_date, trained_model = model_object.train(data=training_data, **pre_prc)
score = trained_model.score(100, "2021-01-01")
```

Manual streaming/window anomaly detection:

```python
from luminaire.exploration.data_exploration import DataExploration
from luminaire.model.window_density import WindowDensityHyperParams, WindowDensityModel

config = WindowDensityHyperParams(freq="10T", detection_method="kldiv", window_length=36).params
data, pre_prc = DataExploration(**config).stream_profile(df=data)
config.update(pre_prc)
success, training_end, model = WindowDensityModel(hyper_params=config).train(data=data)
processed, _ = DataExploration(freq=model._params["freq"]).stream_profile(
    df=scoring_data, impute_only=True, impute_zero=True
)
score, scored_window = model.score(processed)
```

## Method Choice

- Use `HyperparameterOptimization` plus `DataExploration.profile` for hands-off batch point outlier monitoring.
- Use `LADStructuralModel` when data has useful temporal/seasonal signal; it supports AR/MA components, Fourier terms, and daily holiday exogenous features.
- Use `LADFilteringModel` for noisy series with weak predictive structure; score sequentially with the returned updated model.
- Use `WindowDensityModel` for high-frequency or streaming cases where sustained window shifts matter more than isolated points.
- Use `DataExploration.profile(..., impute_only=True)` only when you need Luminaire imputation without full profiling.

## Scoring and Metrics

- Batch `score(value, date)` returns fields such as `Prediction`, `StdErr`, `CILower`, `CIUpper`, `ConfLevel`, `IsAnomaly`, `IsAnomalyExtreme`, `AnomalyProbability`, `DownAnomalyProbability`, `UpAnomalyProbability`, and `ModelFreshness`.
- Filtering scores return `(scores, model_update)`; use the update for the next timestamp.
- Window density scores return an anomaly summary and the scored window.
- With labels, report precision, recall, F1 for anomaly class, PR-AUC/average precision, ROC-AUC when both classes exist, alert rate, false positives per period, and detection delay for window alerts.
- Luminaire docs do not document a dedicated plotting API; plot `raw`, predictions/intervals, anomaly probabilities, and alert windows with matplotlib or Plotly.

## Anti-Leakage Rules

- Never random split time-indexed monitoring data. Use chronological train/validation/test or rolling-origin evaluation.
- Fit `DataExploration`, imputation, log-transform choice, change-point truncation, `HyperparameterOptimization`, model class, thresholds, and window settings only on train/validation.
- Score only future timestamps/windows relative to the trained model date. Do not tune on the final test period.
- Holiday exogenous features are documented for daily structural modeling; use them only when known at score time.
- For panels, split each entity by time and fit separate models unless production explicitly shares configuration across series.
- For streaming windows, build scoring windows from data available at alert time; no centered windows for online detection.

## Common Errors

- Passing a wide multivariate table directly instead of one `raw` series per model.
- Leaving the timestamp as a normal feature rather than the DataFrame index.
- Running optimization on the full history and then evaluating on that same history.
- Ignoring `ModelFreshness`; Luminaire expires model objects when freshness exceeds `1`.
- Treating filtering model scores as independent; use `model_update` for sequential scoring.
- Using streaming `WindowDensityModel` for arbitrary frequencies without manual `custom` configuration.

## References

- Read `references/luminaire-api-map.md` for official classes, parameters, outputs, and model inventory.
- Read `references/luminaire-data-workflow.md` for data preparation, validation, panels, metrics, and leakage controls.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_luminaire_anomaly_input.py` to sanity-check CSV inputs before Luminaire profiling/training.

## Ready Checklist

- Task is point outlier detection or sustained window anomaly detection on ordered time-series data.
- Input has datetime index semantics and exactly one numeric `raw` signal per Luminaire model.
- Frequency and batch vs streaming mode are documented.
- Profiling, optimization, transforms, model training, and thresholds are fitted only on training windows.
- Metrics include anomaly-class behavior and alert volume, not only aggregate accuracy.
