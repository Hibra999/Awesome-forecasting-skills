---
name: changepoint-greykite
description: Use LinkedIn Greykite for offline long-term changepoint analysis after validating prepared time-series data, including pandas time/value inputs, ChangepointDetector adaptive-lasso trend changepoints, seasonality changepoints, Silverkite changepoints configuration, level-shift regressors, forecast backtests, plotting, parameter tuning, and anti-leakage safeguards.
---

# Greykite Changepoints

Use this skill after time-series data preparation when the task is to detect or model long-term trend, seasonality, or level-shift changes with Greykite.

Do not present Greykite as a general offline segmentation library or online minimal-delay detector. Official docs frame changepoints mainly as Silverkite/forecasting components plus `ChangepointDetector` and `ShiftDetection`.

## Minimum Install

```bash
pip install greykite
```

PyPI lists `greykite 1.1.0` as latest, released February 20, 2025, requiring Python `>=3.10`. The public docs page still labels `1.0.0` as latest documentation; verify installed APIs when pinning.

## Data Contract

- Use a pandas `DataFrame` with a parseable time column and one numeric target column.
- Sort by time, deduplicate timestamps, document frequency and gaps, and keep a temporal train/validation/test split.
- `ChangepointDetector` is documented for one observed value column. For panels or multiple metrics, loop per entity/metric.
- Missing target values reduce usable observations. The detector raises an error when fewer than 5 non-null rows remain.
- For forecasting integration, configure `MetadataParam(time_col=..., value_col=..., freq=...)` and pass changepoints through `ModelComponentsParam(changepoints=...)`.

Read `references/greykite-changepoint-data.md` before adapting panels, labels, custom dates, or forecast validation.

## Core Patterns

Standalone trend changepoints:

```python
from greykite.algo.changepoint.adalasso.changepoint_detector import ChangepointDetector

detector = ChangepointDetector()
res = detector.find_trend_changepoints(
    df=df,
    time_col="ts",
    value_col="y",
    resample_freq="7D",
    regularization_strength=0.5,
    potential_changepoint_n=25,
    no_changepoint_proportion_from_end=0.2,
)
trend_changepoints = res["trend_changepoints"]
fig = detector.plot(plot=False)
```

Seasonality changepoints after trend review:

```python
res = detector.find_seasonality_changepoints(
    df=df,
    time_col="ts",
    value_col="y",
    regularization_strength=0.4,
    no_changepoint_proportion_from_end=0.2,
)
seasonality_changepoints = res["seasonality_changepoints"]
```

Silverkite forecast with automatic changepoints:

```python
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum

config = ForecastConfig(
    model_template=ModelTemplateEnum.SILVERKITE.name,
    forecast_horizon=365,
    metadata_param=MetadataParam(time_col="ts", value_col="y", freq="D"),
    model_components_param=ModelComponentsParam(
        changepoints={
            "changepoints_dict": {
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.5,
                "potential_changepoint_n": 25,
                "no_changepoint_proportion_from_end": 0.2,
            }
        }
    ),
)
result = Forecaster().run_forecast_config(df=df, config=config)
```

## Method Choice

- `ChangepointDetector.find_trend_changepoints`: adaptive-lasso trend changepoint detection with pre-aggregation, yearly seasonality terms, regularization, and minimum-distance filtering.
- `ChangepointDetector.find_seasonality_changepoints`: detects changes in seasonality magnitude/shape by component after removing trend.
- `model_components.changepoints["changepoints_dict"]`: Silverkite trend changepoint config with `method` `"uniform"`, `"custom"`, or `"auto"`.
- `model_components.changepoints["seasonality_changepoints_dict"]`: Silverkite seasonality changepoint config.
- `ShiftDetection.detect`: z-score based level-shift detection that creates `ctp_*` regressors for Silverkite. Treat it as level-shift regressor generation, not generic segmentation.
- Prophet changepoints are available through Greykite's `PROPHET` template, but use `changepoint-prophet` for Prophet-specific work.

Read `references/greykite-changepoint-api.md` before selecting parameters or claiming support.

## Validation and Metrics

- For changepoint labels, score detected dates externally with tolerance windows: precision, recall, F1, false positives, and absolute/median delay.
- For forecasting impact, use Greykite temporal backtest/CV via `EvaluationPeriodParam` and compare MAPE, RMSE, MAE-like metrics, coverage, and interval width where configured.
- Tune `regularization_strength`, potential changepoint spacing/count, no-changepoint end buffers, seasonality components, and level-shift z-score only on validation periods.
- Plot with `ChangepointDetector.plot(plot=False)`, `ShiftDetection.plot_level_shift()`, `result.model[-1].plot_trend_changepoint_detection(...)`, `backtest.plot()`, and `backtest.plot_components()`.

## Anti-Leakage Rules

- Never random split. Define chronological train, validation, and final test periods before detection, scaling, anomaly handling, or parameter tuning.
- Detect changepoints only on data available at that cutoff. Do not use full-history detected dates to explain earlier operational decisions.
- Keep `no_changepoint_distance_from_end` or `no_changepoint_proportion_from_end` so the final segment has enough data for estimation and backtest validation.
- Add manual `dates` only from domain knowledge available at the training cutoff, not from final test inspection.
- Fit anomaly adjustments, regressors, level-shift columns, and seasonality/trend choices inside each train/CV fold.
- For panels, split and detect per entity; do not pool future behavior from other entities unless that pooling is explicitly part of the production policy.

## Common Errors

- Treating Greykite changepoints as optimal offline segmentation.
- Placing too many Silverkite changepoints; docs recommend at most 3 for non-auto Silverkite unless interactions are controlled.
- Putting changepoints too close to the end of the data.
- Using `regularization_strength` outside `[0, 1]`; larger values mean fewer changepoints.
- Using unsupported frequency units such as `W`, `M`, or `Y` for distance parameters that Greykite validates as at most day-level units.
- Running seasonality changepoint detection before reviewing trend changepoints on the same data.
- Forgetting to map detected dates back to the original entity, frequency, and validation fold.

## References

- Read `references/greykite-changepoint-api.md` for exact APIs, parameters, outputs, plotting, and limitations.
- Read `references/greykite-changepoint-data.md` for data format, panels, labels, validation, and leakage.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_greykite_changepoints.py` to sanity-check CSV input and common detector settings.

## Ready Checklist

- Data is chronological, one-target pandas format with documented frequency and gaps.
- The task is long-term trend/seasonality/level-shift analysis, not online detection.
- Method and parameters match the documented Greykite API.
- End buffers and validation windows leave enough post-changepoint data.
- Metrics and plots are computed on temporal validation/test periods without leakage.
