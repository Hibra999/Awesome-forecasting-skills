# Data Preparation Library Selection

Load this reference when deciding which dependencies to use. Prefer the smallest set that solves the task clearly and reproducibly.

## Recommended Libraries

- **pandas**: default choice for moderate tabular time-series preparation. Use for datetime parsing, timezone handling, indexing, resampling, joins, missing values, rolling windows, and diagnostics. Official docs: https://pandas.pydata.org/docs/user_guide/timeseries.html
- **numpy**: use for numeric validation, quantiles, vectorized masks, and deterministic array operations. Usually paired with pandas.
- **polars**: use for larger-than-memory-pressure tabular workflows, lazy scans, fast group-by/dynamic windows, and Arrow-native pipelines. Official docs: https://docs.pola.rs/user-guide/transformations/time-series/rolling/
- **pyarrow**: use for efficient Parquet/Arrow interchange and schema-preserving storage between preparation and modeling tools.
- **scikit-learn**: use for train-only preprocessing pipelines, scalers, encoders, imputers, and `TimeSeriesSplit` when data is already ordered and equally spaced. Official docs: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- **sktime**: use when a task benefits from forecasting-aware splitters, forecasting horizons, transformations, and a consistent time-series interface. Official docs: https://www.sktime.net/en/stable/examples/01_forecasting.html
- **statsmodels**: use for classical diagnostics such as decomposition, autocorrelation checks, stationarity tests, and robust exploratory baselines. Keep it out of simple cleaning pipelines unless diagnostics require it.
- **holidays**: use for country/market holiday calendars when future holiday indicators are needed and supported by the domain.
- **feature-engine**: use only when its scikit-learn-compatible transformers reduce custom preprocessing code and can be fit strictly on train.

## Dependency Guidance

- Start with pandas plus numpy for most skills and scripts.
- Add polars when file size or group-by volume makes pandas a bottleneck.
- Add scikit-learn when preprocessing must be packaged as fit/transform components.
- Add sktime when the workflow needs forecasting-native splitters or horizon semantics.
- Add statsmodels for diagnostics, not as a default data-cleaning dependency.
- Add holidays only when calendar effects are expected and the jurisdiction is known.
- Avoid adding library-specific modeling dependencies to this base skill.

## Notes for Library-Specific Skills

Before using Prophet, Darts, StatsForecast, NeuralForecast, Nixtla, GluonTS, sktime, or another modeling library:

- Convert the prepared contract to that library's required schema.
- Preserve the original temporal split cutoffs.
- Reuse known-future, observed-past, and static covariate classifications.
- Re-run leakage checks after library-specific feature generation or dataset conversion.
