---
name: etna-forecasting
description: Use ETNA 3.0 forecasting after forecasting-data-prep, including TSDataset long/wide pandas data, single or multiple segments, exogenous regressors with df_exog and known_future, Pipeline and AutoRegressivePipeline workflows, statistical/ML/neural/pretrained models, ensembles, hierarchical pipelines, prediction intervals, temporal backtesting, metrics, plotting, residual diagnostics, and anti-leakage safeguards.
---

# ETNA Forecasting

Use this skill after `forecasting-data-prep`. ETNA is best when the user needs one unified Python workflow for multi-segment time-series forecasting with transforms, feature generation, model pipelines, smart backtesting, model combinations, hierarchical series, and prediction intervals.

Do not use ETNA as a generic multi-target VAR library unless official docs for that workflow are supplied. `TSDataset` expects one `target` feature per segment; represent independent targets as separate `segment`s or build separate workflows.

## Minimum Install

```bash
pip install etna
pip install "etna[all]"
```

Install extras only when needed: `etna[prophet]`, `etna[torch]`, `etna[statsforecast]`, `etna[auto]`, `etna[chronos]`, `etna[timesfm]`, `etna[wandb]`, or `etna[clearml]`. Missing extras raise import errors for optional models.

## Data Contract

- Start from a `forecasting-data-prep` contract: sorted timestamps, segment IDs, target, frequency, horizon, temporal cutoffs, known-future covariates, and leakage notes.
- Build `TSDataset(df, freq=...)` from pandas long or wide format. Long format has `timestamp`, `segment`, and at least one feature; forecasting data must include `target`.
- Use `freq` as a pandas offset alias or `DateOffset`; use `None` only for integer timestamps.
- Use one segment for univariate forecasting and multiple segment IDs for panel/multiple-series forecasting.
- Use `df_exog=` for exogenous features and `known_future=` only for covariates that are available for every future timestamp at prediction time.
- Use `TSDataset.create_from_misaligned` only when you intentionally need ETNA's documented misaligned-data alignment workflow.

Read `references/etna-data-validation.md` before using exogenous variables, panels, hierarchical data, intervals, misaligned data, transforms, or saved pipelines.

## Model Selection

Use model names exactly from `etna.models` or `etna.models.nn`. Main official forecaster families:

- Baselines and smoothing: `NaiveModel`, `MovingAverageModel`, `SeasonalMovingAverageModel`, `DeadlineMovingAverageModel`, `SimpleExpSmoothingModel`, `HoltModel`, `HoltWintersModel`.
- Classical/statistical: `AutoARIMAModel`, `SARIMAXModel`, `BATSModel`, `TBATSModel`, `ProphetModel`.
- StatsForecast wrappers: `StatsForecastARIMAModel`, `StatsForecastAutoARIMAModel`, `StatsForecastAutoCESModel`, `StatsForecastAutoETSModel`, `StatsForecastAutoThetaModel`.
- Regression wrappers: `CatBoostMultiSegmentModel`, `CatBoostPerSegmentModel`, `ElasticMultiSegmentModel`, `ElasticPerSegmentModel`, `LinearMultiSegmentModel`, `LinearPerSegmentModel`, `SklearnMultiSegmentModel`, `SklearnPerSegmentModel`.
- Neural and pretrained: `RNNModel`, `MLPModel`, `DeepStateModel`, `NBeatsGenericModel`, `NBeatsInterpretableModel`, `PatchTSTModel`, `DeepARModel`, `TFTModel`, `ChronosModel`, `ChronosBoltModel`, `TimesFMModel`.
- Ensembles: `DirectEnsemble`, `StackingEnsemble`, `VotingEnsemble`.

Read `references/etna-model-map.md` before claiming support for a model, extra dependency, ensemble, interval method, or hierarchical workflow.

## Forecasting Pattern

```python
from etna.datasets import TSDataset
from etna.metrics import MAE, SMAPE
from etna.models import CatBoostMultiSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import DateFlagsTransform, LagTransform, MeanTransform

horizon = 14
ts = TSDataset(df=prepared_long_df, freq="D", df_exog=future_known_df, known_future="all")
train_ts, test_ts = ts.train_test_split(test_size=horizon)

transforms = [
    LagTransform(in_column="target", lags=list(range(horizon, horizon + 28)), out_column="target_lag"),
    MeanTransform(in_column=f"target_lag_{horizon}", window=7),
    DateFlagsTransform(out_column="date_flag"),
]
pipeline = Pipeline(model=CatBoostMultiSegmentModel(), transforms=transforms, horizon=horizon)
pipeline.fit(train_ts)
forecast_ts = pipeline.forecast()

metrics = [MAE(), SMAPE()]
metric_values = {
    metric.__class__.__name__: metric(y_true=test_ts, y_pred=forecast_ts)
    for metric in metrics
}
```

For transforms that need near-target lags like `lag=1`, use `AutoRegressivePipeline(..., step=...)`; a regular `Pipeline` with direct regression features should use lag values available at prediction time, commonly `lags >= horizon`.

## Validation, Metrics, and Diagnostics

- Never random split. Use `TSDataset.train_test_split(test_size=horizon)` for holdout and `pipeline.backtest(ts, metrics=[...], n_folds=..., mode="expand"|"constant", stride=...)` for temporal validation.
- Recommended deterministic metrics: `MAE`, `RMSE`, `MSE`, `SMAPE`, `WAPE`, `MedAE`, and `MaxDeviation`. Avoid `MAPE` when actuals can be zero or near zero.
- For intervals, use `Coverage` and `Width` with temporal backtests.
- Plot with `ts.plot()` for raw data, `plot_forecast`, `plot_backtest`, `plot_metric_per_segment`, and `plot_residuals`.
- Diagnose residuals with `get_residuals`, `plot_residuals`, `acf_plot`, `qq_plot`, and segment-level metric distributions on held-out or backtest periods only.

## Prediction Intervals

Use documented interval wrappers from `etna.prediction_intervals`: `NaiveVariancePredictionIntervals`, `ConformalPredictionIntervals`, or `EmpiricalPredictionIntervals`. Wrap a fitted-compatible pipeline, then call:

```python
forecast_ts = interval_pipeline.forecast(
    prediction_interval=True,
    quantiles=(0.025, 0.975),
    n_folds=5,
)
```

Calibrate and evaluate intervals on temporal validation windows. Do not estimate interval residuals on the final test window after tuning.

## Anti-Leakage Rules

- Do not random split time rows, segments, or windows.
- Fit ETNA transforms only inside `Pipeline.fit` or each `backtest` fold; do not prefit imputers, scalers, trend removers, encoders, feature selectors, interval calibrators, or outlier transforms on full data.
- Build lag and rolling/statistical features only from past values. For regular `Pipeline`, ensure target lags are known for every forecast step; for recursive workflows, use `AutoRegressivePipeline`.
- Use future exogenous regressors only if they are known at the prediction timestamp and supplied across the full horizon.
- Keep `horizon`, `freq`, `stride`, `mode`, fold masks, and segment alignment consistent with the data-prep contract.
- For hierarchical or panel data, split by time cutoff across segments; do not let future values from any segment leak into shared transforms.

## Common Errors

- Passing long data without `timestamp`, `segment`, or `target`.
- Setting the wrong `freq`; ETNA may warn when discovered and supplied frequencies differ.
- Using target rolling statistics directly instead of applying statistics to a lagged target feature.
- Forgetting to install extras for Prophet, torch, statsforecast, Chronos, TimesFM, AutoML, or loggers.
- Treating `known_future="all"` as permission to use observed-only future covariates.
- Expecting a saved pipeline loaded with `dill` to be safe from untrusted files; official docs warn that loading can execute arbitrary code.
- Expecting model classes not listed in official ETNA 3.0 docs, such as LightGBM or XGBoost wrappers, to exist without verifying local source/docs.

## References

- Read `references/etna-model-map.md` for official models, pipelines, intervals, metrics, and capability caveats.
- Read `references/etna-data-validation.md` for data formats, exogenous variables, backtesting, plotting, diagnostics, and leakage controls.
- Read `references/official-sources.md` for official sources consulted.

## Ready Checklist

- `forecasting-data-prep` contract is complete and leakage risks are documented.
- `TSDataset` has valid long/wide pandas input, explicit `freq`, one `target` per segment, and aligned segment timestamps.
- Future-known exogenous columns are present for every horizon row if `known_future` is used.
- Model, optional extras, pipeline type, transforms, and horizon match the forecasting task.
- Validation uses temporal holdout or `backtest`, not random split.
- Metrics, plots, residuals, and interval coverage are computed on held-out/backtest periods only.
