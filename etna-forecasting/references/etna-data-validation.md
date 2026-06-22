# ETNA Data, Validation, And Leakage Notes

Use this file when converting prepared data to ETNA, adding regressors, validating, plotting, diagnosing residuals, or using saved inference.

## Input Formats

`TSDataset` accepts pandas DataFrames in long or wide ETNA format.

Long format:
- Has a `timestamp` column.
- Has a `segment` column.
- Has at least one feature column; forecasting workflows should include `target`.
- `timestamp` is expected to be integer or `pd.Timestamp`; ETNA may convert.
- `segment` is cast to string by `TSDataset`.

Wide format:
- Index stores timestamps.
- Columns are a two-level MultiIndex named `segment` and `feature`.
- Each column stores one feature for one segment.
- Use `TSDataset.to_dataset(long_df)` to convert long to wide and `TSDataset.to_pandas(flatten=True)` or `TSDataset.to_flatten` to return to long form.

Frequency:
- Pass `freq` as a pandas `DateOffset`, a pandas offset alias string, or `None` for integer timestamps.
- Verify supplied frequency against the prepared data. ETNA can warn if the supplied frequency differs from discovered frequency.

## Univariate, Panel, And Multivariate Use

- Univariate: use one `segment` with one `target`.
- Multiple series/panel: use multiple segment IDs; ETNA reports per-segment metrics and plots.
- Multivariate features: add exogenous features and generated transforms to each segment.
- Multi-target endogenous forecasting is not documented as a separate first-class workflow in the browsed ETNA 3.0 docs. If the user has several targets, represent them as segments only when independent/parallel treatment is valid, or build separate ETNA workflows.

## Exogenous Regressors

Use `TSDataset(df=target_df, freq=..., df_exog=regressor_df, known_future=...)`.

Rules:
- Put target series in `df` and exogenous data in `df_exog`; the official tutorial says this separation helps avoid mixing future target information with regressors.
- Set `known_future="all"` only if every exogenous column is known for all prediction timestamps.
- Otherwise pass an explicit sequence of future-known regressor column names.
- Provide future exogenous rows for every segment and every timestamp in the forecast horizon.
- Do not include observed-only future variables, target-derived future summaries, or post-event features as regressors.

## Transforms And Feature Safety

Common documented transform groups include lags, rolling/statistical features, calendar/date flags, Fourier terms, trend/decomposition transforms, imputers, scalers, encoders, outlier transforms, feature selection, and embeddings.

Leakage controls:
- Apply transforms inside `Pipeline` or backtest folds so fit state is learned from train only.
- For regular `Pipeline`, direct target lags must be known at forecast time; the get-started tutorial uses `lags=list(range(HORIZON, ...))`.
- Statistical transforms such as `MeanTransform` should be applied to lagged target columns, not directly to the future target, because documented rolling windows include the current timestamp.
- Calendar features are safe only when derived from timestamp and known before prediction.
- Imputation, scaling, trend removal, outlier detection, feature selection, and interval residual estimation must be fit per train window.

## Forecast Horizon And Prediction

- Set `horizon` on `Pipeline` or `AutoRegressivePipeline`.
- `pipeline.forecast()` without a `ts` argument forecasts the next `horizon` timestamps after the fitted training dataset.
- `pipeline.forecast(ts=...)` can forecast from a supplied dataset when the workflow has been designed for that.
- `pipeline.predict(...)` is in-sample prediction over a range; do not use it as a substitute for future forecasting.
- Use `TSDataset.make_future(...)` only with future timestamps/features that are valid for the horizon.

## Temporal Validation

Holdout:

```python
train_ts, test_ts = ts.train_test_split(test_size=horizon)
```

Backtest:

```python
from etna.metrics import MAE, SMAPE

backtest_result = pipeline.backtest(
    ts=ts,
    metrics=[MAE(), SMAPE()],
    n_folds=5,
    mode="expand",
    stride=horizon,
    refit=True,
)
metrics_df = backtest_result["metrics"]
forecast_ts_list = backtest_result["forecasts"]
fold_info_df = backtest_result["fold_info"]
```

Validation rules:
- Do not random split rows.
- Use `mode="expand"` for expanding train windows or `mode="constant"` for rolling fixed train length when documented and appropriate.
- Use `stride` to control distance between fold starts; default is the pipeline horizon when not set.
- Prefer `refit=True` when transforms/models should be refit at every fold.
- Use `FoldMask` only for explicit, documented temporal folds.
- Tune models/transforms on validation/backtest windows, then keep a final untouched test cutoff.

## Metrics And Reporting

Recommended ETNA metrics:
- Point forecasts: `MAE`, `RMSE`, `MSE`, `SMAPE`, `WAPE`, `MedAE`, `MaxDeviation`.
- Probabilistic/interval forecasts: `Coverage`, `Width`.
- Use `compute_metrics` when combining multiple metrics in ETNA-native format.

Report:
- Per-segment metrics and aggregate summaries.
- Fold-level stability and segment-level failures, not only an overall mean.
- Business-unit-scaled metrics from `forecasting-data-prep` when available.

## Plotting And Residual Diagnostics

Use:
- `ts.plot()` for input series.
- `plot_forecast(forecast_ts=..., test_ts=..., train_ts=..., n_train_samples=...)`.
- `plot_backtest` or `plot_backtest_interactive` for backtest forecasts.
- `plot_metric_per_segment` and `metric_per_segment_distribution_plot` for segment diagnostics.
- `get_residuals`, `plot_residuals`, `acf_plot`, `qq_plot`, and `prediction_actual_scatter_plot` for residual checks.

Residuals should come from validation/backtest windows. Avoid tuning decisions based on final test residuals.

## Prediction Intervals

Available wrappers:
- `NaiveVariancePredictionIntervals`
- `ConformalPredictionIntervals`
- `EmpiricalPredictionIntervals`

Pattern:

```python
from etna.prediction_intervals import EmpiricalPredictionIntervals

interval_pipeline = EmpiricalPredictionIntervals(pipeline=pipeline, coverage=0.95)
interval_pipeline.fit(train_ts)
forecast_ts = interval_pipeline.forecast(prediction_interval=True, quantiles=(0.025, 0.975), n_folds=5)
```

Evaluate:
- `Coverage` should be close to the requested interval level on validation/backtest windows.
- `Width` should be tracked with coverage; wider intervals are not automatically better.
- Calibrate intervals with historical residuals from train/backtest windows only.

## Hierarchical And Misaligned Data

- ETNA documents `TSDataset.to_hierarchical_dataset`, `HierarchicalStructure`, `HierarchicalPipeline`, `BottomUpReconciliator`, and `TopDownReconciliator` for hierarchy workflows.
- Check that every hierarchy level and aggregation relationship is defined before modeling.
- ETNA documents `infer_alignment`, `apply_alignment`, `make_timestamp_df_from_alignment`, and `TSDataset.create_from_misaligned` for misaligned data.
- Do not silently align arbitrary panels; record the alignment rule in the data-prep contract and validate that it is causal.

## Saved Inference

- `Pipeline.save(path)` and `Pipeline.load(path, ts=...)` are documented.
- Official API warns `load` uses `dill` and can execute arbitrary code. Never load untrusted or tampered artifacts.
- For new data inference, recreate the same `TSDataset` schema, frequency, segment names, exogenous columns, and transform expectations used during training.

## Anti-Leakage Checklist

- Split by timestamp before any target-aware fitting.
- Fit all ETNA transforms inside pipeline/backtest folds.
- Use `lags >= horizon` in direct regular pipelines unless a documented recursive strategy is used.
- Apply rolling/statistical transforms to lagged features, not raw future target values.
- Use only known-future exogenous regressors.
- Recompute validation metrics per temporal fold.
- Keep final test data untouched until model and interval choices are frozen.
