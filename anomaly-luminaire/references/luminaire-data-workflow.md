# Luminaire Data Workflow

Use this reference before converting raw data into Luminaire batch or streaming anomaly workflows.

## Input preparation

Expected practical input:

- one timestamp column convertible to pandas datetime;
- one numeric signal column renamed to `raw`;
- no feature matrix unless handled outside Luminaire;
- optional labels for evaluation only, never as model input;
- optional entity id only for external looping, not for one Luminaire model.

Recommended conversion:

```python
data = (
    pd.read_csv(path, parse_dates=["time"])
    .sort_values("time")
    .set_index("time")
    .rename(columns={"value": "raw"})[["raw"]]
)
```

For panels, group by entity, split each entity chronologically, and run the Luminaire workflow separately per entity. If you share optimized configs across entities, justify that this matches production.

## Batch workflow

1. Chronologically split raw history into train/validation/test.
2. Run `HyperparameterOptimization(freq=...).run(data=train)` only on train or train+validation inside tuning.
3. Run `DataExploration(freq=..., **config).profile(train)` only on the same training window.
4. Instantiate `LADStructuralModel` or `LADFilteringModel` from the optimized or manual config.
5. Train with `model_object.train(data=training_data, **pre_prc)`.
6. Score future timestamps in validation/test with `trained_model.score(value, date)`.
7. Select operational thresholds from validation only.

`LADFilteringModel` scoring is sequential: use the returned model update to score the next timestamp.

## Streaming/window workflow

1. Choose window anomaly detection when sustained fluctuations matter more than point alerts.
2. Start with `WindowDensityHyperParams(...).params` or defaults.
3. Run `DataExploration(**config).stream_profile(df=train)` on training data only.
4. Update config with profiler output, then train `WindowDensityModel`.
5. For each scoring window, run imputation-only stream profiling with the trained model frequency.
6. Score equal-sized or explicitly documented rolling windows with `model.score(processed_window)`.

Keep window length, step/stride, start/end timestamps, and overlap policy explicit before evaluation.

## Validation and metrics

Use chronological validation. For rare anomaly labels, prefer:

- anomaly-class precision, recall, and F1;
- PR-AUC or average precision from anomaly probabilities;
- ROC-AUC only when both classes are present;
- precision@K/top-N alerts;
- false positives per day/week;
- alert volume versus operator budget;
- detection delay and overlap for window anomalies.

Do not rely on accuracy for imbalanced monitoring data.

## Leakage controls

Avoid:

- running `DataExploration.profile` or `stream_profile` on full history before splitting;
- letting imputation, log-transform choice, change-point truncation, or hyperparameter optimization inspect validation/test;
- choosing anomaly thresholds after seeing final test labels;
- using centered rolling windows for online streaming detection;
- mixing entity histories before chronological splitting;
- using labels as columns in the `raw` input frame.

## When to avoid Luminaire

Use another library when:

- you need a multivariate feature-based outlier detector;
- you need generic sklearn-compatible estimators;
- you need native panel modeling across many series;
- you need maintained support for Python versions beyond those documented by PyPI;
- the task is offline changepoint segmentation rather than future monitoring, even though Luminaire profiling detects change points internally.
