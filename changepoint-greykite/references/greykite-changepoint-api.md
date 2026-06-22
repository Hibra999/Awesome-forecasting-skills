# Greykite Changepoint API Map

Use exact documented names. Greykite changepoints are mainly forecasting/model components and exploratory detectors, not a generic segmentation API.

## Standalone Detector

`greykite.algo.changepoint.adalasso.changepoint_detector.ChangepointDetector`

- `find_trend_changepoints(df, time_col, value_col, shift_detector=None, yearly_seasonality_order=8, yearly_seasonality_change_freq=None, resample_freq="D", trend_estimator="ridge", adaptive_lasso_initial_estimator="ridge", regularization_strength=None, actual_changepoint_min_distance="30D", potential_changepoint_distance=None, potential_changepoint_n=100, potential_changepoint_n_max=None, no_changepoint_distance_from_begin=None, no_changepoint_proportion_from_begin=0.0, no_changepoint_distance_from_end=None, no_changepoint_proportion_from_end=0.0, fast_trend_estimation=True)`
- Returns a `dict` with `trend_feature_df`, `trend_changepoints`, `changepoints_dict`, and `trend_estimation`.
- Raises when fewer than 5 non-null observations remain.
- `regularization_strength` is in `[0, 1]`; larger values imply fewer changepoints. `None` runs cross-validation to select the tuning parameter.
- `potential_changepoint_distance` overrides `potential_changepoint_n`.
- `no_changepoint_distance_from_begin/end` override the corresponding proportions.

## Seasonality Changepoints

`ChangepointDetector.find_seasonality_changepoints(df, time_col, value_col, seasonality_components_df=..., resample_freq="H", regularization_strength=0.6, actual_changepoint_min_distance="30D", potential_changepoint_distance=None, potential_changepoint_n=50, no_changepoint_distance_from_end=None, no_changepoint_proportion_from_end=0.0, trend_changepoints=None)`

- Detects points where seasonality magnitude/shape changes.
- Removes estimated trend first. If trend detection was already run on the same object/data, it reuses that information; otherwise it runs trend detection automatically with the provided/default parameters.
- Returns `seasonality_feature_df`, `seasonality_changepoints`, `seasonality_estimation`, and `seasonality_components_df`.
- `seasonality_components_df` has columns `name`, `period`, `order`, and `seas_names`. Examples include `tod`, `tow`, and `conti_year`.

## Plotting

`ChangepointDetector.plot(...)` can include observations, trend estimate, trend changepoints, yearly seasonality estimate, adaptive-lasso trend estimate, seasonality changepoints, and seasonality estimate. With `plot=False`, it returns a Plotly figure object.

`greykite.algo.changepoint.adalasso.changepoints_utils.plot_change(...)` is the lower-level plotting utility.

## Silverkite Configuration

Use `ModelComponentsParam(changepoints=...)` or `ForecastConfig.from_dict(...)`.

Trend changepoints live under:

```python
changepoints={
    "changepoints_dict": {...}
}
```

Documented `changepoints_dict["method"]` values:

- `"uniform"` requires `n_changepoints`.
- `"custom"` requires `dates`, parsable by `pandas.to_datetime`.
- `"auto"` accepts parameters matching `find_trend_changepoints` except `df`, `time_col`, and `value_col`.

For automatic trend changepoints, optional custom dates can be added with:

- `dates`
- `combine_changepoint_min_distance`
- `keep_detected`

Other trend keys include:

- `continuous_time_col`
- `growth_func`
- `auto_growth=True` to let Silverkite auto-configure growth and trend changepoint detection.

Seasonality changepoints live under:

```python
changepoints={
    "seasonality_changepoints_dict": {...}
}
```

An empty `seasonality_changepoints_dict` triggers seasonality changepoint detection with defaults.

## Level Shift Detection

`greykite.algo.changepoint.shift_detection.shift_detector.ShiftDetection`

- Main method: `detect(original_df, time_col, value_col, forecast_horizon=0, freq="D", z_score_cutoff=3)`.
- Finds level shifts by first differencing, z-scoring absolute differences, and thresholding.
- Returns `regressor_col` and a `final_df` containing `ctp_*` level-shift regressors.
- `freq` must be one of `T`, `H`, `D`, `W`, `M`, or `Y`.
- `plot_level_shift()` plots observations with level-shift regions after `detect()` or `find_shifts()`.

## Forecast Validation

Use `EvaluationPeriodParam` for temporal validation:

- `test_horizon`
- `periods_between_train_test`
- `cv_horizon`
- `cv_min_train_periods`
- `cv_periods_between_splits`
- `cv_periods_between_train_test`
- `cv_max_splits`
- `cv_expanding_window`

Use `result.backtest`, `result.forecast`, `result.grid_search`, and component plots to judge whether changepoint settings improve held-out forecast behavior.

## Limitations

- No documented online minimal-delay changepoint detector.
- No documented multivariate/panel changepoint detector; loop per target/entity.
- No built-in changepoint precision/recall metric was found in official docs; score labeled changepoints externally.
- Distance parameters validated by Greykite allow units no larger than day-level strings such as `10D`, `5H`, `100T`, or `200S`; `W`, `M`, and `Y` are rejected for those parameters.
- Docs warn that changepoints too close to the end do not leave enough data to learn the new trend.
- Docs warn that too many Silverkite changepoints can overfit and conflate trend with yearly seasonality.
