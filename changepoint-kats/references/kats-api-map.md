# Kats Changepoint API Map

Use exact documented names. If a class is documented as anomaly or trend detection, do not rename it as a generic changepoint detector.

## Core Data Structures

- `kats.consts.TimeSeriesData`: core time-series container.
- `kats.consts.TimeSeriesChangePoint`: common return object with `start_time`, `end_time`, and `confidence`.

## Explicit Changepoint Detectors

| API | Use For | Key Parameters/Outputs |
| --- | --- | --- |
| `kats.detectors.cusum_detection.CUSUMDetector` | One increase/decrease level-shift changepoint in Gaussian mean-change data. | `.detector(threshold=0.01, max_iter=10, delta_std_ratio=1.0, min_abs_change=0, start_point=None, change_directions=None, interest_window=None, magnitude_quantile=None, magnitude_ratio=1.3, magnitude_comparable_day=0.5, return_all_changepoints=False)`. |
| `kats.detectors.cusum_detection.VectorizedCUSUMDetector` | Vectorized CUSUM over multiple series. | Same CUSUM-style detector parameters; returns list per series in official source. |
| `kats.detectors.bocpd.BOCPDetector` | Bayesian online changepoint detection. | `.detector(model=BOCPDModelType.NORMAL_KNOWN_MODEL, model_parameters=None, lag=10, choose_priors=True, changepoint_prior=0.01, threshold=0.5, debug=False, agg_cp=True)`. |
| `kats.detectors.robust_stat_detection.RobustStatDetector` | Robust univariate statistical changepoints. | `.detector(p_value_cutoff=1e-2, smoothing_window_size=5, comparison_window=-2)`. |
| `kats.detectors.cusum_model.CUSUMDetectorModel` | Multiple level shifts with rolling CUSUM. | Constructor includes `scan_window`, `historical_window`, `step_window`, `threshold`, `delta_std_ratio`, `magnitude_quantile`, `magnitude_ratio`, `change_directions`, `score_func`, `remove_seasonality`; use `.fit_predict(data, historical_data=None)`, then inspect `.cps`. |

## BOCPD Models

Use `BOCPDModelType`:

- `NORMAL_KNOWN_MODEL`: level shifts in normally distributed data.
- `TREND_CHANGE_MODEL`: slope/trend changes; uses `TrendChangeParameters`.
- `POISSON_PROCESS_MODEL`: count data with many values near zero.

Parameter classes:

- `NormalKnownParameters`
- `TrendChangeParameters`
- `PoissonModelParameters`

If `choose_priors=True`, Kats uses hyperparameter tuning dependencies; official source raises a runtime warning if the Ax platform is missing.

## Related Statistical/Trend Detectors

| API | Use For | Notes |
| --- | --- | --- |
| `kats.detectors.trend_mk.MKDetector` | Persistent monotonic trend detection using Mann-Kendall. | `.detector(window_size=20, training_days=None, direction="both", freq=None)`. Supports univariate/multivariate input through the `multivariate` flag. |
| `kats.detectors.stat_sig_detector.StatSigDetectorModel` | Rolling univariate test/control significant mean changes. | Requires enough `historical_data`; docs suggest `n_control >= 30`. |
| `kats.detectors.stat_sig_detector.MultiStatSigDetectorModel` | Rolling multivariate statistical changes with FDR adjustment. | Uses `n_control`, `n_test`, `time_units`, and `method`. |
| `kats.detectors.prophet_detector.ProphetDetectorModel` | Prophet-based anomaly scoring. | Official docs describe anomaly detection, not changepoint detection. |

## Evaluation APIs

- `kats.detectors.changepoint_evaluator.get_cp_index(changepoints, tsd)`: convert detector outputs to indexes.
- `kats.detectors.changepoint_evaluator.f_measure(annotations, predictions, margin=5, alpha=0.5)`: compute F-measure from human annotations and 0-based predicted locations.
- `kats.detectors.changepoint_evaluator.true_positives(T, X, margin=5)`: tolerance-based true positives.
- `TuringEvaluator`: benchmark evaluator for changepoint algorithms and compatible custom datasets.

## Plotting

- `CUSUMDetector.plot(change_points)` plots univariate CUSUM results.
- `BOCPDetector.plot(change_points, ts_names=None)` plots detected changepoints by series.
- `RobustStatDetector.plot(change_points)` plots red vertical changepoint markers.
- `MKDetector.plot(detected_time_points)` and `MKDetector.plot_heat_map()` are documented.
- CUSUM docs state multivariate CUSUM plotting is not yet supported.

## Documented Limitations

- CUSUM assumes at most one increase and/or one decrease changepoint per run and uses a Gaussian model for mean-change testing.
- `RobustStatDetector` official source supports only univariate series.
- `CUSUMDetectorModel.predict` is documented as not implemented.
- `BOCPDetector` has detection delay controlled by `lag`.
- `MKDetector` expects no seasonality for the MK test unless a documented frequency/seasonality handling path is used.
- PyPI marks Kats development status as alpha and latest release is 0.2.0 from 2022.
