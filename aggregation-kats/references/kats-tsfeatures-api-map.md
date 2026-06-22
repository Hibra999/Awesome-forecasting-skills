# Kats TSFeatures API Map

Use exact documented/source names. Kats public docs are old, and official pages/source disagree on fixed feature counts; pin the installed version for production use.

## Main API

| API | Purpose | Output |
| --- | --- | --- |
| `kats.tsfeatures.tsfeatures.TsFeatures` | Feature extractor class. | Configured extractor object. |
| `TsFeatures().transform(x)` | High-level transformation of `TimeSeriesData` into feature maps. | Dict for univariate; list of dicts for multivariate. |
| `kats.consts.TimeSeriesData` | Kats time-series container. | Stores time and value data. |

## Constructor Parameters

Documented parameters include:

- `window_size=20`: sliding window length for level shift, lumpiness, and stability.
- `spectral_freq=1`: frequency for periodogram/entropy.
- `stl_period=7`: period for STL/seasonality-related features.
- `nbins=10`: bins for flat spots and histogram mode.
- `lag_size=30`: maximum lag for Hurst exponent.
- `acfpacf_lag=6`: largest lag for ACF/PACF features.
- `decomp="additive"` and `iqr_mult=3.0`: outlier detector settings.
- `threshold=0.8`: trend detector threshold.
- `window=5`, `n_fast=12`, `n_slow=21`: nowcasting feature settings.
- `selected_features=None`: opt-in list of feature names or feature group names.
- `**kwargs`: feature or group switches such as `stl_features=False`, `cusum_detector=True`, or `time=True`.

## Default and Optional Groups

Current source defaults these groups on when `selected_features` is not supplied:

- `statistics`
- `stl_features`
- `level_shift_features`
- `acfpacf_features`
- `special_ac`
- `holt_params`
- `hw_params`

Current source defaults these groups off unless selected or enabled:

- `cusum_detector`
- `robust_stat_detector`
- `bocp_detector`
- `outlier_detector`
- `trend_detector`
- `nowcasting`
- `seasonalities`
- `time`

## Feature Group Mapping

Current source maps groups to these feature names:

- `statistics`: `length`, `mean`, `var`, `entropy`, `lumpiness`, `stability`, `flat_spots`, `hurst`, `std1st_der`, `crossing_points`, `binarize_mean`, `unitroot_kpss`, `heterogeneity`, `histogram_mode`, `linearity`.
- `stl_features`: `trend_strength`, `seasonality_strength`, `spikiness`, `peak`, `trough`.
- `level_shift_features`: `level_shift_idx`, `level_shift_size`.
- `acfpacf_features`: `y_acf1`, `y_acf5`, `diff1y_acf1`, `diff1y_acf5`, `diff2y_acf1`, `diff2y_acf5`, `y_pacf5`, `diff1y_pacf5`, `diff2y_pacf5`, `seas_acf1`, `seas_pacf1`.
- `special_ac`: `firstmin_ac`, `firstzero_ac`.
- `holt_params`: `holt_alpha`, `holt_beta`.
- `hw_params`: `hw_alpha`, `hw_beta`, `hw_gamma`.
- `cusum_detector`: `cusum_num`, `cusum_conf`, `cusum_cp_index`, `cusum_delta`, `cusum_llr`, `cusum_regression_detected`, `cusum_stable_changepoint`, `cusum_p_value`.
- `robust_stat_detector`: `robust_num`, `robust_metric_mean`.
- `bocp_detector`: `bocp_num`, `bocp_conf_max`, `bocp_conf_mean`.
- `outlier_detector`: `outlier_num`.
- `trend_detector`: `trend_num`, `trend_num_increasing`, `trend_avg_abs_tau`.
- `nowcasting`: `nowcast_roc`, `nowcast_ma`, `nowcast_mom`, `nowcast_lag`, `nowcast_macd`, `nowcast_macdsign`, `nowcast_macddiff`.
- `seasonalities`: `seasonal_period`, `trend_mag`, `seasonality_mag`, `residual_std`.
- `time`: `time_years`, `time_months`, `time_monthsofyear`, `time_weeks`, `time_weeksofyear`, `time_days`, `time_daysofyear`, `time_avg_timezone_offset`, `time_length_days`, `time_freq_Monday`, `time_freq_Tuesday`, `time_freq_Wednesday`, `time_freq_Thursday`, `time_freq_Friday`, `time_freq_Saturday`, `time_freq_Sunday`.

## Documented Gaps

- No supervised feature selection API is documented for `TsFeatures`.
- No sklearn transformer wrapper is documented for `TsFeatures`.
- No long/panel container for many independent series is documented; loop over ids.
- No dedicated plotting API is documented for `TsFeatures`; use raw series plots and downstream feature/model plots.
- Public docs are old; verify installed source for exact feature names and dependency behavior.
