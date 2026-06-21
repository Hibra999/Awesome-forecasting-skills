# Prophet Data and Features

Load this reference when converting prepared data to Prophet or deciding which documented Prophet components to use.

## Data Formats

- Python API uses a pandas `DataFrame`; R uses a dataframe.
- Required columns are `ds` and `y`.
- `ds` should be a date or timestamp parseable by pandas/R. For Python, remove timezone information after converting to the canonical timezone.
- `y` must be numeric. Prophet tolerates missing `y` rows in history, but do not use missingness to hide unresolved data-quality issues.
- Prediction data needs `ds` and every required future column: regressors, condition columns, `cap`, and `floor` when used.

## Frequency and Horizon

- `make_future_dataframe(periods=..., freq=...)` creates future dates; Python `freq` follows pandas offset aliases.
- Sub-daily data is supported with timestamp `ds`; daily seasonality is fit automatically with sub-daily data.
- For regular gaps, only forecast valid timestamp windows observed in history. Example: if history contains 00:00-06:00 only, filter future to those hours.
- For monthly data, forecast monthly, for example with `freq="MS"`. Avoid daily forecasts from monthly data because seasonal components between monthly observations are unidentified.
- For weekly/monthly aggregated data, holidays only apply to the exact `ds`; move holiday dates to the aggregate timestamp if the effect should be modeled.

## Trends

- `growth="linear"`: default piecewise linear trend with automatic changepoints.
- `growth="logistic"`: saturating trend; requires `cap` for every history and future row. Optional `floor` gives a saturating minimum, but `cap` is still required.
- `growth="flat"`: no trend growth; useful when known regressors or seasonality should explain movement.
- `n_changepoints`, `changepoint_range`, `changepoint_prior_scale`, and explicit `changepoints` control trend flexibility. Tune by temporal validation, not in-sample fit.
- Custom trend functions beyond the three built-ins require source-code modification according to official docs.

## Seasonality

- Built-ins: yearly, weekly, daily. Values can be `auto`, boolean, or a Fourier order number.
- Custom seasonalities use `add_seasonality(name, period, fourier_order, prior_scale=None, mode=None, condition_name=None)`.
- Higher Fourier order allows faster-changing seasonal patterns and increases overfitting risk.
- Conditional seasonalities require boolean condition columns in both history and future.
- `seasonality_mode="multiplicative"` makes seasonalities scale with trend; individual seasonalities can override `mode`.

## Holidays and Events

- Custom holidays dataframe requires `holiday` and `ds`; optional `lower_window`, `upper_window`, and `prior_scale`.
- Include future occurrences for recurring holidays/events through the forecast horizon.
- One-off shocks can be included only in history; Prophet then models their historical impact without repeating them in the future.
- Built-in country holidays use `add_country_holidays(country_name=...)`.
- Python subdivision holidays can be created with `prophet.make_holidays.make_holidays_df`.

## Extra Regressors

- Add before fit with `m.add_regressor(name, prior_scale=None, standardize="auto", mode=None)`.
- The regressor column must exist in both fit and predict dataframes.
- Future regressor values must be known, forecast separately, or excluded. Forecasted regressors transfer their own error and can leak if built from future target information.
- Regressors can be additive or multiplicative. Binary regressors are not standardized under `standardize="auto"`; non-binary regressors are standardized by default.
- Coefficients can be inspected with utilities documented under coefficients of additional regressors when needed.

## Panel and Multivariate Use

- Prophet forecasts one `y` series per model.
- For panel data, loop over series IDs and fit one model per series, preserving per-series cutoffs and diagnostics.
- For multivariate data, choose one target as `y` and include other columns only as extra regressors if their future values are available.
- Prophet does not document native multi-output, global panel, or hierarchical reconciliation APIs.
