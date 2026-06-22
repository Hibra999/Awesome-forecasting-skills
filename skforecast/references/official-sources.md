# Official Sources Consulted

Checked on 2026-06-22.

- skforecast GitHub README/repo: https://github.com/skforecast/skforecast
- Documentation home: https://skforecast.org/
- Quick start: https://skforecast.org/latest/quick-start/quick-start-skforecast
- Installation guide: https://skforecast.org/latest/quick-start/how-to-install
- Input data: https://skforecast.org/latest/user_guides/input-data
- Recursive multi-step forecasting: https://skforecast.org/latest/user_guides/autoregressive-forecaster
- Direct multi-step forecasting: https://skforecast.org/latest/user_guides/direct-multi-step-forecasting
- Independent multi-time series forecasting: https://skforecast.org/latest/user_guides/independent-multi-time-series-forecasting
- Dependent multivariate forecasting: https://skforecast.org/latest/user_guides/dependent-multi-series-multivariate-forecasting
- Deep learning RNN guide: https://skforecast.org/latest/user_guides/forecasting-with-deep-learning-rnn-lstm
- Statistical models / ForecasterStats: https://skforecast.org/latest/user_guides/forecasting-statistical-models
- ARIMA/SARIMAX/AutoARIMA: https://skforecast.org/latest/user_guides/forecasting-sarimax-arima
- ETS/AutoETS: https://skforecast.org/latest/user_guides/forecasting-ets
- Exogenous variables: https://skforecast.org/latest/user_guides/exogenous-variables
- Backtesting: https://skforecast.org/latest/user_guides/backtesting
- Probabilistic forecasting docs: bootstrapped residuals, conformal prediction, conformal calibration, quantile regression, probabilistic global models, CRPS.
- Plotting, explainability, save/load, deployment, examples/tutorials, API reference, and official AI context page: https://skforecast.org/latest/quick-start/ai-assisted-forecasting.html
- Official `llms-full.txt` context referenced by docs: https://skforecast.org/latest/llms-full.txt

Documentation notes:

- Stable docs checked: skforecast 0.22.0.
- README citation references v0.22.0 in April 2026.
- The official AI context warns against outdated names such as `ForecasterAutoreg`; this skill uses current class names.
- skforecast forecasters depend on the estimator's capabilities; not every estimator supports feature importance, intervals, quantile output, or probabilistic forecasts.
