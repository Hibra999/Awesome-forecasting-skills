# Official Sources Consulted

Checked on 2026-06-21.

- sktime/PyTorch Forecasting GitHub README: https://github.com/sktime/pytorch-forecasting
- Stable documentation home: https://pytorch-forecasting.readthedocs.io/en/stable/
- Getting started: https://pytorch-forecasting.readthedocs.io/en/stable/getting-started.html
- Installation: https://pytorch-forecasting.readthedocs.io/en/stable/installation.html
- Data docs and `TimeSeriesDataSet`: https://pytorch-forecasting.readthedocs.io/en/stable/data.html
- `TimeSeriesDataSet` API: https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries._timeseries.TimeSeriesDataSet.html
- Models docs: https://pytorch-forecasting.readthedocs.io/en/stable/models.html
- Metrics docs: https://pytorch-forecasting.readthedocs.io/en/stable/metrics.html
- API index: https://pytorch-forecasting.readthedocs.io/en/stable/api.html
- Release notes: https://pytorch-forecasting.readthedocs.io/en/stable/CHANGELOG.html
- Tutorials index: https://pytorch-forecasting.readthedocs.io/en/stable/tutorials.html
- TFT tutorial: https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html
- DeepAR/DeepVAR tutorial: https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/deepar.html
- Official model source tree: https://github.com/sktime/pytorch-forecasting/tree/main/pytorch_forecasting/models
- Public model exports: https://raw.githubusercontent.com/sktime/pytorch-forecasting/main/pytorch_forecasting/models/__init__.py

Documentation notes:

- Latest GitHub release observed: v1.7.0 on 2026-04-05.
- The high-level model comparison and the API index are not identical. The skill separates main documented/top-level model workflows from API-indexed or v2/source classes.
- No official high-level universal rolling-origin backtesting helper or residual diagnostics API was identified in the consulted docs.
- The official docs emphasize pandas `DataFrame` plus `TimeSeriesDataSet` as the mainstream data path; v2 data-module APIs are indexed but not the default tutorial workflow.
