# Official Sources Consulted

Consulted on 2026-06-22.

- FEDOT GitHub README: https://github.com/aimclub/FEDOT
- FEDOT documentation 0.7.5 index: https://fedot.readthedocs.io/en/latest/
- Time Series Forecasting basics: https://fedot.readthedocs.io/en/latest/basics/ts_forecasting.html
- Time series forecasting example: https://fedot.readthedocs.io/en/latest/examples/ts_forecasting.html
- Time series forecasting pipelines: https://fedot.readthedocs.io/en/latest/examples/ts_pipelines.html
- Notebooks index: https://fedot.readthedocs.io/en/latest/examples/notebooks.html
- FEDOT API reference: https://fedot.readthedocs.io/en/latest/api/api.html
- Data API reference: https://fedot.readthedocs.io/en/latest/api/data.html
- Repository/API reference for tasks and metrics: https://fedot.readthedocs.io/en/latest/api/repository.html
- Official model operation repository: https://raw.githubusercontent.com/aimclub/FEDOT/master/fedot/core/repository/data/model_repository.json
- Official data operation repository: https://raw.githubusercontent.com/aimclub/FEDOT/master/fedot/core/repository/data/data_operation_repository.json
- Official metrics source: https://raw.githubusercontent.com/aimclub/FEDOT/master/fedot/core/repository/metrics_repository.py
- Official exogenous time-series example: https://github.com/aimclub/FEDOT/blob/master/examples/advanced/time_series_forecasting/exogenous.py

Notes:
- GitHub README lists latest release 0.7.5 on 2025-03-10; Read the Docs page title also identifies FEDOT 0.7.5.
- The official docs document point forecasting; prediction intervals or probabilistic forecasting for time-series were not found in the consulted official sources.
- `knnreg` appears in the official exogenous example's `available_operations`, but it was not returned by a task-filtered extraction of `model_repository.json` for `TaskTypesEnum.ts_forecasting`. Prefer the task-filtered operation list unless reproducing that exact example and verifying installed FEDOT behavior.
