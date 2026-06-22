# Official Sources Consulted

Consulted for the `changepoint-ruptures` skill:

- ruptures GitHub repository: https://github.com/deepcharles/ruptures
- ruptures documentation home: https://centre-borelli.github.io/ruptures-docs/
- Installation docs: https://centre-borelli.github.io/ruptures-docs/install/
- Basic usage: https://centre-borelli.github.io/ruptures-docs/getting-started/basic-usage/
- Fitting and predicting: https://centre-borelli.github.io/ruptures-docs/fit-and-predict/
- Custom cost function: https://centre-borelli.github.io/ruptures-docs/custom-cost-function/
- Search methods: `Dynp`, `Pelt`, `KernelCPD`, `Binseg`, `BottomUp`, `Window` pages under https://centre-borelli.github.io/ruptures-docs/user-guide/detection/
- Cost functions: `CostL1`, `CostL2`, `CostNormal`, `CostRbf`, `CostCosine`, `CostLinear`, `CostCLinear`, `CostRank`, `CostMl`, `CostAR`, and custom cost pages under https://centre-borelli.github.io/ruptures-docs/user-guide/costs/
- Metrics and display: precision/recall, Hausdorff, Rand index, and display pages under https://centre-borelli.github.io/ruptures-docs/user-guide/
- PyPI package page: https://pypi.org/project/ruptures/

Notes captured from official sources:

- `ruptures` is documented as a Python library for off-line change point detection and signal segmentation.
- Installation is `python -m pip install ruptures` or `conda install ruptures`.
- Latest stable PyPI release is 1.1.10 from September 10, 2025; PyPI also lists a newer pre-release 1.1.11rc1.
- The estimator API uses `.fit()`, `.predict()`, and `.fit_predict()`.
- Predicted breakpoints are regime-end indexes and include the final sample index.
- Search methods documented are `Dynp`, `Pelt`, `KernelCPD`, `Binseg`, `BottomUp`, and `Window`.
- Cost functions documented include `CostL1`, `CostL2`, `CostNormal`, `CostRbf`, `CostCosine`, `CostLinear`, `CostCLinear`, `CostRank`, `CostMl`, `CostAR`, and custom `BaseCost` subclasses.
- Official metrics include precision/recall, Hausdorff, and Rand index.
- Official docs do not present `ruptures` as an online minimal-delay detector.
