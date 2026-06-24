---
name: changepoint-ruptures
description: Use ruptures for offline change point detection and signal segmentation after validating ordered time-series data, including univariate/multivariate numpy signals, exact and approximate search methods, cost models, known or unknown breakpoint counts, penalties, segmentation metrics, plotting, custom costs, and leakage-safe offline evaluation.
---

# ruptures Changepoints

Use this skill after time-series data preparation. `ruptures` is best for offline change point detection: segment a complete non-stationary signal into regimes and identify indexes where the process changes.

Do not use `ruptures` as an online detector unless you build a separate rolling/streaming wrapper. Official docs describe off-line change point detection and segmentation, not minimal-delay online detection.

## Minimum Install

```bash
python -m pip install ruptures
```

The docs also list `conda install ruptures`. PyPI identifies `ruptures 1.1.10` as the latest stable release, published September 10, 2025.

## Data Contract

- Require a prepared contract: entity id, sorted time/order column, numeric signal columns, sampling policy, train/validation/test periods, candidate detection objective, and leakage notes.
- Convert data to a numpy signal with shape `(n_samples,)` for univariate or `(n_samples, n_features)` for multivariate.
- Keep the sample order fixed. `ruptures` returns breakpoint indexes, not timestamps.
- Breakpoint lists include the final sample index `n_samples` by convention.
- Choose `min_size`, `jump`, and any window `width` from domain constraints before final evaluation.

Read `references/ruptures-data-workflow.md` before adapting panels, multiple entities, timestamps, or real-time-like workflows.

## Core Pattern

```python
import ruptures as rpt

signal = df[value_cols].to_numpy()
algo = rpt.Pelt(model="rbf", min_size=10, jump=1).fit(signal)
bkps = algo.predict(pen=10)
```

Use `rpt.display(signal, true_bkps, bkps)` for visual comparison when labels exist.

## Search Method Choice

- `Dynp`: exact dynamic programming when `n_bkps` is known; can be expensive.
- `Pelt`: penalized optimal segmentation when the number of changes is unknown.
- `KernelCPD`: kernel change detection with linear, RBF, or cosine kernels.
- `Binseg`: fast approximate binary segmentation.
- `BottomUp`: fast approximate bottom-up segmentation.
- `Window`: fast window-based segmentation; `width` should be smaller than the smallest regime.

Read `references/ruptures-api-map.md` for method/cost compatibility and exact API names.

## Cost Model Choice

- `model="l2"` / `CostL2`: mean shifts.
- `model="l1"` / `CostL1`: robust central-location shifts.
- `model="normal"` / `CostNormal`: mean and covariance changes in Gaussian data.
- `model="rbf"` / `CostRbf`: non-parametric distribution changes.
- `model="cosine"` / `CostCosine`: cosine-kernel changes, useful for directional/text/music-like representations.
- `model="linear"` / `CostLinear`: piecewise linear regression changes.
- `model="clinear"` / `CostCLinear`: continuous piecewise linear changes.
- `model="rank"` / `CostRank`: rank-based multivariate distribution changes.
- `model="mahalanobis"` / `CostMl`: mean changes under a Mahalanobis-type metric.
- `model="ar"` / `CostAR`: autoregressive changes; docs state this cost is limited to 1D signals.

For unsupported change types, subclass `ruptures.base.BaseCost` and implement `.fit(signal)` and `.error(start, end)`.

## Prediction and Tuning

- Use `.fit(signal)`, `.predict(...)`, or `.fit_predict(...)`.
- Use `predict(n_bkps=K)` when the number of changes is known or constrained.
- Use `predict(pen=...)` for penalized methods when the number is unknown.
- Use `predict(epsilon=...)` only for methods that document residual-budget stopping.
- Tune penalties, `n_bkps`, `min_size`, `jump`, `width`, and model choice on train/validation periods or labeled synthetic checks, not on final test.

## Metrics and Plotting

- With ground-truth breakpoints, use `ruptures.metrics.precision_recall`, `hausdorff`, and `randindex`.
- Set `margin` explicitly for precision/recall because tolerance is task-dependent.
- Without labels, inspect regime stability, residual/cost curves, domain plausibility, and sensitivity to `pen`, `min_size`, and `jump`.
- Plot with `rpt.display(signal, bkps)` or `rpt.display(signal, true_bkps, predicted_bkps)`.

## Anti-Leakage Rules

- Split time/entity groups before scaling, imputation, dimensionality reduction, cost selection, penalty tuning, or downstream modeling.
- Offline segmentation of the full series uses future observations. Do not use full-series breakpoints to make historical predictions unless the task is explicitly retrospective.
- For forecasting/anomaly workflows, detect breakpoints inside each train fold only, or simulate online use with expanding/rolling windows and measure detection delay separately.
- Fit scalers, PCA, embeddings, covariance metrics, and custom-cost parameters only on train data.
- Keep breakpoints, segment labels, and regime-derived features from validation/test out of train fitting.

## Common Errors

- Treating `ruptures` as online changepoint detection; official docs describe offline detection.
- Forgetting that the last breakpoint is `n_samples`.
- Passing unsorted rows or timestamp strings directly instead of ordered numeric signal values.
- Using `jump` too large and missing plausible breakpoint locations.
- Setting `min_size` or `width` larger than possible regimes.
- Comparing breakpoint indexes across resampled data without mapping back to timestamps.
- Tuning `pen` on the final test segment.

## References

- Read `references/ruptures-data-workflow.md` for signal shape, timestamps, panels, offline/online caveats, and leakage.
- Read `references/ruptures-api-map.md` for search methods, costs, metrics, plotting, and custom costs.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_ruptures_signal.py` to sanity-check CSV signals and breakpoint lists.

## Ready Checklist

- Data is ordered, numeric, finite, and shaped as univariate or multivariate signal.
- The task is offline segmentation or explicitly accepts retrospective use.
- Search method, cost model, `min_size`, `jump`, and stopping rule are documented.
- Evaluation uses validation periods, synthetic labels, or official metrics where true breakpoints exist.
- No preprocessing, penalty tuning, or segment-derived feature generation leaks validation/test future data.
