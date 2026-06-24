# ruptures API Map

Use exact documented names.

## Estimator Interface

All main detection algorithms use an object-oriented API inspired by scikit-learn:

- `.fit(signal)`: fit the algorithm to the signal.
- `.predict(...)`: return a list of regime-end indexes; final element is `n_samples`.
- `.fit_predict(...)`: call fit then predict.

Constructor arguments documented for estimators include:

- `model`: cost model string, such as `"l1"`, `"l2"`, `"normal"`, `"rbf"`, `"linear"`.
- `custom_cost`: a `ruptures.base.BaseCost` instance.
- `jump`: restrict candidate breakpoint grid.
- `min_size`: minimum samples between breakpoints.

## Search Methods

| Class | Use When | Stopping |
| --- | --- | --- |
| `rpt.Dynp` | Exact dynamic programming; known number of changes. | `predict(n_bkps=K)` |
| `rpt.Pelt` | Penalized optimal segmentation; unknown number of changes. | `predict(pen=...)` |
| `rpt.KernelCPD` | Kernel change detection with linear/RBF/cosine kernels. | `n_bkps` or penalty per class docs/API |
| `rpt.Binseg` | Fast approximate binary segmentation. | `n_bkps`, `pen`, or `epsilon` |
| `rpt.BottomUp` | Fast approximate bottom-up segmentation. | `n_bkps`, `pen`, or `epsilon` |
| `rpt.Window` | Fast sliding-window segmentation. | `n_bkps`, `pen`, or `epsilon` |

Performance tradeoffs documented in the user guide:

- `Dynp`: exact but can be expensive, about `O(C K n^2)`.
- `Pelt`: pruning reduces cost; average complexity can be about `O(C K n)` under conditions.
- `Binseg`: low complexity about `O(C n log n)`.
- `BottomUp`: low complexity about `O(n log n)`.
- `Window`: about `O(n w)`, where `w` is window length.

## Cost Functions and Model Strings

| Cost | Model String | Change Type |
| --- | --- | --- |
| `rpt.costs.CostL1` | `"l1"` | Robust central-location/median shifts. |
| `rpt.costs.CostL2` | `"l2"` | Mean shifts. |
| `rpt.costs.CostNormal` | `"normal"` | Mean and covariance changes in Gaussian data. |
| `rpt.costs.CostRbf` | `"rbf"` | Kernelized/non-parametric distribution changes. |
| `rpt.costs.CostCosine` | `"cosine"` | Cosine-kernel embedded mean changes. |
| `rpt.costs.CostLinear` | `"linear"` | Piecewise linear regression changes. |
| `rpt.costs.CostCLinear` | `"clinear"` | Continuous piecewise linear changes. |
| `rpt.costs.CostRank` | `"rank"` | Rank-based multivariate distribution changes. |
| `rpt.costs.CostMl` | `"mahalanobis"` | Mean shifts under a Mahalanobis-type metric. |
| `rpt.costs.CostAR` | `"ar"` | Autoregressive changes; documented as limited to 1D. |

Cost objects support `.fit(signal)`, `.error(start, end)`, and `sum_of_costs(bkps)` in the docs.

## Custom Costs

Subclass `ruptures.base.BaseCost` and implement:

- `.fit(signal)`: set internal parameters and return `self`.
- `.error(start, end)`: return the segment cost for `signal[start:end]`.

Then pass the instance as `custom_cost=...` to a search method.

## Metrics

Official metrics:

- `ruptures.metrics.precision_recall(true_bkps, pred_bkps, margin=...)`
- `ruptures.metrics.hausdorff(true_bkps, pred_bkps)`
- `ruptures.metrics.randindex(true_bkps, pred_bkps)`

Always include the final `n_samples` endpoint in both true and predicted breakpoint lists.

## Plotting

Use:

```python
rpt.display(signal, bkps)
rpt.display(signal, true_bkps, predicted_bkps)
```

The display function alternates colors by regime and draws estimated breakpoints as dashed vertical lines when a second breakpoint list is passed.

## Documented Limitations

- Official positioning is off-line change point detection.
- Online minimal-delay detection is not documented as a core capability.
- `CostAR` is documented as limited to 1D signals.
- Approximate methods trade precision for speed; `jump` speeds detection at the expense of candidate resolution.
- `Window.width` should be smaller than the smallest regime.
