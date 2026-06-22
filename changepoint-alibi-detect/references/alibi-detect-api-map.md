# Alibi Detect API Map for Changepoint-Like Workflows

Use exact documented names. Alibi Detect is a drift/outlier detection library; do not claim it provides native changepoint segmentation.

## Online Drift Detectors

| API | Use For | Key Arguments |
| --- | --- | --- |
| `alibi_detect.cd.MMDDriftOnline` | Multivariate kernel two-sample drift between fixed reference and sliding test window. | `x_ref`, `ert`, `window_size`, `backend`, `preprocess_fn`, `kernel`, `sigma`, `n_bootstraps`, `device`, `data_type`. |
| `alibi_detect.cd.LSDDDriftOnline` | Multivariate least-squares density-difference drift. | `x_ref`, `ert`, `window_size`, `backend`, `preprocess_fn`, `sigma`, `n_bootstraps`, `n_kernel_centers`, `lambda_rd_max`, `device`, `data_type`. |
| `alibi_detect.cd.CVMDriftOnline` | Continuous univariate-per-feature online drift; can correct for multivariate features. | `x_ref`, `ert`, `window_sizes`, `preprocess_fn`, `n_bootstraps`, `batch_size`, `n_features`, `data_type`. |
| `alibi_detect.cd.FETDriftOnline` | Binary/Bernoulli online drift, commonly error-rate monitoring. | `x_ref`, `ert`, `window_sizes`, `preprocess_fn`, `n_bootstraps`, `t_max`, `alternative`, `lam`, `n_features`, `data_type`. |

Common online behavior:

- `predict(x_t, return_test_stat=True)` processes a single instance.
- Returned dictionary has `meta` and `data`.
- `data` contains `is_drift`, `time`, `ert`, and optionally `test_stat`, `threshold`.
- `save_state`, `load_state`, and `reset_state` manage online state.
- `n_bootstraps` configures thresholds; docs recommend values at least an order of magnitude larger than `ert` for accurate targeting.

## Offline Drift Detectors for Rolling Windows

Use these when comparing a reference window to a candidate window retrospectively:

- `MMDDrift`
- `LSDDDrift`
- `KSDrift`
- `CVMDrift`
- `FETDrift`
- `ChiSquareDrift`
- `TabularDrift`
- `ClassifierDrift`
- `LearnedKernelDrift`
- `ContextMMDDrift`
- `SpotTheDiffDrift`
- `ClassifierUncertaintyDrift`
- `RegressorUncertaintyDrift`

Offline drift detectors generally use `predict(x, ...)` on batches and return `is_drift`, p-values/thresholds and/or distances depending on detector. They do not locate optimal changepoints; a rolling-window procedure must define the candidate timestamp convention.

## Related Time-Series Outlier Detectors

Official algorithm overview lists time-series outlier support for:

- `OutlierProphet`
- `SpectralResidual`
- `OutlierSeq2Seq`
- `LLR`

Use these only for anomalous point/interval detection and convert intervals to changepoint candidates outside Alibi Detect. They are not documented as changepoint detectors.

## Backends and Optional Dependencies

- Drift detectors support TensorFlow and PyTorch backends where applicable.
- `MMDDrift` also supports KeOps.
- Install extras as needed: `alibi-detect[tensorflow]`, `alibi-detect[torch]`, `alibi-detect[keops]`, `alibi-detect[prophet]`.
- PyPI release `0.13.0` requires Python `>=3.9`.

## Parameter Notes

- `ert`: expected runtime in the absence of drift. Larger means fewer false alarms but often longer delay.
- `window_size` / `window_sizes`: smaller windows react faster to severe drift; larger windows have more power for slight drift.
- `p_val`: significance level for offline drift tests.
- `update_x_ref`: offline detectors may update reference data by `{"last": n}` or `{"reservoir_sampling": n}`. Treat this as a production policy because it changes the reference distribution.
- `x_ref_preprocessed`: set only when `x_ref` has already been preprocessed.
- `data_type="time-series"` only sets metadata; it does not enforce temporal validation.

## Limitations

- No official native changepoint segmentation API is documented.
- Online drift alarms occur after enough evidence accumulates; the alarm time may lag the true change.
- Timestamps and plotting are user-managed.
- `CVMDriftOnline` docs warn about Numba/OpenMP issues on macOS and memory scaling during threshold configuration.
- `CVMDriftOnline` and `FETDriftOnline` multivariate ERT behavior assumes feature independence for accurate targeting.
- `FETDriftOnline` rejects non-binary data.
