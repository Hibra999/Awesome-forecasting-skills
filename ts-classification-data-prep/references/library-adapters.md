# Library Adapters

Use the prepared contract before adapting to a classifier.

## Common Targets

- `classification-sktime`: accepts several panel formats. Choose the format after documenting equal or unequal length, channel order, and missing-value policy.
- `classification-tslearn`: usually expects 3D arrays shaped `(n_ts, sz, d)`. Use the contract to map `n_samples`, `n_timestamps`, and `n_channels`.
- `classification-pyts`: many estimators expect fixed-length 2D univariate arrays; multivariate workflows need documented channel handling.
- `classification-time-series-library`: preserve official UEA TRAIN/TEST files or write them from the prepared split contract.
- `classification-dl-4-tsc`: preserve archive-style train/test files and fixed tensor shapes.
- `classification-etna`: binary workflows need documented `0/1` labels and fold masks.

Do not change split policy during adapter conversion unless the change is documented and leakage-safe.
