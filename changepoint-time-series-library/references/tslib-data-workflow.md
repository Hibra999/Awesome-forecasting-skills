# TSLib Anomaly/Event Data Workflow

TSLib does not document a native changepoint task. Use anomaly detection outputs as event intervals and map interval starts/ends to changepoint candidates outside TSLib.

## Official Anomaly Datasets

`data_provider.data_factory` maps anomaly detection to these loaders:

- `PSM`: `PSMSegLoader`
- `MSL`: `MSLSegLoader`
- `SMAP`: `SMAPSegLoader`
- `SMD`: `SMDSegLoader`
- `SWAT`: `SWATSegLoader`

The anomaly branch passes `root_path`, `win_size=args.seq_len`, and `flag`.

## Expected Local Files

| Dataset | Files |
| --- | --- |
| `PSM` | `train.csv`, `test.csv`, `test_label.csv`; loaders use all columns after the first column as values/labels. |
| `MSL` | `MSL_train.npy`, `MSL_test.npy`, `MSL_test_label.npy`. |
| `SMAP` | `SMAP_train.npy`, `SMAP_test.npy`, `SMAP_test_label.npy`. |
| `SMD` | `SMD_train.npy`, `SMD_test.npy`, `SMD_test_label.npy`. |
| `SWAT` | `swat_train2.csv`, `swat2.csv`; train/test values use all columns except the last, and the last test column is the label. |

If local files are absent, current loaders attempt to fetch data from the Hugging Face repo `thuml/Time-Series-Library`.

## Windowing and Labels

- `seq_len` becomes `win_size`.
- Most anomaly loaders use `step=1`; `SMDSegLoader` defaults to `step=100`.
- `__getitem__` returns `(window_values, window_labels)`.
- Train/validation windows return non-operational labels from the start of test labels, so do not interpret train labels.
- Test labels are binary point labels. Convert contiguous runs of `1` into event intervals before deriving changepoints.

## Scaling

Official anomaly loaders fit `StandardScaler` on train data and transform train/test. Preserve this train-only scaling pattern in custom loaders.

## Custom Data

`--data custom` is not wired to `Exp_Anomaly_Detection`. To use a custom dataset, implement or adapt a loader and register it in `data_provider.data_factory.data_dict`; do not assume the forecasting `Dataset_Custom` works for anomaly detection.

## Converting Events to Changepoints

After prediction:

1. Build contiguous predicted anomaly intervals.
2. Choose start, end, or both as changepoint candidates based on the process definition.
3. If timestamps exist, map point indexes back to timestamps.
4. Evaluate with a tolerance window and detection delay if ground-truth changepoints exist.

Do not use ground-truth intervals to expand predictions in production; the official `adjustment` function is a benchmark metric convention.
