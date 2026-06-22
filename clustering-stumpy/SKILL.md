---
name: clustering-stumpy
description: Use STUMPY for time-series pattern search, motif discovery, matrix profiles, nearest-neighbor subsequence search, query matching with stumpy.match or stumpy.mass, motif discovery with stump/motifs, multidimensional motifs with mstump/mmotifs, MPdist similarity for clustering, snippets, semantic segmentation, shapelet discovery, streaming matrix profiles, Dask/Ray/GPU scaling, and leakage-aware time-series pattern validation.
---

# STUMPY Pattern Discovery

Use this skill when the task is searching for a specific sequence inside a longer time series, discovering repeated subsequences, computing subsequence similarity, clustering series by matrix-profile distance, or finding motifs/discords without labels.

Do not use STUMPY as a forecasting model. It is a matrix-profile/time-series data-mining library, not a classifier API.

## Minimum Install

```bash
python -m pip install stumpy
conda install -c conda-forge stumpy
```

STUMPY depends on NumPy, Numba, and SciPy. Use Dask/Ray variants for cluster execution and GPU variants only when compatible CUDA/Numba devices are available.

## Data Contract

- Require cleaned, time-ordered data with the original sampling order preserved.
- For query search, `Q` is the known pattern and `T` is the longer target series; both are 1D array-like numeric sequences.
- For matrix profiles, `T` is a 1D numeric sequence and `m` is the subsequence/window length.
- For multidimensional motifs, STUMPY uses shape `(n_dimensions, n_timestamps)`: each row is a dimension and each column is aligned time.
- Do not randomize timestamps. Preserve contiguous subsequences.
- Choose `m` from domain knowledge, sampling rate, known event duration, or a validation-only search; do not tune `m` on held-out outcomes.
- Read `references/stumpy-data-validation.md` before handling missing values, non-normalized distances, multidimensional data, or evaluation splits.

## Task Selection

- Known query pattern in one long series: use `stumpy.match(Q, T)` for unique matches with an exclusion zone, or `stumpy.mass(Q, T)` for the full distance profile.
- Unknown repeated patterns: compute `mp = stumpy.stump(T, m)` and inspect low matrix-profile values or use `stumpy.motifs(T, mp.P_, ...)`.
- Anomalies/discords: compute `stumpy.stump(T, m)` and inspect high matrix-profile values.
- Large/distributed data: use `stumpy.stumped(client, T, m)` with Dask/Ray, or `stumpy.gpu_stump(T, m, device_id=...)` on GPU.
- Multidimensional motif discovery: use `stumpy.mstump(T, m)` and `stumpy.mmotifs(T, P, I, ...)`.
- Distance between whole time series for clustering: use `stumpy.mpdist(T_A, T_B, m)`; build a pairwise distance matrix outside STUMPY.
- Streaming: use `stumpy.stumpi`.
- Segmentation: use `stumpy.fluss` for batch or `stumpy.floss` for streaming.
- Representative subsequences: use `stumpy.snippets`.
- Window-size exploration: use pan matrix profile with `stumpy.stimp`, `stumpy.stimped`, or `stumpy.gpu_stimp`.

Read `references/stumpy-task-map.md` before choosing APIs.

## Pattern Search Pattern

```python
import numpy as np
import stumpy

Q = np.asarray(query_values, dtype=float)
T = np.asarray(target_values, dtype=float)

matches = stumpy.match(
    Q,
    T,
    max_distance=lambda D: max(np.nanmean(D) - 4 * np.nanstd(D), np.nanmin(D)),
    max_matches=20,
)

for distance, start_idx in matches:
    subseq = T[start_idx : start_idx + len(Q)]
```

Use `stumpy.mass(Q, T)` when you need every distance:

```python
D = stumpy.mass(Q, T)
best_idx = int(np.nanargmin(D))
```

## Motif Discovery Pattern

```python
import numpy as np
import stumpy

T = np.asarray(values, dtype=float)
m = 50
mp = stumpy.stump(T, m)

motif_idxs = np.argsort(mp.P_)[:5]
motif_distances, motif_indices = stumpy.motifs(T, mp.P_, max_motifs=5)
discord_idx = int(np.nanargmax(mp.P_))
```

For self-joins, keep `ignore_trivial=True` so STUMPY applies the exclusion zone and avoids self-matches. For AB-joins between `T_A` and `T_B`, set `ignore_trivial=False`.

## Validation And Metrics

- There is no universal accuracy metric for unsupervised motif search. Validate with domain labels, known event windows, holdout periods, precision/recall over annotated events, or human review of top matches.
- For query search, report match start indices, distances, threshold, window length, normalization setting, and exclusion-zone policy.
- For clustering with `mpdist`, build a pairwise distance matrix and evaluate clusters with task-appropriate labels or stability checks.
- For shapelet discovery, evaluate downstream classifier performance on a train/validation split where shapelets are learned only on train.

## Anti-Leakage Rules

- Never choose query subsequences, thresholds, `m`, normalization mode, or cluster counts using held-out labels or future periods.
- If using `mpdist` for train/test classification-like workflows, compute distances from test samples only to train-derived exemplars/prototypes.
- Fit any scaling, imputation, dimensional selection, threshold calibration, or clustering model on train/validation only.
- For streaming/online use, use `stumpi`/`floss` or recompute only with data available at that timestamp.
- When benchmarking pattern retrieval, keep annotated event windows hidden until after search parameters are fixed.

## Common Errors

- Treating `stumpy.match` as exact string matching; it returns approximate z-normalized subsequence matches unless `normalize=False`.
- Passing multidimensional data as `(n_timestamps, n_dimensions)` instead of `(n_dimensions, n_timestamps)`.
- Forgetting `ignore_trivial=False` for AB-joins.
- Comparing raw distances across different `m`, normalization modes, or sampling rates.
- Letting overlapping matches dominate results; use `stumpy.match` or an exclusion-zone policy for unique occurrences.
- Assuming STUMPY handles forecasting, supervised classification, probability estimates, or labels directly.

## References

- Read `references/stumpy-task-map.md` for the official function map and task routing.
- Read `references/stumpy-data-validation.md` for shape, missing-value, normalization, leakage, and evaluation guidance.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_stumpy_input.py` before running `match`, `mass`, `stump`, or `mstump` on local CSV/TXT/NPY data.

## Ready Checklist

- Query/target or matrix-profile task is clearly stated.
- Window length `m` or query length is justified and not tuned on held-out answers.
- Data is numeric, time-ordered, and shaped correctly.
- Self-join vs AB-join behavior and exclusion-zone policy are explicit.
- Outputs include indices, distances, threshold, normalization mode, and validation method.
- Leakage controls are documented for any downstream classification, clustering, or evaluation.
