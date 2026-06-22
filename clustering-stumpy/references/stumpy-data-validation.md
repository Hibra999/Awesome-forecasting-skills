# STUMPY Data Validation Notes

## Shapes

- 1D functions such as `stump`, `mass`, `match`, `mpdist`, `fluss`, `stumpi`, and `snippets` expect 1D numeric arrays for each series.
- Multidimensional functions such as `mstump` and `mmotifs` expect `T.shape == (n_dimensions, n_timestamps)`.
- Query `Q` length defines the window length for `match`/`mass`.
- Matrix-profile window size `m` must be shorter than or equal to the available series length and should match the duration of the pattern of interest.

## Normalization And Distances

- Most core APIs default to z-normalized distances with `normalize=True`.
- If `normalize=False`, STUMPY reroutes to non-normalized equivalents and `p` controls the Minkowski norm.
- Do not compare distances across different `m`, sampling rates, normalization settings, or value units without recalibration.
- Constant subsequences and NaN/inf subsequences can affect behavior; use documented `*_subseq_isconstant` parameters when domain-specific constant handling matters.

## Missing Values

- Prefer fixing or segmenting missing data before STUMPY.
- If gaps mark true discontinuities, split the series and search each contiguous segment instead of interpolating across the gap.
- Document whether NaN/inf values were removed, interpolated, split, or intentionally left to STUMPY handling.

## Validation Workflow

1. Confirm the user wants query search, motif discovery, discord detection, clustering by distance, segmentation, or shapelet discovery.
2. Validate numeric time order and sample frequency.
3. Choose `m` from domain duration or query length.
4. Decide self-join versus AB-join.
5. Set normalization and threshold/exclusion-zone policy.
6. Evaluate top matches against held-out labels, event windows, or human review after parameters are fixed.

## Leakage Controls

- For supervised downstream use, compute motifs/shapelets/prototypes using train data only.
- For validation with known events, fix `m`, `max_distance`, and `max_matches` before measuring hit rate on held-out events.
- For streaming systems, never recompute a batch matrix profile using future points when reporting online performance.
- If clustering many full time series with `mpdist`, build the distance matrix using only data that would be available for the intended deployment time.

## Plotting

- Plot the raw target series with match windows highlighted by `[start_idx, start_idx + len(Q))`.
- Plot z-normalized query and matched subsequences overlaid when `normalize=True`.
- Plot the matrix profile below the raw series; low valleys are motifs and high peaks are discords.
- For segmentation, plot the corrected arc curve and regime-change indices from `fluss`.
