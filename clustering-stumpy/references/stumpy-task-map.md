# STUMPY Task Map

STUMPY computes matrix-profile-based time-series data-mining primitives. Use the task, not model-family language, when choosing APIs.

| Task | Official API | Notes |
| --- | --- | --- |
| Exact 1D matrix profile | `stumpy.stump(T_A, m, T_B=None, ignore_trivial=True, normalize=True, p=2.0, k=1)` | Returns distances `.P_`, nearest-neighbor indices `.I_`, left/right indices. Use self-join for one series and AB-join for two series. |
| Distributed 1D matrix profile | `stumpy.stumped(client, T_A, m, ...)` | Requires Dask or Ray client. Cluster setup is outside STUMPY. |
| GPU 1D matrix profile | `stumpy.gpu_stump(T_A, m, ..., device_id=...)` | Uses one or more CUDA devices through Numba. |
| Distance profile for query search | `stumpy.mass(Q, T)` | Computes distances from one query subsequence to all equal-length subsequences in `T`. |
| Approximate matrix profile | `stumpy.scrump(...)` | Use when exact matrix profile is too expensive and approximation is acceptable. |
| Streaming matrix profile | `stumpy.stumpi(...)` | Incremental matrix profile for online/streaming data. |
| Multidimensional matrix profile | `stumpy.mstump(T, m)` | `T` shape is `(n_dimensions, n_timestamps)`. Returns multidimensional profiles and indices. |
| Distributed multidimensional profile | `stumpy.mstumped(client, T, m)` | Distributed variant for multidimensional data. |
| Multidimensional subspace | `stumpy.subspace(...)` | Finds dimensions relevant to a subsequence/nearest-neighbor pair. |
| MDL dimension selection | `stumpy.mdl(...)` | Computes minimum description length for multidimensional motif search. |
| Time series chains | `stumpy.atsc(...)`, `stumpy.allc(...)` | Uses left/right matrix-profile indices. |
| Semantic segmentation | `stumpy.fluss(...)`, `stumpy.floss(...)` | FLUSS is batch; FLOSS is streaming. |
| Consensus motif across series | `stumpy.ostinato(...)`, `stumpy.ostinatoed(...)`, `stumpy.gpu_ostinato(...)` | Finds a z-normalized consensus motif across multiple time series. |
| Whole-series similarity | `stumpy.mpdist(...)`, `stumpy.mpdisted(...)`, `stumpy.gpu_mpdist(...)` | Matrix-profile distance for comparing two time series; useful for clustering after building pairwise distances. |
| Motif discovery | `stumpy.motifs(T, P, ...)` | Discovers top motifs from a 1D matrix profile. |
| Query matches | `stumpy.match(Q, T, max_distance=None, max_matches=None, ...)` | Finds all matches of query `Q` in `T`; returns `[distance, index]` sorted by distance. |
| Multidimensional motifs | `stumpy.mmotifs(T, P, I, ...)` | Motif discovery for multidimensional time series. |
| Snippets | `stumpy.snippets(T, m, k, ...)` | Finds representative subsequences for summarizing long time series. |
| Pan matrix profile | `stumpy.stimp(...)`, `stumpy.stimped(...)`, `stumpy.gpu_stimp(...)` | Explore subsequence window sizes. |

## Choosing Between `match`, `mass`, `stump`, And `motifs`

- Use `match` when the user asks, "find this pattern in that longer series" and wants unique occurrences.
- Use `mass` when the user needs the full distance profile or custom ranking.
- Use `stump` when the user does not know the pattern and wants motifs/discords.
- Use `motifs` after `stump` when the user wants grouped motif occurrences instead of inspecting the lowest matrix-profile entries manually.
- Use `mpdist` when the user wants a distance between whole time series for clustering, retrieval, or nearest-series search.

## Key Output Conventions

- `stump(..., k=1)` returns four columns: matrix profile distance, matrix profile index, left index, right index.
- `stump(..., k>1)` returns `2*k + 2` columns: top-k distances, top-k indices, left index, right index.
- `.P_`, `.I_`, `.left_I_`, and `.right_I_` are named array attributes for common access.
- `match` returns an array where column 0 is distance and column 1 is the start index in `T`.
