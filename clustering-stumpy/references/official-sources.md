# Official Sources Consulted

- GitHub repository: https://github.com/stumpy-dev/stumpy
- README: https://github.com/stumpy-dev/stumpy/blob/main/README.rst
- Documentation home: https://stumpy.readthedocs.io/en/latest/
- Installation: https://stumpy.readthedocs.io/en/latest/install.html
- API reference: https://stumpy.readthedocs.io/en/latest/api.html
- Tutorials index: https://stumpy.readthedocs.io/en/latest/tutorials.html
- Fast Pattern Matching tutorial: https://stumpy.readthedocs.io/en/latest/Tutorial_Pattern_Matching.html
- STUMPY Basics tutorial: https://stumpy.readthedocs.io/en/latest/Tutorial_STUMPY_Basics.html
- Consensus Motif Search tutorial: https://stumpy.readthedocs.io/en/latest/Tutorial_Consensus_Motif.html
- Multidimensional Motif Discovery tutorial: https://stumpy.readthedocs.io/en/latest/Tutorial_Multidimensional_Motif_Discovery.html
- Shapelet Discovery tutorial: https://stumpy.readthedocs.io/en/latest/Tutorial_Shapelet_Discovery.html

## Documented Gaps

- STUMPY does not provide forecasting models or supervised classifier estimators.
- Clustering is not a single high-level STUMPY estimator; use `mpdist` to create pairwise distances, then cluster with an external clustering algorithm.
- Threshold choice for `match(max_distance=...)` is domain dependent and should be validated outside STUMPY.
