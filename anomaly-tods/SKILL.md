---
name: anomaly-tods
description: Use TODS for automated time-series outlier detection after validating multivariate time-series data, including D3M pipeline primitives, default pipeline evaluation, AutoML pipeline search, point-wise, pattern-wise, and system-wise detection, PyOD wrappers, DeepLog, Telemanom, MatrixProfile, feature extraction, metrics, and anti-leakage safeguards.
---

# TODS Anomaly Detection

Use this skill when the task needs a full-stack time-series outlier detection pipeline rather than a single detector: preprocessing, time-series transforms, feature analysis, detector primitives, optional human-rule filtering, default pipeline evaluation, or AutoML search.

Important scope: TODS is a D3M-style pipeline system for multivariate time-series outlier detection. Its public docs are old (`0.0.1` docs, PyPI `0.0.2` from 2020) and some APIs differ between docs and README; verify installed imports before production use.

## Minimum Install

README install path:

```bash
sudo apt-get install libssl-dev libcurl4-openssl-dev libyaml-dev build-essential libopenblas-dev libcap-dev ffmpeg
git clone https://github.com/datamllab/tods.git
cd tods
pip install -e .
```

PyPI also lists `pip install tods` for `0.0.2`, released September 21, 2020. README says Python 3.7+ and pip 19+; the hosted docs mention Python 3.6. Treat modern Python compatibility as unverified.

## Data Contract

- Start from an ordered multivariate time-series table, usually a pandas `DataFrame` converted with `generate_dataset(df, target_index=...)`.
- `target_index` points to the label/target column in the official examples.
- TODS primitives operate on D3M container datasets/DataFrames and pipeline semantic types such as `Attribute` and `TrueTarget`.
- Detection outputs are binary labels where `1` marks outliers and `0` marks normal; many primitives also expose scores through `produce_score`.
- TODS targets three scenarios documented by README/docs: point-wise time-point outliers, pattern-wise subsequence outliers, and system-wise sets of time series as outliers.

Read `references/tods-data-workflow.md` before adapting CSVs, labels, entity panels, subsequence windows, or temporal validation.

## Core Patterns

Default pipeline evaluation:

```python
import pandas as pd
from tods import evaluate_pipeline, generate_dataset
from tods import schemas as schemas_utils

df = pd.read_csv("datasets/anomaly/raw_data/yahoo_sub_5.csv")
dataset = generate_dataset(df, target_index=6)
pipeline = schemas_utils.load_default_pipeline()
result = evaluate_pipeline(dataset, pipeline, "F1_MACRO")
print(result)
```

AutoML pipeline search:

```python
import pandas as pd
from axolotl.backend.simple import SimpleRunner
from tods import generate_dataset, generate_problem
from tods.searcher import BruteForceSearch

df = pd.read_csv("datasets/yahoo_sub_5.csv")
dataset = generate_dataset(df, target_index=6)
problem = generate_problem(dataset, "F1_MACRO")
search = BruteForceSearch(problem_description=problem, backend=SimpleRunner(random_seed=0))
best_runtime, best_result = search.search_fit(input_data=[dataset], time_limit=30)
best_scores = search.evaluate(best_runtime.pipeline).scores
```

Manual D3M pipelines use `Pipeline`, `PrimitiveStep`, `index.get_primitive(...)`, `set_training_data`, `fit`, `produce`, and `produce_score`. Read `references/tods-api-map.md` before constructing primitive paths.

## Method Choice

- Use default pipeline when validating install/data conversion or establishing a baseline.
- Use AutoML `BruteForceSearch` when the goal is pipeline search under a time budget.
- Use point-wise PyOD wrapper primitives for timestamp-level outlier labels on prepared features.
- Use `KDiscordODetect`, `MatrixProfile`, `DeepLog`, `Telemanom`, `LSTMODetect`, or `DAGMM` for pattern-wise or sequence-model style detection only when their windowing assumptions match the task.
- Use `SystemWiseDetection` and `Ensemble` only when the task is comparing sets of series or combining detector outputs.
- Use feature-analysis primitives when time/frequency/statistical features are needed inside the pipeline; fit them inside each temporal fold.

## Evaluation and Plotting

- Official examples use `F1_MACRO`; for anomaly labels also report precision, recall, F1 for the anomaly class, PR-AUC/average precision, ROC-AUC when valid, precision@N, false positives, and alert volume.
- For subsequence outputs, map each subsequence score to start/end timestamps before scoring.
- TODS docs do not document a dedicated plotting API. Plot original series, output labels, anomaly scores, and subsequence windows with matplotlib or Plotly outside TODS.

## Anti-Leakage Rules

- Never random split time-indexed data for time-series anomaly detection. Use chronological or rolling-origin validation.
- Fit validation, imputation, scaling, smoothing, decomposition, feature extraction, detectors, AutoML search, thresholds, and reinforcement filters only on train/reference windows.
- Build lag/rolling/subsequence features from past/current values only for online detection; centered or full-series windows are retrospective.
- Tune `target_index`, metric, primitive choices, contamination, window sizes, thresholds, and search budget on validation only.
- If labels are used by `generate_problem`/evaluation, keep them out of `Attribute` features and final test selection.
- For panels or system-wise detection, split by time and preserve entity boundaries; do not let future behavior of one series set thresholds for another unless that is production-available.

## Common Errors

- Treating TODS as a maintained modern PyOD replacement; it is an older pipeline system with pinned legacy dependencies.
- Passing raw timestamps or labels as detector features.
- Running AutoML search on the full dataset before temporal validation.
- Using `F1_MACRO` alone on highly imbalanced anomalies without anomaly-class precision/recall.
- Calling pattern-wise subsequence labels point anomalies without documenting the window-to-point mapping.
- Assuming docs and README examples use the same import paths; verify installed version.

## References

- Read `references/tods-api-map.md` for official modules, primitives, detector inventory, APIs, and documented inconsistencies.
- Read `references/tods-data-workflow.md` for data format, validation, pipeline search, metrics, and leakage controls.
- Read `references/official-sources.md` for official sources consulted.
- Use `scripts/validate_tods_anomaly_input.py` to sanity-check CSV inputs before `generate_dataset` or D3M pipeline construction.

## Ready Checklist

- Task is time-series outlier detection and the desired output unit is point, subsequence, or series-set.
- Data is ordered, numeric where needed, and labels/target column are separated from attributes.
- Pipeline primitives and import paths are documented from official TODS sources.
- Temporal validation is defined before transforms, feature extraction, AutoML, and thresholding.
- Metrics include anomaly-class behavior, not only aggregate F1.
