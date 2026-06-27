# Validation and Leakage

Time-series classification leaks easily because one entity can produce many overlapping samples.

## Split Policy

- Use official benchmark splits when reproducing published datasets.
- Use group splits when samples share a subject, device, entity, patient, site, or session.
- Use chronological splits when the future deployment setting predicts later samples from earlier data.
- Do not let overlapping windows from the same original series cross train and validation/test.
- Do not tune on the final test split.

## Preprocessing Boundaries

Fit these only on train, or inside each cross-validation fold:

- scaling and normalization;
- imputation values;
- padding/truncation length unless it is fixed external metadata;
- resampling parameters;
- symbolic vocabularies and discretization bins;
- shapelets, ROCKET kernels, feature selectors, PCA, or any learned transform;
- class weights and threshold calibration.

## Label Checks

- Confirm labels are not derived from future observations that would be unavailable at prediction time.
- Keep labels aligned after sorting, padding, filtering, or tensor conversion.
- Report classes missing from a split instead of silently dropping them.
