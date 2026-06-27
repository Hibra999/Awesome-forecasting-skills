# Classification Data Contract

Use this contract before choosing a classification library.

## Required Fields

- `sample_id`: stable identifier for each independent time-series instance.
- `label`: target class for the sample.
- `split`: train, validation, test, or fold identifier.
- `shape`: sample-major shape and axis order.
- `time_policy`: original, resampled, padded, truncated, interpolated, masked, or unequal length.
- `channels`: names, order, units, and sampling rate for each variable.
- `groups`: subject, entity, device, site, or session IDs that must not leak across splits.
- `label_timing`: when the label is known relative to the observed series.

## Shape Rules

- 2D univariate arrays should be documented as `(n_samples, n_timestamps)`.
- 3D arrays must document whether axes are `(n_samples, n_timestamps, n_channels)` or `(n_samples, n_channels, n_timestamps)`.
- Panel or nested formats must still have one label per sample.
- Unequal-length samples need a library-specific representation or an explicit padding/masking rule.

## Output

The prepared output should include the data files, a short contract note, class counts, split counts, and any accepted risks.
