# Official Sources Consulted

- GitHub repository: https://github.com/hfawaz/dl-4-tsc
- README: https://github.com/hfawaz/dl-4-tsc/blob/master/README.md
- Raw README: https://raw.githubusercontent.com/hfawaz/dl-4-tsc/master/README.md
- Experiment entry point: https://github.com/hfawaz/dl-4-tsc/blob/master/main.py
- Utilities and metrics: https://github.com/hfawaz/dl-4-tsc/blob/master/utils/utils.py
- Classifier constants: https://github.com/hfawaz/dl-4-tsc/blob/master/utils/constants.py
- Classifier source directory: https://github.com/hfawaz/dl-4-tsc/tree/master/classifiers
- Paper linked by README: https://arxiv.org/abs/1809.04356
- InceptionTime paper linked by README: https://arxiv.org/abs/1909.04939

## Documented Gaps

- The repository is research code and does not document a package-level Python API.
- The repository does not document sklearn-compatible cross-validation, `predict_proba`, calibration, probability intervals, or residual diagnostics.
- The repository does not document native unequal-length classifier inputs; conversion/interpolation is handled before model training.
- The README contains legacy command/table details, while current source constants include `inception` and use `UCRArchive_2018`.
