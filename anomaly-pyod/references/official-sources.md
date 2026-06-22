# Official Sources Consulted

- PyOD GitHub repository: <https://github.com/yzhao062/pyod>
- PyOD README raw source: <https://raw.githubusercontent.com/yzhao062/pyod/master/README.rst>
- PyOD documentation home: <https://pyod.readthedocs.io/en/latest/>
- PyOD installation guide: <https://pyod.readthedocs.io/en/latest/install.html>
- PyOD API cheatsheet: <https://pyod.readthedocs.io/en/latest/api_cc.html>
- PyOD tabular detector API: <https://pyod.readthedocs.io/en/latest/pyod.models.tabular.html>
- PyOD time-series detector API: <https://pyod.readthedocs.io/en/latest/pyod.models.timeseries.html>
- PyOD all models/API reference checked for module inventory: <https://pyod.readthedocs.io/en/latest/pyod.models.html>
- PyOD examples: <https://pyod.readthedocs.io/en/latest/example.html>
- PyOD outlier detection 101: <https://pyod.readthedocs.io/en/latest/relevant_knowledge.html>
- PyOD PyPI package: <https://pypi.org/project/pyod/>

Consultation notes:

- PyPI lists `pyod 3.6.1` as latest, released June 17, 2026, requiring Python `>=3.9`.
- Official README describes PyOD 3 as supporting 61 detectors across tabular, time series, graph, text, image, and audio data.
- Official docs define a common detector API with `fit`, `decision_function`, `predict`, `predict_proba`, confidence, rejection, `decision_scores_`, and `labels_`.
- Official time-series docs define input shape `(n_timestamps,)` or `(n_timestamps, n_channels)` and one score per timestamp.
