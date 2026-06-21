# Official Sources Consulted

Checked on 2026-06-21.

- THUML Time-Series-Library GitHub README: https://github.com/thuml/Time-Series-Library
- Official `models/` tree: https://github.com/thuml/Time-Series-Library/tree/main/models
- Official `scripts/` tree: https://github.com/thuml/Time-Series-Library/tree/main/scripts
- Official `tutorial/` tree, including `TimesNet_tutorial.ipynb`: https://github.com/thuml/Time-Series-Library/tree/main/tutorial
- `run.py` CLI/task dispatcher: https://github.com/thuml/Time-Series-Library/blob/main/run.py
- Data loaders: https://github.com/thuml/Time-Series-Library/blob/main/data_provider/data_loader.py
- Data factory: https://github.com/thuml/Time-Series-Library/blob/main/data_provider/data_factory.py
- Long-term forecasting experiment: https://github.com/thuml/Time-Series-Library/blob/main/exp/exp_long_term_forecasting.py
- Short-term forecasting experiment: https://github.com/thuml/Time-Series-Library/blob/main/exp/exp_short_term_forecasting.py
- Metrics implementation: https://github.com/thuml/Time-Series-Library/blob/main/utils/metrics.py
- Requirements file: https://github.com/thuml/Time-Series-Library/blob/main/requirements.txt

Documentation notes:

- The official repo README and repository source/scripts are the primary docs. No separate stable API reference site was identified in the consulted official materials.
- The official tutorial folder contains a TimesNet notebook and images, not broad API documentation for every forecasting model.
- Model coverage was based on both the README model lists and the current official `models/` tree. When those conflicted, the skill calls out the mismatch explicitly, for example README-mentioned `Toto` without a browsed `Toto.py` model file.
- The skill intentionally does not document unsupported APIs such as a universal probabilistic interval interface, a generic panel container, or a universal rolling-origin backtesting API.
