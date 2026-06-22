# Official Sources Consulted

Consulted on 2026-06-22.

- Time-Series-Library GitHub README: https://github.com/thuml/Time-Series-Library
- Official README classification command and notes under "Getting Started" and project architecture.
- Official `run.py`: https://github.com/thuml/Time-Series-Library/blob/main/run.py
- Official `exp/exp_classification.py`: https://github.com/thuml/Time-Series-Library/blob/main/exp/exp_classification.py
- Official `exp/exp_basic.py`: https://github.com/thuml/Time-Series-Library/blob/main/exp/exp_basic.py
- Official `data_provider/data_factory.py`: https://github.com/thuml/Time-Series-Library/blob/main/data_provider/data_factory.py
- Official `data_provider/data_loader.py`: https://github.com/thuml/Time-Series-Library/blob/main/data_provider/data_loader.py
- Official `data_provider/uea.py`: https://github.com/thuml/Time-Series-Library/blob/main/data_provider/uea.py
- Official `scripts/classification/` recipes: https://github.com/thuml/Time-Series-Library/tree/main/scripts/classification
- Official `requirements.txt`: https://github.com/thuml/Time-Series-Library/blob/main/requirements.txt

Notes:
- The README states that TSLib covers five tasks: long/short forecasting, imputation, anomaly detection, and classification.
- The README's 2026-04 note says maintainers are not actively adding new features and recommend newer benchmarks for current progress; baseline implementations remain correct.
- The classification source reports accuracy only; probability metrics and stratified CV are not documented as native workflows.
