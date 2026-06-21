# Time-Series-Library Model Map

Use this reference to choose model names exactly as documented in the official README, scripts, and `models/` tree. TSLib imports model modules from `models/`; task support still depends on the model implementation and the official script patterns.

## Forecasting and Benchmark Models

The official README's compared forecasting models and leaderboard families include:

- `TimeXer`
- `TimeMixer`
- `TSMixer`
- `iTransformer`
- `PatchTST`
- `TimesNet`
- `DLinear`
- `LightTS`
- `ETSformer`
- `Nonstationary_Transformer`
- `FEDformer`
- `Pyraformer`
- `Autoformer`
- `Informer`
- `Reformer`
- `Transformer`

The official `models/` tree also includes these forecasting-oriented baselines or architecture files:

- `Crossformer`
- `FiLM`
- `FreTS`
- `Koopa`
- `MICN`
- `MSGNet`
- `Mamba`
- `MambaSimple`
- `MambaSingleLayer`
- `MultiPatchFormer`
- `PAttn`
- `SCINet`
- `SegRNN`
- `TemporalFusionTransformer`
- `TiDE`
- `TimeFilter`
- `WPMixer`

`KANAD` appears in the official model tree and README as KAN-AD for anomaly detection. Do not present it as a forecasting model unless the local source has been changed and verified.

## Large Time Series Models and Zero-Shot Forecasting

The README documents zero-shot evaluation for Large Time Series Models. The browsed official `models/` tree contains:

- `Chronos`
- `Chronos2`
- `Moirai`
- `Sundial`
- `TiRex`
- `TimeMoE`
- `TimesFM`

The README also mentions `Toto`, but the browsed official `models/` tree did not show `Toto.py`. Treat this as README-mentioned but not directly runnable by `--model Toto` unless the checked-out repository contains the corresponding module.

## Dependency Notes

- Core install uses Python 3.11, Torch 2.5.1 with a CUDA-specific wheel, and `requirements.txt`.
- `Mamba.py` requires `mamba_ssm`; the README warns the wheel is Linux and CUDA-version specific.
- `Moirai.py` requires `uni2ts --no-deps`.
- Foundation model workflows may require model downloads, Hugging Face access, or extra packages from `requirements.txt`.

## Task Caveats

- TSLib covers long-term forecasting, short-term forecasting, imputation, anomaly detection, and classification. This skill covers forecasting; verify source before reusing a model for other tasks.
- The README's `--model` help text is not exhaustive. Prefer the actual `models/` files and official scripts.
- Some model names in papers differ from file names, for example Non-stationary Transformer is exposed as `Nonstationary_Transformer`, and Temporal Fusion Transformer is exposed as `TemporalFusionTransformer`.
- Not every model supports `long_term_forecast`, `short_term_forecast`, `zero_shot_forecast`, exogenous variables, or all feature modes. Prefer an official script for the same task/model/data family before adapting.
- New model development is documented by adding a model file under `./models` and a matching script under `./scripts`; do not invent that a custom model is supported until both are present and tested.

## Selection Heuristics

- Start with a simple official baseline such as `DLinear`, `PatchTST`, `TimesNet`, or the model named in the user request.
- Use `TimeXer` when the explicit goal is exogenous-variable forecasting and official `scripts/exogenous_forecast/` patterns fit the data.
- Use `m4` plus `short_term_forecast` only for M4-style short-term benchmark workflows.
- Use `zero_shot_forecast` only for documented LTSM model files and inference-style runs.
- For production-like use, validate against simpler baselines outside TSLib and document that TSLib is a research benchmark codebase.
