# Repository Workflow

- Work must be done on the `develop` branch or on feature branches created from `develop`.
- Do not make direct work commits on `main`.
- Merge into `main` only after the work has been reviewed, tested, and confirmed to be ready.
- The assistant must tell the user when the current work is ready to merge into `main`.
- Do not merge into `main` unless the user explicitly asks for it after that readiness check.
- Keep `main` as the stable branch that reflects only validated changes.

# Repository Purpose

- This repo stores skills for AI agents focused on forecasting and time-series libraries.
- The first foundational skill must cover forecasting data preparation before any library-specific modeling skill is created.
- Library-specific skills should be generated from an official GitHub repo, documentation site, or package documentation provided by the user.
- Library-specific skills should depend conceptually on the data-preparation skill: prepare and validate data first, then model.
- Classification skills must depend conceptually on `ts-classification-data-prep`: define samples, labels, shapes, splits, class balance, and leakage-safe preprocessing before classifier-specific work.

# Agent Architecture

- Keep one folder per skill with `SKILL.md`, `references/`, optional `scripts/`, and `agents/openai.yaml`.
- Keep `skills.catalog.yaml` updated for every skill. It is the machine-readable routing and dependency map for agents.
- Agent flow should be prep skill first, task/library skill second, validation and leakage checks throughout.
- Run `python scripts/validate_skill_repo.py` before considering work ready for review or merge.

# Skill Standards

- Review official documentation, README files, examples, notebooks, API references, and relevant official guides before writing a library skill.
- Do not invent APIs, model names, parameters, or capabilities that are not supported by the library documentation.
- Cover supported forecasting models, accepted data types, univariate and multivariate usage, exogenous variables, panel or multiple series support, probabilistic forecasts, plotting, metrics, validation, and anti data leakage rules when applicable.
- Keep each `SKILL.md` concise and practical. Use `references/` for long documentation notes and `scripts/` only for reusable deterministic checks or utilities.
- Every forecasting skill must include temporal validation practices and explicit anti-leakage safeguards.
