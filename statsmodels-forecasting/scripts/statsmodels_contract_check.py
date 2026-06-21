#!/usr/bin/env python3
"""Validate prepared data against common Statsmodels forecasting contracts."""

from __future__ import annotations

import argparse
import json
import math
import numbers
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "statsmodels_contract_check.py requires pandas. Install it with: python -m pip install pandas"
    ) from exc


def read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise SystemExit(f"Unsupported file type: {suffix}. Use csv, parquet, json, or jsonl.")


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def clean(value: Any) -> Any:
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def group_iterator(df: pd.DataFrame, panel_cols: list[str]):
    if not panel_cols:
        yield "__single_series__", df
        return
    group_key: str | list[str] = panel_cols[0] if len(panel_cols) == 1 else panel_cols
    for key, group in df.groupby(group_key, dropna=False, sort=False):
        if isinstance(key, tuple):
            name = "|".join(str(part) for part in key)
        else:
            name = str(key)
        yield name, group


def parse_time(series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    raw = series.dropna().astype(str)
    tz_rate = 0.0
    if not raw.empty:
        tz_rate = float(raw.str.contains(r"(?:Z$|[+-]\d{2}:?\d{2}$|\bUTC\b|\bGMT\b)", regex=True).mean())
    use_utc = tz_rate > 0
    parsed = pd.to_datetime(series, errors="coerce", utc=use_utc)
    if getattr(parsed.dt, "tz", None) is not None:
        check_values = parsed.dt.tz_convert(None)
    else:
        check_values = parsed
    metadata = {
        "parse_nulls": int(parsed.isna().sum()),
        "parse_null_rate": float(parsed.isna().mean()) if len(parsed) else 0.0,
        "timezone_token_rate": tz_rate,
        "timezone_status": "timezone_tokens_present_converted_to_utc" if use_utc else "naive_or_not_encoded",
        "min": check_values.min().isoformat() if check_values.notna().any() else None,
        "max": check_values.max().isoformat() if check_values.notna().any() else None,
    }
    return check_values, metadata


def check_numeric_columns(
    df: pd.DataFrame, cols: list[str], label: str, errors: list[str]
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for col in cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        inf_count = int((numeric == float("inf")).sum() + (numeric == float("-inf")).sum())
        non_numeric = int((numeric.isna() & df[col].notna()).sum())
        raw_missing = int(df[col].isna().sum())
        missing = int(numeric.isna().sum())
        report[col] = {
            "missing_or_non_numeric": missing,
            "raw_missing_values": raw_missing,
            "non_numeric_values": non_numeric,
            "infinite_values": inf_count,
            "non_missing_numeric": int(numeric.notna().sum()),
        }
        if raw_missing:
            errors.append(f"{label} column {col!r} contains missing values.")
        if non_numeric:
            errors.append(f"{label} column {col!r} contains non-numeric values.")
        if inf_count:
            errors.append(f"{label} column {col!r} contains infinite values.")
        if numeric.notna().sum() < 2 and label == "target":
            errors.append(f"Target column {col!r} has fewer than two numeric observations.")
    return report


def infer_frequency_report(
    df: pd.DataFrame, panel_cols: list[str], expected_freq: str | None, max_groups: int
) -> list[dict[str, Any]]:
    per_group: list[dict[str, Any]] = []
    for index, (name, group) in enumerate(group_iterator(df, panel_cols)):
        if index >= max_groups:
            break
        times = group["_statsmodels_time"].dropna().drop_duplicates().sort_values()
        inferred = None
        missing_periods = None
        if len(times) >= 3:
            try:
                inferred = pd.infer_freq(times)
            except ValueError:
                inferred = None
        if expected_freq and len(times) >= 2:
            try:
                expected_index = pd.date_range(times.iloc[0], times.iloc[-1], freq=expected_freq)
                missing_periods = int(len(expected_index.difference(pd.DatetimeIndex(times))))
            except ValueError:
                missing_periods = None
        per_group.append(
            {
                "series": name,
                "rows": int(len(group)),
                "unique_timestamps": int(len(times)),
                "is_monotonic_increasing": bool(times.is_monotonic_increasing),
                "inferred_freq": inferred,
                "expected_freq": expected_freq,
                "missing_periods_at_expected_freq": missing_periods,
            }
        )
    return per_group


def check_contract(args: argparse.Namespace) -> dict[str, Any]:
    df = read_frame(Path(args.path))
    target_cols = parse_csv_list(args.target_cols)
    exog_cols = parse_csv_list(args.exog_cols)
    panel_cols = parse_csv_list(args.panel_cols)
    required = [args.time_col, *target_cols, *exog_cols, *panel_cols]
    errors: list[str] = []
    warnings: list[str] = []

    report: dict[str, Any] = {
        "input": {"path": args.path, "rows": len(df), "columns": len(df.columns)},
        "contract": {
            "time_col": args.time_col,
            "target_cols": target_cols,
            "exog_cols": exog_cols,
            "panel_cols": panel_cols,
            "freq": args.freq,
            "horizon": args.horizon,
            "model_kind": args.model_kind,
        },
        "errors": errors,
        "warnings": warnings,
    }

    if not target_cols:
        errors.append("At least one target column is required via --target-cols.")
    for col in required:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    if errors:
        report["status"] = "fail"
        return report

    if args.horizon is not None and args.horizon < 1:
        errors.append("--horizon must be a positive integer.")
    if len(target_cols) > 1 and args.model_kind == "univariate":
        errors.append("Multiple target columns require --model-kind multivariate.")
    if panel_cols:
        warnings.append(
            "Statsmodels is not a native global panel forecaster; fit one model per panel group unless targets are truly jointly endogenous."
        )
    if exog_cols and not args.future_exog_confirmed:
        warnings.append(
            "Exogenous columns were supplied; confirm future values are known at every forecast cutoff and horizon step."
        )

    times, time_meta = parse_time(df[args.time_col])
    df = df.copy()
    df["_statsmodels_time"] = times
    report["time"] = time_meta
    if times.isna().any():
        errors.append("Time column has missing or unparseable timestamps.")
    if time_meta["timezone_token_rate"] > 0:
        warnings.append("Timezone tokens were found; normalize all data to one timezone before splitting and forecasting.")

    duplicate_keys = panel_cols + ["_statsmodels_time"]
    duplicate_count = int(df.duplicated(duplicate_keys, keep=False).sum())
    report["duplicates"] = {"duplicate_rows_for_panel_time_key": duplicate_count}
    if duplicate_count:
        errors.append("Duplicate timestamps found within the modeled series key.")

    report["targets"] = check_numeric_columns(df, target_cols, "target", errors)
    report["exog"] = check_numeric_columns(df, exog_cols, "exog", errors)

    frequency = infer_frequency_report(df, panel_cols, args.freq, args.max_groups_report)
    report["frequency_by_group"] = frequency
    for item in frequency:
        if args.freq and item["inferred_freq"] and item["inferred_freq"] != args.freq:
            warnings.append(
                f"Series {item['series']!r} inferred frequency {item['inferred_freq']!r} differs from expected {args.freq!r}."
            )
        if args.freq and item["missing_periods_at_expected_freq"]:
            warnings.append(
                f"Series {item['series']!r} has {item['missing_periods_at_expected_freq']} gaps at expected frequency {args.freq!r}."
            )
        needed = max(args.max_lag, args.seasonal_period, 0) + (args.horizon or 0) + 5
        if needed > 5 and item["unique_timestamps"] < needed:
            warnings.append(
                f"Series {item['series']!r} may be too short: {item['unique_timestamps']} timestamps for horizon/lag/seasonality requirement {needed}."
            )

    if args.require_fixed_freq:
        missing_fixed = [item["series"] for item in frequency if item["inferred_freq"] is None]
        if missing_fixed:
            errors.append(
                "Fixed frequency could not be inferred for: "
                + ", ".join(missing_fixed[:10])
                + ("..." if len(missing_fixed) > 10 else "")
            )

    report["status"] = "pass" if not errors else "fail"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a prepared dataset for Statsmodels forecasting.")
    parser.add_argument("path", help="Input CSV, Parquet, JSON, or JSONL file.")
    parser.add_argument("--time-col", default="ds", help="Timestamp column, default ds.")
    parser.add_argument("--target-cols", default="y", help="Comma-separated target/endog columns.")
    parser.add_argument("--exog-cols", help="Comma-separated exogenous regressor columns.")
    parser.add_argument("--panel-cols", help="Comma-separated panel/group columns.")
    parser.add_argument("--freq", help="Expected pandas frequency alias, for example D, MS, Q, or H.")
    parser.add_argument("--horizon", type=int, help="Forecast horizon in periods.")
    parser.add_argument("--max-lag", type=int, default=0, help="Largest planned lag feature/order lookback.")
    parser.add_argument("--seasonal-period", type=int, default=0, help="Main seasonal period in rows.")
    parser.add_argument(
        "--model-kind",
        choices=["univariate", "multivariate", "panel-per-series"],
        default="univariate",
        help="Intended Statsmodels modeling shape.",
    )
    parser.add_argument(
        "--future-exog-confirmed",
        action="store_true",
        help="Acknowledge that future exog is known for the full forecast horizon.",
    )
    parser.add_argument(
        "--require-fixed-freq",
        action="store_true",
        help="Fail if fixed frequency cannot be inferred for checked series.",
    )
    parser.add_argument("--max-groups-report", type=int, default=20, help="Maximum panel groups to inspect.")
    args = parser.parse_args()
    print(json.dumps(clean(check_contract(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
