#!/usr/bin/env python3
"""Validate a prepared dataframe against Prophet's documented data contract."""

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
        "prophet_contract_check.py requires pandas. Install it with: python -m pip install pandas"
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
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def check_contract(args: argparse.Namespace) -> dict[str, Any]:
    df = read_frame(Path(args.path))
    time_col = args.time_col
    target_col = args.target_col
    regressors = parse_csv_list(args.regressors)
    conditions = parse_csv_list(args.conditions)
    errors: list[str] = []
    warnings: list[str] = []

    report: dict[str, Any] = {
        "input": {"path": args.path, "rows": len(df), "columns": len(df.columns)},
        "required_columns": {"time_col": time_col, "target_col": target_col},
        "regressors": regressors,
        "conditions": conditions,
        "growth": args.growth,
        "errors": errors,
        "warnings": warnings,
    }

    for col in [time_col, target_col]:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    for col in regressors + conditions:
        if col not in df.columns:
            errors.append(f"Missing requested future column: {col}")
    if errors:
        return report

    ds = pd.to_datetime(df[time_col], errors="coerce")
    report["time"] = {
        "parse_nulls": int(ds.isna().sum()),
        "duplicate_timestamps": int(ds.duplicated(keep=False).sum()),
        "is_monotonic_increasing": bool(ds.dropna().is_monotonic_increasing),
        "min": ds.min().isoformat() if ds.notna().any() else None,
        "max": ds.max().isoformat() if ds.notna().any() else None,
    }
    if ds.isna().any():
        errors.append("Column ds/time has unparseable or missing timestamps.")
    if getattr(ds.dt, "tz", None) is not None:
        errors.append("Prophet Python does not support timezone-aware ds; convert timezone then remove tz.")
    if ds.duplicated(keep=False).any():
        errors.append("Duplicate ds timestamps found; Prophet expects one row per timestamp per model.")
    if not ds.dropna().is_monotonic_increasing:
        warnings.append("Timestamps are not sorted; sort by ds before fitting.")

    y = pd.to_numeric(df[target_col], errors="coerce")
    inf_count = int((y == float("inf")).sum() + (y == float("-inf")).sum())
    report["target"] = {
        "numeric_nulls_after_parse": int(y.isna().sum()),
        "infinite_values": inf_count,
        "missing_rate": float(y.isna().mean()) if len(y) else 0.0,
    }
    if inf_count:
        errors.append("Infinite values found in target y.")
    if y.notna().sum() < 2:
        errors.append("Target y has fewer than two non-missing numeric values.")

    if args.freq and ds.notna().sum() >= 3:
        try:
            inferred = pd.infer_freq(ds.dropna().drop_duplicates().sort_values())
        except ValueError:
            inferred = None
        report["frequency"] = {"expected": args.freq, "inferred": inferred}
        if inferred and inferred != args.freq:
            warnings.append(f"Inferred frequency {inferred!r} differs from expected {args.freq!r}.")

    if args.growth == "logistic":
        if "cap" not in df.columns:
            errors.append('growth="logistic" requires cap in history and future dataframes.')
        else:
            cap = pd.to_numeric(df["cap"], errors="coerce")
            floor = pd.to_numeric(df["floor"], errors="coerce") if "floor" in df.columns else 0
            if cap.isna().any():
                errors.append("cap contains missing or non-numeric values.")
            if hasattr(floor, "isna") and floor.isna().any():
                errors.append("floor contains missing or non-numeric values.")
            if (cap <= floor).any():
                errors.append("cap must be greater than floor, which defaults to 0.")

    for col in regressors:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isna().any():
            errors.append(f"Regressor {col!r} has missing or non-numeric values.")

    for col in conditions:
        valid = df[col].isin([True, False])
        if not valid.all():
            errors.append(f"Condition column {col!r} must contain only booleans.")

    report["status"] = "pass" if not errors else "fail"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Prophet dataframe contract.")
    parser.add_argument("path", help="Input CSV, Parquet, JSON, or JSONL file.")
    parser.add_argument("--time-col", default="ds", help="Timestamp column, default ds.")
    parser.add_argument("--target-col", default="y", help="Target column, default y.")
    parser.add_argument("--regressors", help="Comma-separated extra regressor columns.")
    parser.add_argument("--conditions", help="Comma-separated conditional seasonality boolean columns.")
    parser.add_argument("--growth", choices=["linear", "logistic", "flat"], default="linear")
    parser.add_argument("--freq", help="Expected pandas frequency alias.")
    args = parser.parse_args()
    print(json.dumps(clean(check_contract(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
