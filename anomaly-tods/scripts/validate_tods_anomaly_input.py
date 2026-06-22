#!/usr/bin/env python3
"""Validate CSV input shape for TODS time-series outlier detection."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path


SCENARIOS = {"point-wise", "pattern-wise", "system-wise"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--time-column", required=True)
    parser.add_argument("--feature-columns", nargs="+", required=True)
    parser.add_argument("--label-column")
    parser.add_argument("--entity-column")
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="point-wise")
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def parse_time(raw: str) -> datetime:
    raw = raw.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    return datetime.fromisoformat(raw)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        return reader.fieldnames, list(reader)


def validate_columns(fieldnames: list[str], rows: list[dict[str, str]], args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if not rows:
        return ["CSV has no data rows"]
    required = {args.time_column, *args.feature_columns}
    if args.label_column:
        required.add(args.label_column)
    if args.entity_column:
        required.add(args.entity_column)
    missing = sorted(required - set(fieldnames))
    if missing:
        errors.append(f"missing columns: {', '.join(missing)}")
    if args.target_index is not None and not 0 <= args.target_index < len(fieldnames):
        errors.append("--target-index is outside CSV column range")
    if args.target_index is not None and args.label_column:
        indexed = fieldnames[args.target_index]
        if indexed != args.label_column:
            errors.append(f"--target-index points to {indexed!r}, not label column {args.label_column!r}")
    return errors


def validate_numeric(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for column in args.feature_columns:
        values = set()
        for row_number, row in enumerate(rows, start=2):
            raw = row.get(column, "").strip()
            if raw == "":
                if args.allow_missing:
                    warnings.append(f"feature {column!r} has missing values; impute inside train folds only")
                    continue
                errors.append(f"row {row_number}: feature {column!r} is missing")
                continue
            try:
                value = float(raw)
            except ValueError:
                errors.append(f"row {row_number}: feature {column!r} is not numeric")
                continue
            if not math.isfinite(value):
                errors.append(f"row {row_number}: feature {column!r} is not finite")
            values.add(value)
        if len(values) <= 1:
            warnings.append(f"feature {column!r} is constant or nearly empty")
    return errors, warnings


def validate_labels(rows: list[dict[str, str]], label_column: str | None) -> tuple[list[str], int | None]:
    if not label_column:
        return [], None
    errors: list[str] = []
    positives = 0
    allowed = {"0", "1", "False", "True", "false", "true"}
    for row_number, row in enumerate(rows, start=2):
        raw = row.get(label_column, "").strip()
        if raw not in allowed:
            errors.append(f"row {row_number}: label {raw!r} is not binary")
        elif raw in {"1", "True", "true"}:
            positives += 1
    return errors, positives


def validate_time(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    grouped: dict[str, list[datetime]] = {}
    for row_number, row in enumerate(rows, start=2):
        key = row.get(args.entity_column, "__all__") if args.entity_column else "__all__"
        try:
            grouped.setdefault(key, []).append(parse_time(row.get(args.time_column, "")))
        except ValueError:
            errors.append(f"row {row_number}: time value is not ISO-8601 parseable")
    if errors:
        return errors, warnings

    for entity, times in grouped.items():
        if len(times) != len(set(times)):
            errors.append(f"entity {entity!r}: duplicate timestamps")
        if any(curr <= prev for prev, curr in zip(times, times[1:])):
            errors.append(f"entity {entity!r}: timestamps must be strictly increasing")
        if len(times) >= 3:
            deltas = [curr - prev for prev, curr in zip(times, times[1:])]
            if len(set(deltas)) > 1:
                warnings.append(f"entity {entity!r}: irregular sampling; use validation/continuity primitives carefully")
    return errors, warnings


def validate_scenario(row_count: int, args: argparse.Namespace) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if args.step_size <= 0:
        errors.append("--step-size must be positive")
    if args.scenario in {"pattern-wise", "system-wise"} and args.window_size is None:
        errors.append(f"--window-size is required for {args.scenario}")
    if args.window_size is not None:
        if args.window_size <= 0:
            errors.append("--window-size must be positive")
        elif row_count <= args.window_size:
            errors.append("CSV has too few rows for the selected window size")
        elif row_count < 3 * args.window_size:
            warnings.append("few windows relative to --window-size; validation may be unstable")
    if args.scenario == "system-wise" and not args.entity_column:
        warnings.append("system-wise detection usually needs --entity-column to identify series sets")
    return errors, warnings


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    positives: int | None = None

    if not args.csv_path.exists():
        errors.append(f"CSV not found: {args.csv_path}")
        fieldnames: list[str] = []
        rows: list[dict[str, str]] = []
    else:
        try:
            fieldnames, rows = read_rows(args.csv_path)
        except Exception as exc:  # noqa: BLE001 - CLI should report parser failures.
            errors.append(str(exc))
            fieldnames, rows = [], []

    if rows:
        errors.extend(validate_columns(fieldnames, rows, args))
        if not errors:
            numeric_errors, numeric_warnings = validate_numeric(rows, args)
            label_errors, positives = validate_labels(rows, args.label_column)
            time_errors, time_warnings = validate_time(rows, args)
            scenario_errors, scenario_warnings = validate_scenario(len(rows), args)
            errors.extend(numeric_errors + label_errors + time_errors + scenario_errors)
            warnings.extend(numeric_warnings + time_warnings + scenario_warnings)

    for warning in sorted(set(warnings)):
        print(f"WARN: {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    label_text = "no label column"
    if positives is not None:
        label_text = f"{positives} outlier labels"
    print(
        "OK: "
        f"{len(rows)} rows, scenario={args.scenario}, "
        f"features={','.join(args.feature_columns)}, {label_text}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
