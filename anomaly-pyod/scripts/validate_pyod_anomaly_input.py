#!/usr/bin/env python3
"""Validate CSV input shape for PyOD anomaly detection workflows."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--feature-columns", nargs="+", required=True)
    parser.add_argument("--time-column")
    parser.add_argument("--entity-column")
    parser.add_argument("--label-column")
    parser.add_argument("--mode", choices=["tabular", "time-series"], default="time-series")
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument("--require-sorted", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def parse_time(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        return list(reader)


def validate_columns(rows: list[dict[str, str]], args: argparse.Namespace) -> list[str]:
    if not rows:
        return ["CSV has no data rows"]
    columns = set(rows[0])
    required = set(args.feature_columns)
    for optional in (args.time_column, args.entity_column, args.label_column):
        if optional:
            required.add(optional)
    missing = sorted(required - columns)
    return [f"missing columns: {', '.join(missing)}"] if missing else []


def validate_features(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for column in args.feature_columns:
        unique_values = set()
        for row_number, row in enumerate(rows, start=2):
            raw = row.get(column, "").strip()
            if raw == "":
                if args.allow_missing:
                    warnings.append(f"column {column!r} has missing values; impute inside train folds only")
                    continue
                errors.append(f"row {row_number}: column {column!r} is missing")
                continue
            try:
                value = float(raw)
            except ValueError:
                errors.append(f"row {row_number}: column {column!r} is not numeric")
                continue
            if not math.isfinite(value):
                errors.append(f"row {row_number}: column {column!r} is not finite")
            unique_values.add(value)
        if len(unique_values) <= 1:
            warnings.append(f"feature {column!r} is constant or nearly empty")
    return errors, sorted(set(warnings))


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
    if args.mode == "time-series" and not args.time_column:
        errors.append("--time-column is required for time-series mode")
        return errors, warnings
    if not args.time_column:
        return errors, warnings

    grouped: dict[str, list[datetime]] = {}
    for row_number, row in enumerate(rows, start=2):
        key = row.get(args.entity_column, "__all__") if args.entity_column else "__all__"
        try:
            grouped.setdefault(key, []).append(parse_time(row.get(args.time_column, "")))
        except ValueError:
            errors.append(f"row {row_number}: time value is not ISO-8601 parseable")

    if errors:
        return errors, warnings

    for key, times in grouped.items():
        if len(times) != len(set(times)):
            errors.append(f"entity {key!r}: duplicate timestamps")
        if any(curr < prev for prev, curr in zip(times, times[1:])):
            errors.append(f"entity {key!r}: timestamps are not sorted ascending")
        if args.require_sorted and any(curr <= prev for prev, curr in zip(times, times[1:])):
            errors.append(f"entity {key!r}: timestamps must be strictly increasing")
        if len(times) >= 3:
            deltas = [curr - prev for prev, curr in zip(times, times[1:])]
            if len(set(deltas)) > 1:
                warnings.append(f"entity {key!r}: irregular sampling; document before window features")
    return errors, warnings


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    if not 0 < args.contamination <= 0.5:
        errors.append("--contamination must be in (0, 0.5]")

    if not args.csv_path.exists():
        errors.append(f"CSV not found: {args.csv_path}")
        rows: list[dict[str, str]] = []
    else:
        try:
            rows = read_rows(args.csv_path)
        except Exception as exc:  # noqa: BLE001 - CLI should report parser failures.
            errors.append(str(exc))
            rows = []

    if rows:
        errors.extend(validate_columns(rows, args))
        if not errors:
            feature_errors, feature_warnings = validate_features(rows, args)
            label_errors, positives = validate_labels(rows, args.label_column)
            time_errors, time_warnings = validate_time(rows, args)
            errors.extend(feature_errors)
            errors.extend(label_errors)
            errors.extend(time_errors)
            warnings.extend(feature_warnings)
            warnings.extend(time_warnings)
        else:
            positives = None
    else:
        positives = None

    if args.mode == "time-series" and len(rows) < 10:
        warnings.append("time-series mode has fewer than 10 rows; most detectors need more history")

    for warning in sorted(set(warnings)):
        print(f"WARN: {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    label_text = "no labels"
    if positives is not None:
        label_text = f"{positives} anomalies in label column"
    print(
        "OK: "
        f"{len(rows)} rows, mode={args.mode}, "
        f"features={','.join(args.feature_columns)}, contamination={args.contamination}, {label_text}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
