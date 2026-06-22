#!/usr/bin/env python3
"""Validate CSV inputs before Kats TSFeatures extraction.

This lightweight checker uses only the standard library. It validates the
contract needed before constructing Kats TimeSeriesData: time/id columns,
finite numeric value columns, duplicate timestamps, and minimum series length.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


def finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def invalid_token(value: str) -> bool:
    return value.strip().lower() in {
        "",
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }


def sortable_value(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def parse_csv(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise SystemExit(f"{path}: missing header row")
        return list(reader)


def require_columns(fieldnames: set[str], columns: list[str]) -> None:
    missing = [column for column in columns if column not in fieldnames]
    if missing:
        raise SystemExit(f"missing required columns: {', '.join(missing)}")


def value_columns(
    rows: list[dict[str, str]],
    excluded: set[str],
    requested: list[str] | None,
) -> list[str]:
    fieldnames = list(rows[0].keys())
    if requested:
        require_columns(set(fieldnames), requested)
        return requested
    inferred = [column for column in fieldnames if column not in excluded]
    if not inferred:
        raise SystemExit("no value columns found; pass --value-columns")
    return inferred


def validate_rows(
    rows: list[dict[str, str]],
    groups: dict[str, list[dict[str, str]]],
    time_column: str,
    values: list[str],
    min_length: int,
) -> list[str]:
    warnings: list[str] = []

    for row_number, row in enumerate(rows, start=2):
        if invalid_token(row[time_column]):
            raise SystemExit(f"row {row_number}: time column must not be empty/NaN/Inf")
        for column in values:
            if not finite_number(row[column]):
                raise SystemExit(
                    f"row {row_number}: value column {column!r} must be finite numeric"
                )

    for group_id, group_rows in groups.items():
        if len(group_rows) < min_length:
            raise SystemExit(
                f"group {group_id!r}: length {len(group_rows)} is shorter than {min_length}"
            )

        seen_times: set[str] = set()
        previous: float | str | None = None
        for row in group_rows:
            timestamp = row[time_column]
            if timestamp in seen_times:
                warnings.append(f"group {group_id!r}: duplicate time {timestamp!r}")
                break
            seen_times.add(timestamp)

            current = sortable_value(timestamp)
            if (
                previous is not None
                and type(previous) is type(current)
                and current < previous
            ):
                warnings.append(f"group {group_id!r}: time column is not sorted")
                break
            previous = current

    return warnings


def validate_single(args: argparse.Namespace) -> int:
    rows = parse_csv(Path(args.csv), args.delimiter)
    if not rows:
        raise SystemExit("CSV has no data rows")

    require_columns(set(rows[0].keys()), [args.time_column])
    values = value_columns(rows, {args.time_column}, args.value_columns)
    groups = {"__single__": rows}
    warnings = validate_rows(rows, groups, args.time_column, values, args.min_length)

    print(f"OK single: rows={len(rows)} value_columns={len(values)}")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


def validate_panel(args: argparse.Namespace) -> int:
    rows = parse_csv(Path(args.csv), args.delimiter)
    if not rows:
        raise SystemExit("CSV has no data rows")

    require_columns(set(rows[0].keys()), [args.id_column, args.time_column])
    values = value_columns(rows, {args.id_column, args.time_column}, args.value_columns)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=2):
        if invalid_token(row[args.id_column]):
            raise SystemExit(f"row {row_number}: id column must not be empty/NaN/Inf")
        groups[row[args.id_column]].append(row)

    warnings = validate_rows(rows, groups, args.time_column, values, args.min_length)

    print(f"OK panel: rows={len(rows)} groups={len(groups)} value_columns={len(values)}")
    for warning in warnings[:20]:
        print(f"WARNING: {warning}", file=sys.stderr)
    if len(warnings) > 20:
        print("WARNING: additional warnings omitted", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate CSV data before Kats TSFeatures extraction."
    )
    subparsers = parser.add_subparsers(dest="format", required=True)

    single = subparsers.add_parser("single", help="Validate one TimeSeriesData CSV.")
    single.add_argument("csv")
    single.add_argument("--time-column", default="time")
    single.add_argument("--value-columns", nargs="*")
    single.add_argument("--min-length", type=int, default=5)
    single.add_argument("--delimiter", default=",")
    single.set_defaults(func=validate_single)

    panel = subparsers.add_parser("panel", help="Validate per-id TimeSeriesData CSV.")
    panel.add_argument("csv")
    panel.add_argument("--id-column", default="series_id")
    panel.add_argument("--time-column", default="time")
    panel.add_argument("--value-columns", nargs="*")
    panel.add_argument("--min-length", type=int, default=5)
    panel.add_argument("--delimiter", default=",")
    panel.set_defaults(func=validate_panel)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
