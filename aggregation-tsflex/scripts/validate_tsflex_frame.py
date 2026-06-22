#!/usr/bin/env python3
"""Validate CSV data before converting it to tsflex pandas inputs.

The script uses only the standard library. It checks wide and long CSV layouts
for required columns, duplicate series names, monotonic time/order, numeric
values when requested, and window/stride compatibility.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


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


def finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def parse_time(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
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


def validate_window_stride(length: int, window_size: int | None, stride: int | None) -> None:
    if window_size is not None:
        if window_size <= 0:
            raise SystemExit("--window-size must be positive")
        if length < window_size:
            raise SystemExit(f"length {length} is shorter than window_size {window_size}")
    if stride is not None and stride <= 0:
        raise SystemExit("--stride must be positive")


def time_warnings(rows: list[dict[str, str]], time_column: str, group_label: str) -> list[str]:
    warnings: list[str] = []
    parsed: list[float | str] = []
    seen: set[str] = set()
    for row in rows:
        raw = row[time_column]
        if invalid_token(raw):
            raise SystemExit(f"group {group_label!r}: time column has empty/NaN/Inf value")
        if raw in seen:
            warnings.append(f"group {group_label!r}: duplicate time {raw!r}")
            break
        seen.add(raw)
        parsed.append(parse_time(raw))

    comparable = all(isinstance(value, (float, int)) for value in parsed)
    if comparable:
        numeric = [float(value) for value in parsed]
        if any(curr < prev for prev, curr in zip(numeric, numeric[1:])):
            warnings.append(f"group {group_label!r}: time/order column is not sorted")
        if len(numeric) >= 3:
            diffs = [round(curr - prev, 12) for prev, curr in zip(numeric, numeric[1:])]
            if len(set(diffs)) > 1:
                warnings.append(
                    f"group {group_label!r}: intervals are irregular; use robust features or approve_sparsity"
                )
    else:
        if any(str(curr) < str(prev) for prev, curr in zip(parsed, parsed[1:])):
            warnings.append(f"group {group_label!r}: time/order column is not lexicographically sorted")
    return warnings


def infer_value_columns(
    rows: list[dict[str, str]],
    excluded: set[str],
    requested: list[str] | None,
) -> list[str]:
    fieldnames = list(rows[0].keys())
    if requested:
        require_columns(set(fieldnames), requested)
        return requested
    values = [column for column in fieldnames if column not in excluded]
    if not values:
        raise SystemExit("no value columns found; pass --value-columns")
    if len(values) != len(set(values)):
        raise SystemExit("duplicate value columns found")
    return values


def validate_numeric(rows: list[dict[str, str]], value_columns: list[str]) -> None:
    for row_number, row in enumerate(rows, start=2):
        for column in value_columns:
            if not finite_number(row[column]):
                raise SystemExit(f"row {row_number}: value column {column!r} must be finite numeric")


def validate_wide(args: argparse.Namespace) -> int:
    rows = parse_csv(Path(args.csv), args.delimiter)
    if not rows:
        raise SystemExit("CSV has no data rows")
    required = [args.time_column] if args.time_column else []
    require_columns(set(rows[0].keys()), required)
    values = infer_value_columns(rows, set(required), args.value_columns)

    if args.numeric:
        validate_numeric(rows, values)
    validate_window_stride(len(rows), args.window_size, args.stride)

    warnings: list[str] = []
    if args.time_column:
        warnings.extend(time_warnings(rows, args.time_column, "__wide__"))
    else:
        warnings.append("no time/order column provided; ensure CSV row order is sorted")

    print(f"OK wide: rows={len(rows)} series={len(values)}")
    for warning in warnings[:20]:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


def validate_long(args: argparse.Namespace) -> int:
    rows = parse_csv(Path(args.csv), args.delimiter)
    if not rows:
        raise SystemExit("CSV has no data rows")
    required = [args.time_column, args.kind_column, args.value_column]
    require_columns(set(rows[0].keys()), required)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=2):
        kind = row[args.kind_column]
        if invalid_token(kind):
            raise SystemExit(f"row {row_number}: kind column must not be empty/NaN/Inf")
        if args.numeric and not finite_number(row[args.value_column]):
            raise SystemExit(f"row {row_number}: value column must be finite numeric")
        groups[kind].append(row)

    warnings: list[str] = []
    for kind, group_rows in groups.items():
        validate_window_stride(len(group_rows), args.window_size, args.stride)
        warnings.extend(time_warnings(group_rows, args.time_column, kind))

    print(f"OK long: rows={len(rows)} series={len(groups)}")
    for warning in warnings[:20]:
        print(f"WARNING: {warning}", file=sys.stderr)
    if len(warnings) > 20:
        print("WARNING: additional warnings omitted", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate CSV data before tsflex aggregation.")
    subparsers = parser.add_subparsers(dest="format", required=True)

    wide = subparsers.add_parser("wide", help="Validate a wide CSV: time/index plus series columns.")
    wide.add_argument("csv")
    wide.add_argument("--time-column", default=None)
    wide.add_argument("--value-columns", nargs="*")
    wide.add_argument("--window-size", type=int)
    wide.add_argument("--stride", type=int)
    wide.add_argument("--delimiter", default=",")
    wide.add_argument("--numeric", action="store_true", help="Require finite numeric values.")
    wide.set_defaults(func=validate_wide)

    long = subparsers.add_parser("long", help="Validate a long CSV: time, kind, value.")
    long.add_argument("csv")
    long.add_argument("--time-column", default="time")
    long.add_argument("--kind-column", default="kind")
    long.add_argument("--value-column", default="value")
    long.add_argument("--window-size", type=int)
    long.add_argument("--stride", type=int)
    long.add_argument("--delimiter", default=",")
    long.add_argument("--numeric", action="store_true", help="Require finite numeric values.")
    long.set_defaults(func=validate_long)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
