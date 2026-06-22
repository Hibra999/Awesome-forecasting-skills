#!/usr/bin/env python3
"""Validate CSV inputs before TSFEL feature extraction.

This script uses only the standard library. It checks numeric value columns,
optional id/time columns, minimum length, window_size/overlap feasibility, and
warns when a time column appears duplicated, unsorted, or unevenly spaced.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from datetime import datetime
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
    return values


def validate_window_args(length: int, min_length: int, window_size: int | None, overlap: float) -> None:
    if length < min_length:
        raise SystemExit(f"length {length} is shorter than minimum {min_length}")
    if not 0 <= overlap < 1:
        raise SystemExit("--overlap must be >= 0 and < 1")
    if window_size is None:
        return
    if window_size <= 0:
        raise SystemExit("--window-size must be positive")
    if length < window_size:
        raise SystemExit(f"length {length} is shorter than window_size {window_size}")
    step = max(1, int(window_size * (1 - overlap)))
    if step <= 0:
        raise SystemExit("window step is zero; reduce --overlap")


def time_warnings(rows: list[dict[str, str]], time_column: str | None, group_label: str) -> list[str]:
    if not time_column:
        return ["no time column provided; ensure row order is already sorted and equally spaced"]

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
            warnings.append(f"group {group_label!r}: time column is not sorted")
        if len(numeric) >= 3:
            diffs = [round(curr - prev, 12) for prev, curr in zip(numeric, numeric[1:])]
            if len(set(diffs)) > 1:
                warnings.append(
                    f"group {group_label!r}: time intervals are not constant; resample before TSFEL"
                )
    else:
        if any(str(curr) < str(prev) for prev, curr in zip(parsed, parsed[1:])):
            warnings.append(f"group {group_label!r}: time column is not lexicographically sorted")

    return warnings


def validate_groups(
    rows: list[dict[str, str]],
    groups: dict[str, list[dict[str, str]]],
    time_column: str | None,
    value_columns: list[str],
    min_length: int,
    window_size: int | None,
    overlap: float,
) -> list[str]:
    for row_number, row in enumerate(rows, start=2):
        for column in value_columns:
            if not finite_number(row[column]):
                raise SystemExit(
                    f"row {row_number}: value column {column!r} must be finite numeric"
                )

    warnings: list[str] = []
    for group_label, group_rows in groups.items():
        validate_window_args(len(group_rows), min_length, window_size, overlap)
        warnings.extend(time_warnings(group_rows, time_column, group_label))
    return warnings


def validate_single(args: argparse.Namespace) -> int:
    rows = parse_csv(Path(args.csv), args.delimiter)
    if not rows:
        raise SystemExit("CSV has no data rows")

    required = [args.time_column] if args.time_column else []
    require_columns(set(rows[0].keys()), required)
    values = infer_value_columns(rows, set(required), args.value_columns)
    warnings = validate_groups(
        rows,
        {"__single__": rows},
        args.time_column,
        values,
        args.min_length,
        args.window_size,
        args.overlap,
    )

    print(f"OK single: rows={len(rows)} value_columns={len(values)}")
    for warning in warnings[:20]:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


def validate_panel(args: argparse.Namespace) -> int:
    rows = parse_csv(Path(args.csv), args.delimiter)
    if not rows:
        raise SystemExit("CSV has no data rows")

    required = [args.id_column]
    if args.time_column:
        required.append(args.time_column)
    require_columns(set(rows[0].keys()), required)
    values = infer_value_columns(rows, set(required), args.value_columns)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=2):
        if invalid_token(row[args.id_column]):
            raise SystemExit(f"row {row_number}: id column must not be empty/NaN/Inf")
        groups[row[args.id_column]].append(row)

    warnings = validate_groups(
        rows,
        groups,
        args.time_column,
        values,
        args.min_length,
        args.window_size,
        args.overlap,
    )

    print(f"OK panel: rows={len(rows)} groups={len(groups)} value_columns={len(values)}")
    for warning in warnings[:20]:
        print(f"WARNING: {warning}", file=sys.stderr)
    if len(warnings) > 20:
        print("WARNING: additional warnings omitted", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate CSV data before TSFEL feature extraction."
    )
    subparsers = parser.add_subparsers(dest="format", required=True)

    single = subparsers.add_parser("single", help="Validate one signal CSV.")
    single.add_argument("csv")
    single.add_argument("--time-column", default=None)
    single.add_argument("--value-columns", nargs="*")
    single.add_argument("--min-length", type=int, default=2)
    single.add_argument("--window-size", type=int)
    single.add_argument("--overlap", type=float, default=0.0)
    single.add_argument("--delimiter", default=",")
    single.set_defaults(func=validate_single)

    panel = subparsers.add_parser("panel", help="Validate per-id signal CSV.")
    panel.add_argument("csv")
    panel.add_argument("--id-column", default="series_id")
    panel.add_argument("--time-column", default=None)
    panel.add_argument("--value-columns", nargs="*")
    panel.add_argument("--min-length", type=int, default=2)
    panel.add_argument("--window-size", type=int)
    panel.add_argument("--overlap", type=float, default=0.0)
    panel.add_argument("--delimiter", default=",")
    panel.set_defaults(func=validate_panel)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
