#!/usr/bin/env python3
"""Validate simple CSV inputs before tsfresh feature extraction.

This script is intentionally lightweight and uses only the Python standard
library. It checks column presence, finite special columns, finite numeric
values, id counts, and sort order warnings for flat or stacked tsfresh inputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def invalid_special_token(value: str) -> bool:
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


def assert_clean_special_columns(rows: list[dict[str, str]], columns: list[str]) -> None:
    for row_number, row in enumerate(rows, start=2):
        for column in columns:
            value = row.get(column, "")
            if invalid_special_token(value):
                raise SystemExit(
                    f"row {row_number}: column {column!r} must not be empty/NaN/Inf"
                )


def assert_numeric_values(rows: list[dict[str, str]], columns: list[str]) -> None:
    for row_number, row in enumerate(rows, start=2):
        for column in columns:
            value = row.get(column, "")
            if value == "" or not finite_number(value):
                raise SystemExit(
                    f"row {row_number}: value column {column!r} must be finite numeric"
                )


def sort_warnings(
    rows: list[dict[str, str]],
    id_column: str,
    sort_column: str | None,
    kind_column: str | None = None,
) -> list[str]:
    if not sort_column:
        return ["column_sort not provided; tsfresh will trust current row order"]

    seen: dict[tuple[str, str], float | str] = {}
    warnings: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        key = (row[id_column], row[kind_column] if kind_column else "")
        current = sortable_value(row[sort_column])
        previous = seen.get(key)
        comparable = previous is None or type(current) is type(previous)
        if previous is not None and not comparable:
            warnings.append(
                f"row {row_number}: mixed sort value types for id={key[0]!r}"
                + (f", kind={key[1]!r}" if kind_column else "")
            )
        elif previous is not None and current < previous:
            warnings.append(
                f"row {row_number}: sort column decreases for id={key[0]!r}"
                + (f", kind={key[1]!r}" if kind_column else "")
            )
            if len(warnings) >= 10:
                warnings.append("additional sort warnings omitted")
                break
        seen[key] = current
    return warnings


def validate_flat(args: argparse.Namespace) -> int:
    rows = parse_csv(Path(args.csv), args.delimiter)
    if not rows:
        raise SystemExit("CSV has no data rows")

    fieldnames = set(rows[0].keys())
    required = [args.id_column]
    if args.sort_column:
        required.append(args.sort_column)
    require_columns(fieldnames, required)

    if args.value_columns:
        value_columns = args.value_columns
        require_columns(fieldnames, value_columns)
    else:
        excluded = set(required)
        value_columns = [column for column in rows[0].keys() if column not in excluded]
        if not value_columns:
            raise SystemExit("no value columns found; pass --value-columns")

    assert_clean_special_columns(rows, required)
    assert_numeric_values(rows, value_columns)

    ids = {row[args.id_column] for row in rows}
    warnings = sort_warnings(rows, args.id_column, args.sort_column)
    print(
        "OK flat: "
        f"rows={len(rows)} ids={len(ids)} value_columns={len(value_columns)}"
    )
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


def validate_stacked(args: argparse.Namespace) -> int:
    rows = parse_csv(Path(args.csv), args.delimiter)
    if not rows:
        raise SystemExit("CSV has no data rows")

    required = [args.id_column, args.kind_column, args.value_column]
    if args.sort_column:
        required.append(args.sort_column)
    require_columns(set(rows[0].keys()), required)
    assert_clean_special_columns(rows, required)
    assert_numeric_values(rows, [args.value_column])

    ids = {row[args.id_column] for row in rows}
    kinds = {row[args.kind_column] for row in rows}
    warnings = sort_warnings(
        rows,
        args.id_column,
        args.sort_column,
        kind_column=args.kind_column,
    )
    print(f"OK stacked: rows={len(rows)} ids={len(ids)} kinds={len(kinds)}")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate CSV data before tsfresh feature extraction."
    )
    subparsers = parser.add_subparsers(dest="format", required=True)

    flat = subparsers.add_parser("flat", help="Validate flat/wide CSV input.")
    flat.add_argument("csv")
    flat.add_argument("--id-column", default="id")
    flat.add_argument("--sort-column", default="time")
    flat.add_argument("--value-columns", nargs="*")
    flat.add_argument("--delimiter", default=",")
    flat.set_defaults(func=validate_flat)

    stacked = subparsers.add_parser("stacked", help="Validate stacked/long CSV input.")
    stacked.add_argument("csv")
    stacked.add_argument("--id-column", default="id")
    stacked.add_argument("--sort-column", default="time")
    stacked.add_argument("--kind-column", default="kind")
    stacked.add_argument("--value-column", default="value")
    stacked.add_argument("--delimiter", default=",")
    stacked.set_defaults(func=validate_stacked)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
