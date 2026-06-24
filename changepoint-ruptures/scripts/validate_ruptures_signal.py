#!/usr/bin/env python3
"""Validate CSV signals and breakpoint lists before using ruptures.

This script uses only the standard library. It checks sorted order, finite
numeric signal values, basic min_size/jump/window feasibility, and the ruptures
breakpoint convention where the final breakpoint equals n_samples.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
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
        raise SystemExit("no signal value columns found; pass --value-columns")
    return values


def order_warnings(rows: list[dict[str, str]], time_column: str | None) -> list[str]:
    if not time_column:
        return ["no time/order column provided; ensure CSV row order is already sorted"]

    parsed: list[float | str] = []
    seen: set[str] = set()
    warnings: list[str] = []
    for row in rows:
        raw = row[time_column]
        if invalid_token(raw):
            raise SystemExit("time/order column has empty/NaN/Inf value")
        if raw in seen:
            warnings.append(f"duplicate time/order value {raw!r}")
            break
        seen.add(raw)
        parsed.append(parse_time(raw))

    comparable = all(isinstance(value, (float, int)) for value in parsed)
    if comparable:
        numeric = [float(value) for value in parsed]
        if any(curr < prev for prev, curr in zip(numeric, numeric[1:])):
            warnings.append("time/order column is not sorted")
        if len(numeric) >= 3:
            diffs = [round(curr - prev, 12) for prev, curr in zip(numeric, numeric[1:])]
            if len(set(diffs)) > 1:
                warnings.append("sampling intervals are not constant; document the sampling policy")
    else:
        if any(str(curr) < str(prev) for prev, curr in zip(parsed, parsed[1:])):
            warnings.append("time/order column is not lexicographically sorted")
    return warnings


def parse_breakpoints(raw: str | None) -> list[int]:
    if not raw:
        return []
    try:
        bkps = [int(piece.strip()) for piece in raw.split(",") if piece.strip()]
    except ValueError as exc:
        raise SystemExit("--breakpoints must be comma-separated integers") from exc
    if not bkps:
        return []
    if any(point <= 0 for point in bkps):
        raise SystemExit("breakpoints must be positive sample indexes")
    if any(curr <= prev for prev, curr in zip(bkps, bkps[1:])):
        raise SystemExit("breakpoints must be strictly increasing")
    return bkps


def validate_breakpoints(bkps: list[int], n_samples: int, min_size: int) -> list[str]:
    if not bkps:
        return []
    warnings: list[str] = []
    if bkps[-1] != n_samples:
        raise SystemExit(
            f"last breakpoint must equal n_samples ({n_samples}) in ruptures convention"
        )
    previous = 0
    for point in bkps:
        if point - previous < min_size:
            warnings.append(
                f"segment [{previous}:{point}] is shorter than min_size {min_size}"
            )
        previous = point
    return warnings


def validate_args(n_samples: int, min_size: int, jump: int, width: int | None) -> None:
    if min_size <= 0:
        raise SystemExit("--min-size must be positive")
    if jump <= 0:
        raise SystemExit("--jump must be positive")
    if n_samples < min_size:
        raise SystemExit(f"n_samples {n_samples} is shorter than min_size {min_size}")
    if width is not None:
        if width <= 0:
            raise SystemExit("--width must be positive")
        if width >= n_samples:
            raise SystemExit("--width must be smaller than n_samples")


def validate_csv(args: argparse.Namespace) -> int:
    rows = parse_csv(Path(args.csv), args.delimiter)
    if not rows:
        raise SystemExit("CSV has no data rows")

    required = [args.time_column] if args.time_column else []
    require_columns(set(rows[0].keys()), required)
    value_columns = infer_value_columns(rows, set(required), args.value_columns)

    for row_number, row in enumerate(rows, start=2):
        for column in value_columns:
            if not finite_number(row[column]):
                raise SystemExit(
                    f"row {row_number}: signal column {column!r} must be finite numeric"
                )

    validate_args(len(rows), args.min_size, args.jump, args.width)
    warnings = order_warnings(rows, args.time_column)
    warnings.extend(
        validate_breakpoints(parse_breakpoints(args.breakpoints), len(rows), args.min_size)
    )

    print(f"OK signal: rows={len(rows)} dimensions={len(value_columns)}")
    for warning in warnings[:20]:
        print(f"WARNING: {warning}", file=sys.stderr)
    if len(warnings) > 20:
        print("WARNING: additional warnings omitted", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a CSV signal before ruptures change point detection."
    )
    parser.add_argument("csv")
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--value-columns", nargs="*")
    parser.add_argument("--min-size", type=int, default=2)
    parser.add_argument("--jump", type=int, default=1)
    parser.add_argument("--width", type=int)
    parser.add_argument("--breakpoints", default=None)
    parser.add_argument("--delimiter", default=",")
    return parser


def main() -> int:
    return validate_csv(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
