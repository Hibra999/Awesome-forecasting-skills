#!/usr/bin/env python3
"""Validate CSV inputs for Prophet trend changepoint workflows.

The script uses only the standard library. It checks Prophet's required ds/y
columns, timezone-naive timestamps, finite numeric values where y is present,
manual changepoints within the training range, and common logistic/regressor
requirements.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path


MISSING = {"", "na", "nan", "null", "none"}


def parse_datetime(raw: str, column: str) -> datetime:
    value = raw.strip()
    if value.lower() in MISSING:
        raise SystemExit(f"{column}: missing datetime value")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(f"{column}: cannot parse datetime value {raw!r}") from exc
    if parsed.tzinfo is not None:
        raise SystemExit(f"{column}: timezone-aware datetimes are not supported by Prophet")
    return parsed


def finite_number(raw: str) -> bool:
    try:
        return math.isfinite(float(raw))
    except (TypeError, ValueError):
        return False


def is_missing(raw: str) -> bool:
    return raw.strip().lower() in MISSING


def parse_csv(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise SystemExit(f"{path}: missing header row")
        return list(reader)


def require_columns(fieldnames: set[str], columns: list[str]) -> None:
    missing = [column for column in columns if column and column not in fieldnames]
    if missing:
        raise SystemExit(f"missing required columns: {', '.join(missing)}")


def validate_y(rows: list[dict[str, str]], y_col: str, allow_missing_y: bool) -> int:
    missing_count = 0
    for row_number, row in enumerate(rows, start=2):
        raw = row[y_col]
        if is_missing(raw):
            missing_count += 1
            if not allow_missing_y:
                raise SystemExit(
                    f"row {row_number}: y is missing; pass --allow-missing-y if intentional"
                )
            continue
        if not finite_number(raw):
            raise SystemExit(f"row {row_number}: y must be finite numeric")
    return missing_count


def validate_order(ds_values: list[datetime]) -> list[str]:
    warnings: list[str] = []
    if len(set(ds_values)) != len(ds_values):
        raise SystemExit("ds contains duplicate timestamps")
    if any(curr < prev for prev, curr in zip(ds_values, ds_values[1:])):
        warnings.append("ds is not sorted; Prophet sorts internally, but prep should be ordered")
    if len(ds_values) >= 3:
        ordered = sorted(ds_values)
        diffs = [curr - prev for prev, curr in zip(ordered, ordered[1:])]
        if len(set(diffs)) > 1:
            warnings.append("sampling intervals are not constant; document missing/gap policy")
    return warnings


def validate_changepoints(raw: str | None, min_ds: datetime, max_ds: datetime) -> int:
    if not raw:
        return 0
    points = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not points:
        return 0
    parsed = [parse_datetime(point, "--changepoints") for point in points]
    for point in parsed:
        if point < min_ds or point > max_ds:
            raise SystemExit(
                f"manual changepoint {point.isoformat()} must fall within training ds range"
            )
    if any(curr <= prev for prev, curr in zip(parsed, parsed[1:])):
        raise SystemExit("manual changepoints must be strictly increasing")
    return len(parsed)


def validate_logistic(rows: list[dict[str, str]], cap_col: str, floor_col: str | None) -> None:
    for row_number, row in enumerate(rows, start=2):
        if not finite_number(row[cap_col]):
            raise SystemExit(f"row {row_number}: cap must be finite numeric for logistic growth")
        cap = float(row[cap_col])
        floor = 0.0
        if floor_col:
            if not finite_number(row[floor_col]):
                raise SystemExit(f"row {row_number}: floor must be finite numeric")
            floor = float(row[floor_col])
        if cap <= floor:
            raise SystemExit(f"row {row_number}: cap must be greater than floor")


def validate_regressors(rows: list[dict[str, str]], regressors: list[str]) -> None:
    for regressor in regressors:
        values: list[str] = []
        for row_number, row in enumerate(rows, start=2):
            raw = row[regressor]
            if is_missing(raw) or not finite_number(raw):
                raise SystemExit(
                    f"row {row_number}: regressor {regressor!r} must be finite numeric"
                )
            values.append(raw)
        if len(set(values)) < 2:
            raise SystemExit(
                f"regressor {regressor!r} is constant throughout history; Prophet will error"
            )


def validate_args(args: argparse.Namespace) -> None:
    if args.changepoint_range < 0 or args.changepoint_range > 1:
        raise SystemExit("--changepoint-range must be in [0, 1]")
    if args.n_changepoints < 0:
        raise SystemExit("--n-changepoints must be non-negative")
    if args.growth not in {"linear", "logistic", "flat"}:
        raise SystemExit("--growth must be one of: linear, logistic, flat")


def validate_csv(args: argparse.Namespace) -> int:
    validate_args(args)
    rows = parse_csv(Path(args.csv), args.delimiter)
    if not rows:
        raise SystemExit("CSV has no data rows")

    fieldnames = set(rows[0].keys())
    required = [args.ds_column, args.y_column]
    if args.growth == "logistic":
        required.append(args.cap_column)
    if args.floor_column:
        required.append(args.floor_column)
    required.extend(args.extra_regressors or [])
    require_columns(fieldnames, required)

    ds_values = [parse_datetime(row[args.ds_column], args.ds_column) for row in rows]
    warnings = validate_order(ds_values)
    missing_y = validate_y(rows, args.y_column, args.allow_missing_y)

    if args.growth == "logistic":
        validate_logistic(rows, args.cap_column, args.floor_column)
    validate_regressors(rows, args.extra_regressors or [])

    cp_count = validate_changepoints(args.changepoints, min(ds_values), max(ds_values))
    hist_size = int(math.floor(len(rows) * args.changepoint_range))
    if cp_count == 0 and args.n_changepoints + 1 > hist_size:
        warnings.append(
            "n_changepoints exceeds eligible history; Prophet will reduce it internally"
        )

    print(
        "OK prophet input: "
        f"rows={len(rows)} missing_y={missing_y} "
        f"manual_changepoints={cp_count} growth={args.growth}"
    )
    for warning in warnings[:20]:
        print(f"WARNING: {warning}", file=sys.stderr)
    if len(warnings) > 20:
        print("WARNING: additional warnings omitted", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate CSV data before Prophet trend changepoint analysis."
    )
    parser.add_argument("csv")
    parser.add_argument("--ds-column", default="ds")
    parser.add_argument("--y-column", default="y")
    parser.add_argument("--allow-missing-y", action="store_true")
    parser.add_argument("--growth", default="linear", choices=["linear", "logistic", "flat"])
    parser.add_argument("--cap-column", default="cap")
    parser.add_argument("--floor-column")
    parser.add_argument("--extra-regressors", nargs="*", default=[])
    parser.add_argument("--changepoints")
    parser.add_argument("--n-changepoints", type=int, default=25)
    parser.add_argument("--changepoint-range", type=float, default=0.8)
    parser.add_argument("--delimiter", default=",")
    return parser


def main() -> int:
    return validate_csv(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
