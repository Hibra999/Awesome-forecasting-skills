#!/usr/bin/env python3
"""Validate CSV inputs and Greykite changepoint settings.

The script uses only the standard library. It checks timestamp order, numeric
target values, minimum non-null sample size, custom changepoint dates, and the
most common parameter ranges documented by Greykite.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path


MISSING = {"", "na", "nan", "null", "none"}


def parse_time(raw: str, column: str) -> datetime:
    value = raw.strip()
    if value.lower() in MISSING:
        raise SystemExit(f"{column}: missing timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(f"{column}: cannot parse datetime value {raw!r}") from exc


def parse_optional_time(raw: str, name: str) -> datetime:
    try:
        return datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(f"{name}: cannot parse datetime value {raw!r}") from exc


def finite_or_missing(raw: str) -> tuple[bool, bool]:
    value = raw.strip()
    if value.lower() in MISSING:
        return False, True
    try:
        return math.isfinite(float(value)), False
    except ValueError:
        return False, False


def binary_value(raw: str) -> bool:
    return raw.strip().lower() in {"0", "1", "false", "true"}


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


def validate_times(times: list[datetime]) -> list[str]:
    warnings: list[str] = []
    if len(set(times)) != len(times):
        raise SystemExit("time column contains duplicate timestamps")
    if any(curr < prev for prev, curr in zip(times, times[1:])):
        raise SystemExit("time column must be sorted before Greykite changepoint detection")
    if len(times) >= 3:
        diffs = [curr - prev for prev, curr in zip(times, times[1:])]
        if len(set(diffs)) > 1:
            warnings.append("sampling intervals are irregular; pass explicit freq in forecast workflows")
    return warnings


def validate_values(rows: list[dict[str, str]], value_col: str) -> int:
    non_missing = 0
    for row_number, row in enumerate(rows, start=2):
        ok, missing = finite_or_missing(row[value_col])
        if missing:
            continue
        if not ok:
            raise SystemExit(f"row {row_number}: value column {value_col!r} must be finite numeric or missing")
        non_missing += 1
    if non_missing < 5:
        raise SystemExit("Greykite ChangepointDetector requires at least 5 non-null observations")
    return non_missing


def validate_labels(rows: list[dict[str, str]], label_col: str | None) -> int:
    if not label_col:
        return 0
    require_columns(set(rows[0].keys()), [label_col])
    positives = 0
    for row_number, row in enumerate(rows, start=2):
        raw = row[label_col].strip()
        if raw.lower() in MISSING:
            raise SystemExit(f"row {row_number}: label column {label_col!r} is missing")
        if not binary_value(raw):
            raise SystemExit(f"row {row_number}: labels must be binary 0/1 or True/False")
        positives += int(raw.lower() in {"1", "true"})
    return positives


def reject_large_freq_units(freq: str | None, name: str) -> None:
    if freq and any(char in freq for char in ["W", "M", "Y"]):
        raise SystemExit(
            f"{name}: Greykite distance parameters allow units no larger than day-level strings"
        )


def validate_params(args: argparse.Namespace, times: list[datetime]) -> list[str]:
    warnings: list[str] = []
    if args.regularization_strength is not None and not 0 <= args.regularization_strength <= 1:
        raise SystemExit("--regularization-strength must be in [0, 1]")
    if args.potential_changepoint_n < 0:
        raise SystemExit("--potential-changepoint-n cannot be negative")
    if args.yearly_seasonality_order < 0:
        raise SystemExit("--yearly-seasonality-order cannot be negative")
    if not 0 <= args.no_changepoint_proportion_from_begin <= 1:
        raise SystemExit("--no-changepoint-proportion-from-begin must be in [0, 1]")
    if not 0 <= args.no_changepoint_proportion_from_end <= 1:
        raise SystemExit("--no-changepoint-proportion-from-end must be in [0, 1]")
    if args.forecast_horizon < 0:
        raise SystemExit("--forecast-horizon cannot be negative")

    reject_large_freq_units(args.actual_changepoint_min_distance, "--actual-changepoint-min-distance")
    reject_large_freq_units(args.potential_changepoint_distance, "--potential-changepoint-distance")
    reject_large_freq_units(args.no_changepoint_distance_from_begin, "--no-changepoint-distance-from-begin")
    reject_large_freq_units(args.no_changepoint_distance_from_end, "--no-changepoint-distance-from-end")

    min_time, max_time = times[0], times[-1]
    for raw_date in args.custom_changepoint_dates or []:
        dt = parse_optional_time(raw_date, "--custom-changepoint-dates")
        if not min_time < dt < max_time:
            warnings.append(f"custom changepoint {raw_date!r} is outside the open data range")
        elif args.min_points_after_custom_date:
            later_points = sum(time > dt for time in times)
            if later_points < args.min_points_after_custom_date:
                warnings.append(
                    f"custom changepoint {raw_date!r} leaves only {later_points} later rows"
                )
    if args.no_changepoint_proportion_from_end == 0 and not args.no_changepoint_distance_from_end:
        warnings.append("no end buffer is configured; Greykite docs advise avoiding changepoints near data end")
    if args.regularization_strength is None:
        warnings.append("regularization_strength=None triggers internal CV; use only within training/validation data")
    if args.potential_changepoint_distance and args.potential_changepoint_n != 100:
        warnings.append("potential_changepoint_distance overrides potential_changepoint_n in Greykite")
    return warnings


def validate_csv(args: argparse.Namespace) -> int:
    rows = parse_csv(Path(args.csv), args.delimiter)
    if not rows:
        raise SystemExit("CSV has no data rows")

    require_columns(set(rows[0].keys()), [args.time_column, args.value_column])
    times = [parse_time(row[args.time_column], args.time_column) for row in rows]
    warnings = validate_times(times)
    non_missing = validate_values(rows, args.value_column)
    positives = validate_labels(rows, args.label_column)
    warnings.extend(validate_params(args, times))

    print(
        "OK greykite changepoint input: "
        f"rows={len(rows)} non_missing={non_missing} labels={positives} "
        f"time_col={args.time_column} value_col={args.value_column} "
        f"potential_changepoint_n={args.potential_changepoint_n} "
        f"regularization_strength={args.regularization_strength}"
    )
    for warning in warnings[:20]:
        print(f"WARNING: {warning}", file=sys.stderr)
    if len(warnings) > 20:
        print("WARNING: additional warnings omitted", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate CSV data and parameters before Greykite changepoint detection."
    )
    parser.add_argument("csv")
    parser.add_argument("--time-column", default="ts")
    parser.add_argument("--value-column", default="y")
    parser.add_argument("--label-column")
    parser.add_argument("--custom-changepoint-dates", nargs="*")
    parser.add_argument("--regularization-strength", type=float)
    parser.add_argument("--potential-changepoint-n", type=int, default=100)
    parser.add_argument("--yearly-seasonality-order", type=int, default=8)
    parser.add_argument("--resample-freq", default="D")
    parser.add_argument("--actual-changepoint-min-distance", default="30D")
    parser.add_argument("--potential-changepoint-distance")
    parser.add_argument("--no-changepoint-distance-from-begin")
    parser.add_argument("--no-changepoint-distance-from-end")
    parser.add_argument("--no-changepoint-proportion-from-begin", type=float, default=0.0)
    parser.add_argument("--no-changepoint-proportion-from-end", type=float, default=0.0)
    parser.add_argument("--forecast-horizon", type=int, default=0)
    parser.add_argument("--min-points-after-custom-date", type=int, default=0)
    parser.add_argument("--delimiter", default=",")
    return parser


def main() -> int:
    return validate_csv(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
