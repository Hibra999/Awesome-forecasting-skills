#!/usr/bin/env python3
"""Validate CSV inputs and window settings for Kats changepoint workflows.

This script uses only the standard library. It checks the TimeSeriesData-style
time/value contract, finite numeric values, sorted timestamps, optional labeled
changepoint indexes, and common CUSUM/BOCPD/window parameter feasibility.
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
        raise SystemExit(f"{column}: missing time value")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(f"{column}: cannot parse datetime value {raw!r}") from exc


def finite_number(raw: str) -> bool:
    try:
        return math.isfinite(float(raw))
    except (TypeError, ValueError):
        return False


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
    time_column: str,
    requested: list[str] | None,
) -> list[str]:
    fieldnames = list(rows[0].keys())
    if requested:
        require_columns(set(fieldnames), requested)
        return requested
    values = [column for column in fieldnames if column != time_column]
    if not values:
        raise SystemExit("no value columns found; pass --value-columns")
    return values


def validate_values(rows: list[dict[str, str]], value_columns: list[str]) -> None:
    for row_number, row in enumerate(rows, start=2):
        for column in value_columns:
            if not finite_number(row[column]):
                raise SystemExit(
                    f"row {row_number}: value column {column!r} must be finite numeric"
                )


def validate_time_order(times: list[datetime]) -> list[str]:
    warnings: list[str] = []
    if len(set(times)) != len(times):
        raise SystemExit("time column contains duplicate timestamps")
    if any(curr < prev for prev, curr in zip(times, times[1:])):
        warnings.append("time column is not sorted; TimeSeriesData can sort, but prep should be ordered")
    if len(times) >= 3:
        ordered = sorted(times)
        diffs = [curr - prev for prev, curr in zip(ordered, ordered[1:])]
        if len(set(diffs)) > 1:
            warnings.append("sampling intervals are not constant; document interpolation/gap policy")
    return warnings


def parse_indexes(raw: str | None, n_rows: int) -> list[int]:
    if not raw:
        return []
    try:
        indexes = [int(piece.strip()) for piece in raw.split(",") if piece.strip()]
    except ValueError as exc:
        raise SystemExit("--label-indexes must be comma-separated integers") from exc
    if any(index < 0 or index >= n_rows for index in indexes):
        raise SystemExit(f"label indexes must be 0-based and within [0, {n_rows - 1}]")
    if any(curr <= prev for prev, curr in zip(indexes, indexes[1:])):
        raise SystemExit("label indexes must be strictly increasing")
    return indexes


def validate_detector_args(args: argparse.Namespace, n_rows: int) -> list[str]:
    warnings: list[str] = []
    if args.threshold <= 0 or args.threshold >= 1:
        raise SystemExit("--threshold must be in (0, 1)")
    if args.lag is not None and args.lag <= 0:
        raise SystemExit("--lag must be positive")
    if args.lag is not None and args.lag >= n_rows:
        raise SystemExit("--lag must be smaller than row count")
    if args.window_size is not None and args.window_size <= 1:
        raise SystemExit("--window-size must be greater than 1")
    if args.window_size is not None and args.window_size >= n_rows:
        raise SystemExit("--window-size must be smaller than row count")
    if args.n_control is not None and args.n_control <= 0:
        raise SystemExit("--n-control must be positive")
    if args.n_test is not None and args.n_test <= 0:
        raise SystemExit("--n-test must be positive")
    if args.n_control is not None and args.n_test is not None:
        if args.n_control + args.n_test >= n_rows:
            raise SystemExit("--n-control + --n-test must be smaller than row count")
        if args.n_control < 30:
            warnings.append("Kats docs suggest n_control >= 30 for StatSigDetectorModel")
    if args.scan_window is not None and args.scan_window <= 0:
        raise SystemExit("--scan-window must be positive seconds")
    if args.historical_window is not None and args.historical_window <= 0:
        raise SystemExit("--historical-window must be positive seconds")
    if args.step_window is not None and args.step_window <= 0:
        raise SystemExit("--step-window must be positive seconds")
    if args.scan_window is not None and args.step_window is not None:
        if args.scan_window <= args.step_window:
            raise SystemExit("CUSUMDetectorModel requires scan_window > step_window")
    if args.interest_window:
        pieces = args.interest_window.split(",")
        if len(pieces) != 2:
            raise SystemExit("--interest-window must be start,end")
        try:
            start, end = [int(piece.strip()) for piece in pieces]
        except ValueError as exc:
            raise SystemExit("--interest-window must contain integers") from exc
        if start < 0 or end <= start or end > n_rows:
            raise SystemExit("--interest-window must satisfy 0 <= start < end <= rows")
    return warnings


def validate_csv(args: argparse.Namespace) -> int:
    rows = parse_csv(Path(args.csv), args.delimiter)
    if not rows:
        raise SystemExit("CSV has no data rows")

    require_columns(set(rows[0].keys()), [args.time_column])
    value_columns = infer_value_columns(rows, args.time_column, args.value_columns)
    times = [parse_time(row[args.time_column], args.time_column) for row in rows]

    warnings = validate_time_order(times)
    validate_values(rows, value_columns)
    labels = parse_indexes(args.label_indexes, len(rows))
    warnings.extend(validate_detector_args(args, len(rows)))

    if args.detector in {"cusum", "robust"} and len(value_columns) != 1:
        warnings.append(f"{args.detector} workflow should use one value column unless using a documented multivariate/vectorized path")
    if args.detector == "bocpd" and len(value_columns) > 1:
        warnings.append("BOCPD model support depends on selected model; verify multivariate support before using multiple value columns")

    print(
        "OK kats input: "
        f"rows={len(rows)} dimensions={len(value_columns)} "
        f"labels={len(labels)} detector={args.detector}"
    )
    for warning in warnings[:20]:
        print(f"WARNING: {warning}", file=sys.stderr)
    if len(warnings) > 20:
        print("WARNING: additional warnings omitted", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate CSV input before Kats changepoint detection."
    )
    parser.add_argument("csv")
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--value-columns", nargs="*")
    parser.add_argument(
        "--detector",
        default="cusum",
        choices=["cusum", "bocpd", "robust", "cusum-model", "mk", "statsig"],
    )
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--lag", type=int)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--n-control", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--scan-window", type=int)
    parser.add_argument("--historical-window", type=int)
    parser.add_argument("--step-window", type=int)
    parser.add_argument("--interest-window")
    parser.add_argument("--label-indexes")
    parser.add_argument("--delimiter", default=",")
    return parser


def main() -> int:
    return validate_csv(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
