#!/usr/bin/env python3
"""Validate CSV shape and detector settings for ADTK changepoint workflows."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path


DETECTORS = {
    "level-shift",
    "persist",
    "volatility-shift",
    "seasonal",
    "autoregression",
    "threshold",
    "quantile",
    "iqr",
    "esd",
    "multivariate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--value-columns", nargs="+", default=["value"])
    parser.add_argument("--label-column")
    parser.add_argument("--detector", choices=sorted(DETECTORS), default="level-shift")
    parser.add_argument("--window", type=int)
    parser.add_argument("--left-window", type=int)
    parser.add_argument("--right-window", type=int)
    parser.add_argument("--c", type=float, default=6.0)
    parser.add_argument("--side", choices=["both", "positive", "negative"], default="both")
    parser.add_argument("--agg", default="median")
    parser.add_argument("--freq", type=int)
    parser.add_argument("--n-steps", type=int, default=1)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--low", type=float)
    parser.add_argument("--high", type=float)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--n-splits", type=int, default=1)
    return parser.parse_args()


def parse_timestamp(value: str) -> datetime:
    value = value.strip()
    if not value:
        raise ValueError("empty timestamp")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"timestamp {value!r} is not ISO-8601 parseable") from exc


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        return list(reader)


def validate_columns(rows: list[dict[str, str]], args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if not rows:
        return ["CSV has no data rows"]

    columns = set(rows[0].keys())
    required = {args.time_column, *args.value_columns}
    if args.label_column:
        required.add(args.label_column)
    missing = sorted(required - columns)
    if missing:
        errors.append(f"missing columns: {', '.join(missing)}")
    return errors


def validate_times(rows: list[dict[str, str]], time_column: str) -> tuple[list[datetime], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    times: list[datetime] = []

    for i, row in enumerate(rows, start=2):
        try:
            times.append(parse_timestamp(row.get(time_column, "")))
        except ValueError as exc:
            errors.append(f"row {i}: {exc}")

    if errors:
        return times, errors, warnings

    try:
        if any(curr <= prev for prev, curr in zip(times, times[1:])):
            errors.append("timestamps must be strictly increasing; sort and deduplicate before ADTK")
    except TypeError:
        errors.append("timestamps mix timezone-aware and timezone-naive values")
        return times, errors, warnings

    duplicate_count = len(times) - len(set(times))
    if duplicate_count:
        errors.append(f"found {duplicate_count} duplicate timestamps")

    if len(times) >= 3:
        deltas = [curr - prev for prev, curr in zip(times, times[1:])]
        if len(set(deltas)) > 1:
            warnings.append("timestamps are irregular; document gaps before relying on frequency-based events")

    return times, errors, warnings


def validate_values(rows: list[dict[str, str]], value_columns: list[str]) -> list[str]:
    errors: list[str] = []
    for column in value_columns:
        for i, row in enumerate(rows, start=2):
            raw = row.get(column, "")
            try:
                value = float(raw)
            except ValueError:
                errors.append(f"row {i}: column {column!r} is not numeric")
                continue
            if not math.isfinite(value):
                errors.append(f"row {i}: column {column!r} is not finite")
    return errors


def validate_labels(rows: list[dict[str, str]], label_column: str | None) -> tuple[int | None, list[str]]:
    if not label_column:
        return None, []
    errors: list[str] = []
    positives = 0
    for i, row in enumerate(rows, start=2):
        raw = row.get(label_column, "").strip()
        if raw not in {"0", "1", "False", "True", "false", "true"}:
            errors.append(f"row {i}: label {raw!r} is not binary")
        if raw in {"1", "True", "true"}:
            positives += 1
    return positives, errors


def max_window(args: argparse.Namespace) -> int:
    windows = [w for w in (args.window, args.left_window, args.right_window) if w is not None]
    return max(windows) if windows else 0


def validate_detector(rows_count: int, args: argparse.Namespace) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for name in ("window", "left_window", "right_window", "freq", "n_steps", "step_size", "n_splits"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            errors.append(f"--{name.replace('_', '-')} must be positive")

    if not 0 < args.train_ratio < 1:
        errors.append("--train-ratio must be between 0 and 1")
    if args.c <= 0:
        errors.append("--c must be positive")
    if not 0 < args.alpha < 1:
        errors.append("--alpha must be between 0 and 1")

    if args.detector in {"level-shift", "volatility-shift"} and not any(
        [args.window, args.left_window, args.right_window]
    ):
        errors.append(f"--window is required for {args.detector}")

    if args.detector == "persist" and args.window is None:
        warnings.append("PersistAD defaults to window=1; pass --window to make the time scale explicit")

    if args.detector == "threshold" and args.low is None and args.high is None:
        errors.append("ThresholdAD requires at least one of --low or --high")

    if args.detector == "persist" and args.agg not in {"mean", "median"}:
        errors.append("PersistAD --agg must be mean or median")

    if args.detector == "volatility-shift" and args.agg not in {"std", "iqr", "idr"}:
        errors.append("VolatilityShiftAD --agg must be std, iqr, or idr")

    if args.detector == "seasonal" and args.freq is None:
        warnings.append("SeasonalAD can use freq=None, but production workflows should document the period")

    if args.detector == "autoregression":
        needed = args.n_steps * args.step_size + 2
        if rows_count <= needed:
            errors.append(f"autoregression needs more than {needed} rows")

    if args.detector == "multivariate" and len(args.value_columns) < 2:
        errors.append("multivariate detectors require at least two --value-columns")

    train_rows = int(rows_count * args.train_ratio)
    test_rows = rows_count - train_rows
    if train_rows < 2 or test_rows < 1:
        errors.append("train/test split leaves too few rows")

    window = max_window(args)
    if window:
        if train_rows <= 2 * window:
            warnings.append("training split is short relative to the selected window")
        if rows_count <= 2 * window:
            errors.append("series is too short for the selected side-by-side window")

    return errors, warnings


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    if not args.csv_path.exists():
        errors.append(f"CSV not found: {args.csv_path}")
    else:
        try:
            rows = read_rows(args.csv_path)
        except Exception as exc:  # noqa: BLE001 - CLI should report parser failures.
            errors.append(str(exc))
            rows = []

        errors.extend(validate_columns(rows, args))
        if not errors:
            _, time_errors, time_warnings = validate_times(rows, args.time_column)
            errors.extend(time_errors)
            warnings.extend(time_warnings)
            errors.extend(validate_values(rows, args.value_columns))
            positives, label_errors = validate_labels(rows, args.label_column)
            errors.extend(label_errors)
            detector_errors, detector_warnings = validate_detector(len(rows), args)
            errors.extend(detector_errors)
            warnings.extend(detector_warnings)
        else:
            positives = None

    for warning in warnings:
        print(f"WARN: {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    label_text = "no label column"
    if positives is not None:
        label_text = f"{positives} positive labels"
    print(
        "OK: "
        f"{len(rows)} rows, detector={args.detector}, "
        f"time={args.time_column}, values={','.join(args.value_columns)}, {label_text}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
