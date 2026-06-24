#!/usr/bin/env python3
"""Validate CSV inputs and detector settings for Alibi Detect change alarms.

The script uses only the standard library. It checks ordered timestamps,
numeric or binary value columns, reference/stream lengths, online window
settings, ERT, p-value, and common leakage-prone configuration mistakes.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path


MISSING = {"", "na", "nan", "null", "none"}
ONLINE = {"mmd-online", "lsdd-online", "cvm-online", "fet-online"}


def parse_time(raw: str, column: str) -> datetime:
    value = raw.strip()
    if value.lower() in MISSING:
        raise SystemExit(f"{column}: missing timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(f"{column}: cannot parse datetime value {raw!r}") from exc


def finite_number(raw: str) -> bool:
    try:
        return math.isfinite(float(raw))
    except (TypeError, ValueError):
        return False


def binary_value(raw: str) -> bool:
    value = raw.strip().lower()
    return value in {"0", "1", "false", "true"}


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
    label_column: str | None,
    requested: list[str] | None,
) -> list[str]:
    fieldnames = list(rows[0].keys())
    if requested:
        require_columns(set(fieldnames), requested)
        return requested
    values = [
        column for column in fieldnames
        if column != time_column and column != label_column
    ]
    if not values:
        raise SystemExit("no value columns found; pass --value-columns")
    return values


def validate_time_order(times: list[datetime]) -> list[str]:
    warnings: list[str] = []
    if len(set(times)) != len(times):
        raise SystemExit("time column contains duplicate timestamps")
    if any(curr < prev for prev, curr in zip(times, times[1:])):
        raise SystemExit("time column must be sorted for online changepoint-style detection")
    if len(times) >= 3:
        diffs = [curr - prev for prev, curr in zip(times, times[1:])]
        if len(set(diffs)) > 1:
            warnings.append("sampling intervals are irregular; keep timestamp mapping explicit")
    return warnings


def validate_values(
    rows: list[dict[str, str]],
    value_columns: list[str],
    detector: str,
    allow_missing: bool,
) -> None:
    for row_number, row in enumerate(rows, start=2):
        for column in value_columns:
            raw = row[column].strip()
            if allow_missing and raw.lower() in MISSING:
                continue
            if detector == "fet-online":
                if not binary_value(raw):
                    raise SystemExit(
                        f"row {row_number}: FETDriftOnline requires binary 0/1 or True/False values"
                    )
            elif not finite_number(raw):
                raise SystemExit(
                    f"row {row_number}: value column {column!r} must be finite numeric"
                )


def validate_labels(rows: list[dict[str, str]], label_column: str | None) -> int:
    if not label_column:
        return 0
    require_columns(set(rows[0].keys()), [label_column])
    positives = 0
    for row_number, row in enumerate(rows, start=2):
        raw = row[label_column].strip()
        if raw.lower() in MISSING:
            raise SystemExit(f"row {row_number}: label column {label_column!r} is missing")
        if not binary_value(raw):
            raise SystemExit(f"row {row_number}: labels must be binary 0/1 or True/False")
        positives += int(raw.lower() in {"1", "true"})
    return positives


def parse_windows(args: argparse.Namespace) -> list[int]:
    windows = args.window_sizes if args.window_sizes else []
    if args.window_size is not None:
        windows.append(args.window_size)
    if not windows:
        windows = [50] if args.detector in {"mmd-online", "lsdd-online"} else [20, 40]
    if any(window <= 0 for window in windows):
        raise SystemExit("window sizes must be positive")
    return sorted(set(windows))


def validate_detector_args(
    args: argparse.Namespace,
    n_rows: int,
    windows: list[int],
) -> list[str]:
    warnings: list[str] = []
    if args.reference_rows <= 0:
        raise SystemExit("--reference-rows must be positive")
    if args.reference_rows >= n_rows:
        raise SystemExit("--reference-rows must leave at least one stream row")
    stream_rows = n_rows - args.reference_rows
    max_window = max(windows)
    if max_window > stream_rows:
        raise SystemExit("largest window size must be <= number of stream rows after reference")
    if args.detector in {"mmd-online", "lsdd-online"} and len(windows) > 1:
        warnings.append("MMD/LSDD online use one window_size; this script will report the largest window")
    if args.ert <= 1:
        raise SystemExit("--ert must be greater than 1")
    if args.n_bootstraps <= 0:
        raise SystemExit("--n-bootstraps must be positive")
    if args.n_bootstraps < 10 * args.ert:
        warnings.append("official docs recommend n_bootstraps at least an order of magnitude larger than ERT")
    if not 0 < args.p_val < 1:
        raise SystemExit("--p-val must be in (0, 1)")
    if args.backend == "keops" and args.detector != "offline-window":
        warnings.append("KeOps is documented for MMDDrift; online MMD supports tensorflow or pytorch backends")
    if args.detector == "fet-online":
        if args.alternative not in {"greater", "less"}:
            raise SystemExit("--alternative must be greater or less")
        if not 0 < args.lam <= 1:
            raise SystemExit("--lam must be in (0, 1]")
    if args.reference_rows < 2 * max_window:
        warnings.append("reference_rows is small relative to window size; drift tests may be unstable")
    return warnings


def validate_csv(args: argparse.Namespace) -> int:
    rows = parse_csv(Path(args.csv), args.delimiter)
    if not rows:
        raise SystemExit("CSV has no data rows")

    require_columns(set(rows[0].keys()), [args.time_column])
    value_columns = infer_value_columns(
        rows=rows,
        time_column=args.time_column,
        label_column=args.label_column,
        requested=args.value_columns,
    )
    times = [parse_time(row[args.time_column], args.time_column) for row in rows]
    windows = parse_windows(args)

    warnings = validate_time_order(times)
    validate_values(rows, value_columns, args.detector, args.allow_missing)
    positives = validate_labels(rows, args.label_column)
    warnings.extend(validate_detector_args(args, len(rows), windows))

    reported_window = max(windows) if args.detector in {"mmd-online", "lsdd-online"} else windows
    print(
        "OK alibi-detect change-alarm input: "
        f"rows={len(rows)} reference_rows={args.reference_rows} "
        f"stream_rows={len(rows) - args.reference_rows} dimensions={len(value_columns)} "
        f"labels={positives} detector={args.detector} windows={reported_window} "
        f"ert={args.ert}"
    )
    for warning in warnings[:20]:
        print(f"WARNING: {warning}", file=sys.stderr)
    if len(warnings) > 20:
        print("WARNING: additional warnings omitted", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate CSV data before Alibi Detect online drift changepoint-style alarms."
    )
    parser.add_argument("csv")
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--value-columns", nargs="*")
    parser.add_argument("--label-column")
    parser.add_argument(
        "--detector",
        default="mmd-online",
        choices=["mmd-online", "lsdd-online", "cvm-online", "fet-online", "offline-window"],
    )
    parser.add_argument("--reference-rows", type=int, default=50)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--window-sizes", nargs="*", type=int)
    parser.add_argument("--ert", type=float, default=150)
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--backend", choices=["tensorflow", "pytorch", "keops"], default="pytorch")
    parser.add_argument("--p-val", type=float, default=0.05)
    parser.add_argument("--alternative", choices=["greater", "less"], default="greater")
    parser.add_argument("--lam", type=float, default=0.99)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--delimiter", default=",")
    return parser


def main() -> int:
    return validate_csv(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
