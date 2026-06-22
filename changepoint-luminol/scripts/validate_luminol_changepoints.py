#!/usr/bin/env python3
"""Validate CSV inputs and common Luminol detector settings.

This script uses only the standard library. It checks two-column time-series
data, timestamp ordering, numeric values, baseline alignment, labels, and
algorithm-specific parameters before using Luminol anomaly windows as
changepoint candidates.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path


MISSING = {"", "na", "nan", "null", "none"}
SUPPORTED_TIME_FORMATS = [
    "%Y%m%d_%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y%m%d %H:%M:%S",
    "%Y-%m-%d_%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y%m%dT%H:%M:%S",
    "%Y-%m-%d_%H:%M:%S.%f",
    "%Y%m%d_%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y%m%dT%H:%M:%S.%f",
    "%H:%M:%S",
    "%Y%m%d %H:%M:%S.%f",
]
BASELINE_ALGORITHMS = {"diff_percent_threshold", "sign_test"}
ALGORITHMS = {
    "bitmap_detector",
    "default_detector",
    "derivative_detector",
    "exp_avg_detector",
    "absolute_threshold",
    "diff_percent_threshold",
    "sign_test",
}


def parse_time(raw: str) -> float:
    value = raw.strip()
    if value.lower() in MISSING:
        raise SystemExit("missing timestamp")
    try:
        return float(value)
    except ValueError:
        pass
    for fmt in SUPPORTED_TIME_FORMATS:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.timestamp() * 1000.0
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000.0
    except ValueError as exc:
        raise SystemExit(f"cannot parse timestamp {raw!r}") from exc


def finite_number(raw: str, label: str) -> float:
    value = raw.strip()
    if value.lower() in MISSING:
        raise SystemExit(f"{label}: missing numeric value")
    try:
        number = float(value)
    except ValueError as exc:
        raise SystemExit(f"{label}: value must be numeric") from exc
    if not math.isfinite(number):
        raise SystemExit(f"{label}: value must be finite")
    return number


def binary_value(raw: str) -> bool:
    return raw.strip().lower() in {"0", "1", "false", "true"}


def read_rows(args: argparse.Namespace) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(args.csv)
    with path.open(newline="", encoding="utf-8") as handle:
        if args.no_header:
            reader = csv.reader(handle, delimiter=args.delimiter)
            rows = []
            for row in reader:
                if len(row) < 2:
                    raise SystemExit("no-header CSV rows must have at least timestamp,value columns")
                record = {args.time_column: row[0], args.value_column: row[1]}
                if args.baseline_column and len(row) >= 3:
                    record[args.baseline_column] = row[2]
                if args.label_column and len(row) >= 4:
                    record[args.label_column] = row[3]
                rows.append(record)
            return [args.time_column, args.value_column], rows
        reader = csv.DictReader(handle, delimiter=args.delimiter)
        if not reader.fieldnames:
            raise SystemExit("CSV has no header row")
        return reader.fieldnames, list(reader)


def require_columns(fieldnames: list[str], columns: list[str]) -> None:
    missing = [column for column in columns if column not in fieldnames]
    if missing:
        raise SystemExit(f"missing required columns: {', '.join(missing)}")


def validate_series(args: argparse.Namespace, fieldnames: list[str], rows: list[dict[str, str]]) -> list[str]:
    warnings: list[str] = []
    if not rows:
        raise SystemExit("CSV has no data rows")
    required = [args.time_column, args.value_column]
    if args.baseline_column:
        required.append(args.baseline_column)
    if args.label_column:
        required.append(args.label_column)
    require_columns(fieldnames, required)

    timestamps = []
    for row_number, row in enumerate(rows, start=2):
        timestamps.append(parse_time(row[args.time_column]))
        finite_number(row[args.value_column], f"row {row_number} value")
        if args.baseline_column:
            finite_number(row[args.baseline_column], f"row {row_number} baseline")
        if args.label_column and not binary_value(row[args.label_column]):
            raise SystemExit(f"row {row_number}: labels must be binary 0/1 or True/False")

    if len(timestamps) != len(set(timestamps)):
        raise SystemExit("timestamps contain duplicates; Luminol dict/TimeSeries inputs collapse duplicate keys")
    if any(curr < prev for prev, curr in zip(timestamps, timestamps[1:])):
        raise SystemExit("timestamps must be sorted before interpreting Luminol anomaly windows")
    if len(rows) < 3:
        raise SystemExit("Luminol workflows need at least a few points; provide at least 3 rows")
    if len(rows) < 50 and args.algorithm_name == "bitmap_detector":
        warnings.append("bitmap_detector may fall back because default lag/future windows require enough points")
    return warnings


def validate_params(args: argparse.Namespace, n_rows: int) -> list[str]:
    warnings: list[str] = []
    if args.algorithm_name not in ALGORITHMS:
        raise SystemExit(f"unsupported algorithm_name {args.algorithm_name!r}")
    if args.algorithm_name in BASELINE_ALGORITHMS and not args.baseline_column:
        raise SystemExit(f"{args.algorithm_name} requires --baseline-column")
    if args.score_threshold is not None and args.score_threshold < 0:
        raise SystemExit("--score-threshold must be non-negative")
    if args.score_percent_threshold is not None and not 0 < args.score_percent_threshold <= 1:
        raise SystemExit("--score-percent-threshold must be in (0, 1]")
    if args.smoothing_factor is not None and not 0 < args.smoothing_factor <= 1:
        raise SystemExit("--smoothing-factor must be in (0, 1]")
    if args.lag_window_size is not None and args.lag_window_size <= 0:
        raise SystemExit("--lag-window-size must be positive")
    if args.future_window_size is not None and args.future_window_size <= 0:
        raise SystemExit("--future-window-size must be positive")
    if args.chunk_size is not None and args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")
    if args.precision is not None and args.precision <= 0:
        raise SystemExit("--precision must be positive")
    if args.scan_window is not None and args.scan_window <= 0:
        raise SystemExit("--scan-window must be positive")
    if args.scan_window is not None and args.scan_window > n_rows:
        raise SystemExit("--scan-window cannot exceed row count")
    if args.confidence is not None and not 0 < args.confidence < 1:
        raise SystemExit("--confidence must be in (0, 1)")

    if args.algorithm_name == "absolute_threshold":
        if args.absolute_threshold_value_upper is None and args.absolute_threshold_value_lower is None:
            raise SystemExit("absolute_threshold requires an upper or lower absolute threshold")
    if args.algorithm_name == "diff_percent_threshold":
        if args.percent_threshold_upper is None and args.percent_threshold_lower is None:
            raise SystemExit("diff_percent_threshold requires an upper or lower percent threshold")
    if args.algorithm_name == "sign_test":
        upper = args.percent_threshold_upper is not None
        lower = args.percent_threshold_lower is not None
        if upper == lower:
            raise SystemExit("sign_test requires exactly one of upper/lower percent threshold")
        if args.scan_window is None:
            raise SystemExit("sign_test requires --scan-window")
        if lower and args.percent_threshold_lower >= 0:
            warnings.append("sign_test lower threshold should be negative for drops below baseline")
    if args.algorithm_name == "bitmap_detector" and args.lag_window_size and args.future_window_size:
        if args.lag_window_size + args.future_window_size > n_rows:
            raise SystemExit("bitmap lag+future window sizes cannot exceed row count")
        if args.lag_window_size + args.future_window_size < 50:
            warnings.append("source default minimal bitmap window total is 50 points")
    if args.algorithm_name == "bitmap_detector":
        warnings.append("bitmap_detector compares lag and future windows; mark online use as retrospective")
    return warnings


def main() -> int:
    args = build_parser().parse_args()
    fieldnames, rows = read_rows(args)
    warnings = validate_series(args, fieldnames, rows)
    warnings.extend(validate_params(args, len(rows)))
    label_count = 0
    if args.label_column:
        label_count = sum(1 for row in rows if row[args.label_column].strip().lower() in {"1", "true"})
    print(
        "OK luminol changepoint-candidate input: "
        f"rows={len(rows)} algorithm={args.algorithm_name} "
        f"time_col={args.time_column} value_col={args.value_column} "
        f"baseline={'yes' if args.baseline_column else 'no'} labels={label_count}"
    )
    for warning in warnings[:20]:
        print(f"WARNING: {warning}", file=sys.stderr)
    if len(warnings) > 20:
        print("WARNING: additional warnings omitted", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate data before using Luminol anomaly windows as changepoint candidates."
    )
    parser.add_argument("csv")
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--value-column", default="value")
    parser.add_argument("--baseline-column")
    parser.add_argument("--label-column")
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--algorithm-name", default="bitmap_detector")
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--score-percent-threshold", type=float)
    parser.add_argument("--smoothing-factor", type=float)
    parser.add_argument("--lag-window-size", type=int)
    parser.add_argument("--future-window-size", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--precision", type=int)
    parser.add_argument("--absolute-threshold-value-upper", type=float)
    parser.add_argument("--absolute-threshold-value-lower", type=float)
    parser.add_argument("--percent-threshold-upper", type=float)
    parser.add_argument("--percent-threshold-lower", type=float)
    parser.add_argument("--scan-window", type=int)
    parser.add_argument("--confidence", type=float)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
