#!/usr/bin/env python3
"""Validate CSV inputs and BOCPD settings for Merlion changepoint workflows.

The script uses only the standard library. It checks a TimeSeries-style CSV
contract, finite numeric values, timestamp order, optional binary labels, and
common BOCPD parameter feasibility before the data is handed to Merlion.
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


def validate_values(
    rows: list[dict[str, str]],
    value_columns: list[str],
    allow_missing: bool,
) -> None:
    for row_number, row in enumerate(rows, start=2):
        for column in value_columns:
            raw = row[column].strip()
            if allow_missing and raw.lower() in MISSING:
                continue
            if not finite_number(raw):
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
        try:
            value = float(raw)
        except ValueError as exc:
            raise SystemExit(f"row {row_number}: label must be 0 or 1") from exc
        if value not in (0.0, 1.0):
            raise SystemExit(f"row {row_number}: label must be 0 or 1")
        positives += int(value == 1.0)
    return positives


def validate_time_order(times: list[datetime]) -> list[str]:
    warnings: list[str] = []
    if len(set(times)) != len(times):
        raise SystemExit("time column contains duplicate timestamps")
    if any(curr < prev for prev, curr in zip(times, times[1:])):
        warnings.append(
            "time column is not sorted; Merlion can sort some pandas indexes, "
            "but prepared changepoint data should be ordered"
        )
    if len(times) >= 3:
        ordered = sorted(times)
        diffs = [curr - prev for prev, curr in zip(ordered, ordered[1:])]
        if len(set(diffs)) > 1:
            warnings.append(
                "sampling intervals are not constant; BOCPD supports this, "
                "but detection delay and alignment policy must be documented"
            )
    return warnings


def validate_bocpd_args(args: argparse.Namespace, n_rows: int, n_values: int) -> list[str]:
    warnings: list[str] = []
    if args.change_kind not in {"Auto", "LevelShift", "TrendChange"}:
        raise SystemExit("--change-kind must be Auto, LevelShift, or TrendChange")
    if not 0 < args.cp_prior < 1:
        raise SystemExit("--cp-prior must be in (0, 1)")
    if args.lag is not None:
        if args.lag <= 0:
            raise SystemExit("--lag must be positive; Merlion source does not recommend lag=0")
        if args.lag >= n_rows:
            raise SystemExit("--lag must be smaller than row count")
    if args.min_likelihood is not None and args.min_likelihood <= 0:
        raise SystemExit("--min-likelihood must be positive")
    if args.target_seq_index is not None:
        if args.target_seq_index < 0 or args.target_seq_index >= n_values:
            raise SystemExit("--target-seq-index must reference a value column")
    elif n_values > 1:
        warnings.append(
            "multivariate BOCPD is supported, but forecast() will default "
            "target_seq_index to 0 if not specified"
        )
    if args.max_delay_sec is not None and args.max_delay_sec < 0:
        raise SystemExit("--max-delay-sec must be non-negative")
    if args.max_early_sec is not None and args.max_early_sec < 0:
        raise SystemExit("--max-early-sec must be non-negative")
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

    warnings = validate_time_order(times)
    validate_values(rows, value_columns, args.allow_missing)
    positives = validate_labels(rows, args.label_column)
    warnings.extend(validate_bocpd_args(args, len(rows), len(value_columns)))

    print(
        "OK merlion changepoint input: "
        f"rows={len(rows)} dimensions={len(value_columns)} "
        f"labels={positives} change_kind={args.change_kind} "
        f"cp_prior={args.cp_prior}"
    )
    for warning in warnings[:20]:
        print(f"WARNING: {warning}", file=sys.stderr)
    if len(warnings) > 20:
        print("WARNING: additional warnings omitted", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate CSV input before Merlion BOCPD changepoint detection."
    )
    parser.add_argument("csv")
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--value-columns", nargs="*")
    parser.add_argument("--label-column")
    parser.add_argument("--change-kind", default="Auto")
    parser.add_argument("--cp-prior", type=float, default=1e-2)
    parser.add_argument("--lag", type=int)
    parser.add_argument("--min-likelihood", type=float, default=1e-16)
    parser.add_argument("--target-seq-index", type=int)
    parser.add_argument("--max-early-sec", type=float)
    parser.add_argument("--max-delay-sec", type=float)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--delimiter", default=",")
    return parser


def main() -> int:
    return validate_csv(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
