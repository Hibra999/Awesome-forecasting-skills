#!/usr/bin/env python3
"""Validate CSV input shape for Luminaire anomaly detection."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path


BATCH_FREQS = {"H", "D", "W", "W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT"}
STREAMING_FREQS = {"S", "T", "15T", "H", "D", "custom"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--time-column", required=True)
    parser.add_argument("--value-column", default="raw")
    parser.add_argument("--label-column")
    parser.add_argument("--entity-column")
    parser.add_argument("--freq", required=True)
    parser.add_argument("--mode", choices=["batch", "streaming"], default="batch")
    parser.add_argument("--window-length", type=int)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--log-transform", action="store_true")
    return parser.parse_args()


def parse_time(raw: str) -> datetime:
    raw = raw.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    return datetime.fromisoformat(raw)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        return reader.fieldnames, list(reader)


def validate_columns(fieldnames: list[str], rows: list[dict[str, str]], args: argparse.Namespace) -> list[str]:
    if not rows:
        return ["CSV has no data rows"]
    required = {args.time_column, args.value_column}
    if args.label_column:
        required.add(args.label_column)
    if args.entity_column:
        required.add(args.entity_column)
    missing = sorted(required - set(fieldnames))
    return [f"missing columns: {', '.join(missing)}"] if missing else []


def validate_frequency(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if args.mode == "batch" and args.freq not in BATCH_FREQS:
        errors.append(f"batch mode freq {args.freq!r} is not documented by Luminaire batch models")
    if args.mode == "streaming" and args.freq not in STREAMING_FREQS:
        warnings.append("streaming defaults are documented for S, T, 15T, H, D; use manual configuration for this freq")
        if args.window_length is None:
            errors.append("--window-length is required for streaming frequencies outside documented defaults")
    if args.mode == "streaming" and args.freq == "custom" and args.window_length is None:
        errors.append("--window-length is required for streaming custom frequency")
    if args.window_length is not None:
        if args.window_length <= 0:
            errors.append("--window-length must be positive")
        elif args.window_length < 2:
            warnings.append("--window-length is very small for density comparison")
    return errors, warnings


def validate_values(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[str], list[str], int]:
    errors: list[str] = []
    warnings: list[str] = []
    usable = 0
    negatives = 0
    values = set()
    for row_number, row in enumerate(rows, start=2):
        raw = row.get(args.value_column, "").strip()
        if raw == "":
            if args.allow_missing:
                warnings.append("missing values present; Luminaire imputation must be fitted on train only")
                continue
            errors.append(f"row {row_number}: value is missing")
            continue
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"row {row_number}: value is not numeric")
            continue
        if not math.isfinite(value):
            errors.append(f"row {row_number}: value is not finite")
        if value < 0:
            negatives += 1
        values.add(value)
        usable += 1
    if len(values) <= 1:
        warnings.append("value column is constant or nearly empty")
    if args.log_transform and negatives:
        warnings.append("negative values present; Luminaire docs say log transform is ignored with negatives")
    return errors, warnings, usable


def validate_labels(rows: list[dict[str, str]], label_column: str | None) -> tuple[list[str], int | None]:
    if not label_column:
        return [], None
    errors: list[str] = []
    positives = 0
    allowed = {"0", "1", "False", "True", "false", "true"}
    for row_number, row in enumerate(rows, start=2):
        raw = row.get(label_column, "").strip()
        if raw not in allowed:
            errors.append(f"row {row_number}: label {raw!r} is not binary")
        elif raw in {"1", "True", "true"}:
            positives += 1
    return errors, positives


def validate_time(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    grouped: dict[str, list[datetime]] = {}
    for row_number, row in enumerate(rows, start=2):
        entity = row.get(args.entity_column, "__all__") if args.entity_column else "__all__"
        try:
            grouped.setdefault(entity, []).append(parse_time(row.get(args.time_column, "")))
        except ValueError:
            errors.append(f"row {row_number}: time value is not ISO-8601 parseable")
    if errors:
        return errors, warnings

    for entity, times in grouped.items():
        if len(times) != len(set(times)):
            warnings.append(f"entity {entity!r}: duplicate timestamps; Luminaire averages duplicates, but explicit deduping is safer")
        if any(curr <= prev for prev, curr in zip(times, times[1:])):
            errors.append(f"entity {entity!r}: timestamps must be strictly increasing")
        if len(times) >= 3:
            deltas = [curr - prev for prev, curr in zip(times, times[1:])]
            if len(set(deltas)) > 1:
                warnings.append(f"entity {entity!r}: irregular sampling; Luminaire may impute missing indexes during profiling")
    if args.entity_column:
        warnings.append("entity column provided; run one Luminaire model per entity after chronological splitting")
    return errors, warnings


def validate_lengths(row_count: int, usable: int, args: argparse.Namespace) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if usable < 10:
        warnings.append("very short series; Luminaire training/profiling may fail or over-alert")
    if args.mode == "streaming" and args.window_length is not None:
        if row_count <= args.window_length:
            errors.append("CSV has too few rows for the selected streaming window length")
        elif row_count < 3 * args.window_length:
            warnings.append("few windows relative to --window-length; validation may be unstable")
    return errors, warnings


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    positives: int | None = None
    usable = 0

    if not args.csv_path.exists():
        errors.append(f"CSV not found: {args.csv_path}")
        fieldnames: list[str] = []
        rows: list[dict[str, str]] = []
    else:
        try:
            fieldnames, rows = read_rows(args.csv_path)
        except Exception as exc:  # noqa: BLE001 - CLI should report parser failures.
            errors.append(str(exc))
            fieldnames, rows = [], []

    if rows:
        errors.extend(validate_columns(fieldnames, rows, args))
        freq_errors, freq_warnings = validate_frequency(args)
        errors.extend(freq_errors)
        warnings.extend(freq_warnings)
        if not errors:
            value_errors, value_warnings, usable = validate_values(rows, args)
            label_errors, positives = validate_labels(rows, args.label_column)
            time_errors, time_warnings = validate_time(rows, args)
            length_errors, length_warnings = validate_lengths(len(rows), usable, args)
            errors.extend(value_errors + label_errors + time_errors + length_errors)
            warnings.extend(value_warnings + time_warnings + length_warnings)

    for warning in sorted(set(warnings)):
        print(f"WARN: {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    label_text = "no label column"
    if positives is not None:
        label_text = f"{positives} anomaly labels"
    print(
        "OK: "
        f"{len(rows)} rows, usable_values={usable}, mode={args.mode}, "
        f"freq={args.freq}, value={args.value_column}, {label_text}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
