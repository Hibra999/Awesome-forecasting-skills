#!/usr/bin/env python3
"""Validate local time-series inputs for STUMPY workflows without importing numpy."""

from __future__ import annotations

import argparse
import ast
import csv
import math
import struct
from pathlib import Path


class ValidationError(Exception):
    """Raised when an input does not satisfy the STUMPY workflow contract."""


def read_npy_shape(path: Path) -> tuple[int, ...]:
    with path.open("rb") as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            raise ValidationError(f"{path} is not a .npy file")

        version = handle.read(2)
        if len(version) != 2:
            raise ValidationError(f"{path} has an incomplete .npy header")

        major = version[0]
        if major == 1:
            header_len = struct.unpack("<H", handle.read(2))[0]
            encoding = "latin1"
        elif major in (2, 3):
            header_len = struct.unpack("<I", handle.read(4))[0]
            encoding = "latin1" if major == 2 else "utf-8"
        else:
            raise ValidationError(f"{path} uses unsupported .npy version {major}")

        header = handle.read(header_len).decode(encoding).strip()

    try:
        metadata = ast.literal_eval(header)
    except (SyntaxError, ValueError) as exc:
        raise ValidationError(f"{path} has an invalid .npy header") from exc

    shape = metadata.get("shape")
    if not isinstance(shape, tuple):
        raise ValidationError(f"{path} header does not contain a tuple shape")
    return tuple(int(dim) for dim in shape)


def count_csv_series(path: Path, column: int, delimiter: str, has_header: bool) -> int:
    count = 0
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        if has_header:
            next(reader, None)
        for line_no, row in enumerate(reader, start=2 if has_header else 1):
            if not row or not any(cell.strip() for cell in row):
                continue
            if column >= len(row):
                raise ValidationError(f"{path}:{line_no} has no column {column}")
            raw = row[column].strip()
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValidationError(f"{path}:{line_no} value is not numeric: {raw!r}") from exc
            if not math.isfinite(value):
                raise ValidationError(f"{path}:{line_no} value is not finite: {raw!r}")
            count += 1
    if count == 0:
        raise ValidationError(f"{path} contains no numeric samples")
    return count


def series_length(path: Path, column: int, delimiter: str, has_header: bool) -> int:
    if path.suffix == ".npy":
        shape = read_npy_shape(path)
        if len(shape) != 1:
            raise ValidationError(f"{path} must be a 1D .npy array; got shape {shape}")
        return shape[0]
    return count_csv_series(path, column, delimiter, has_header)


def multidim_shape(path: Path) -> tuple[int, int]:
    if path.suffix != ".npy":
        raise ValidationError("multidimensional validation currently supports .npy only")
    shape = read_npy_shape(path)
    if len(shape) != 2:
        raise ValidationError(f"{path} must be a 2D .npy array; got shape {shape}")
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValidationError(f"{path} has invalid non-positive shape {shape}")
    return shape


def validate_match(args: argparse.Namespace) -> str:
    q_len = series_length(args.query, args.query_column, args.delimiter, args.has_header)
    t_len = series_length(args.target, args.target_column, args.delimiter, args.has_header)
    if q_len > t_len:
        raise ValidationError(f"query length {q_len} exceeds target length {t_len}")
    if q_len < 2:
        raise ValidationError("query length must be at least 2")
    return f"OK: query length {q_len}, target length {t_len}; use m={q_len}."


def validate_profile(args: argparse.Namespace) -> str:
    if args.multidim:
        n_dims, n_timestamps = multidim_shape(args.series)
        if args.window_size > n_timestamps:
            raise ValidationError(
                f"window size {args.window_size} exceeds timestamp count {n_timestamps}"
            )
        return (
            f"OK: multidimensional shape ({n_dims}, {n_timestamps}); "
            f"STUMPY expects (n_dimensions, n_timestamps), m={args.window_size}."
        )

    n = series_length(args.series, args.column, args.delimiter, args.has_header)
    if args.window_size > n:
        raise ValidationError(f"window size {args.window_size} exceeds series length {n}")
    if args.window_size < 2:
        raise ValidationError("window size must be at least 2")
    return f"OK: 1D series length {n}, m={args.window_size}."


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate inputs for STUMPY match/mass/stump/mstump workflows.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    match = subparsers.add_parser("match", help="Validate query and target series")
    match.add_argument("query", type=Path)
    match.add_argument("target", type=Path)
    match.add_argument("--query-column", type=int, default=0)
    match.add_argument("--target-column", type=int, default=0)
    match.add_argument("--delimiter", default=",")
    match.add_argument("--has-header", action="store_true")

    profile = subparsers.add_parser("profile", help="Validate one series and a matrix-profile window")
    profile.add_argument("series", type=Path)
    profile.add_argument("--window-size", "-m", type=int, required=True)
    profile.add_argument("--column", type=int, default=0)
    profile.add_argument("--delimiter", default=",")
    profile.add_argument("--has-header", action="store_true")
    profile.add_argument("--multidim", action="store_true", help="Validate .npy shape for mstump/mmotifs")

    args = parser.parse_args()
    try:
        if args.mode == "match":
            print(validate_match(args))
        else:
            print(validate_profile(args))
    except ValidationError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
