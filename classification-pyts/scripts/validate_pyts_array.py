#!/usr/bin/env python3
"""Validate pyts classification array headers without importing numpy."""

from __future__ import annotations

import argparse
import ast
import csv
import struct
from pathlib import Path


class ValidationError(Exception):
    """Raised when an input does not satisfy the pyts classification contract."""


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

    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{path} has a non-integer shape {shape}") from exc


def count_text_labels(path: Path, has_header: bool) -> int:
    with path.open(newline="") as handle:
        rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]

    if has_header and rows:
        rows = rows[1:]

    if not rows:
        raise ValidationError(f"{path} contains no labels")

    if len(rows) == 1 and len(rows[0]) > 1:
        return sum(1 for cell in rows[0] if cell.strip())

    if any(len(row) != 1 for row in rows):
        raise ValidationError(
            f"{path} must contain one label per row, or one comma-separated row"
        )

    return len(rows)


def count_labels(path: Path, has_header: bool) -> int:
    if path.suffix == ".npy":
        shape = read_npy_shape(path)
        if len(shape) == 1:
            return shape[0]
        if len(shape) == 2 and 1 in shape:
            return max(shape)
        raise ValidationError(f"{path} labels must have shape (n_samples,), (n_samples, 1), or (1, n_samples)")

    return count_text_labels(path, has_header)


def validate(x_path: Path, y_path: Path, expected_dim: int | None, has_header: bool) -> str:
    x_shape = read_npy_shape(x_path)
    if expected_dim is not None and len(x_shape) != expected_dim:
        raise ValidationError(f"X must be {expected_dim}D; got shape {x_shape}")
    if expected_dim is None and len(x_shape) not in (2, 3):
        raise ValidationError(
            f"X must be 2D univariate or 3D multivariate; got shape {x_shape}"
        )
    if any(dim <= 0 for dim in x_shape):
        raise ValidationError(f"X has invalid non-positive dimensions: {x_shape}")

    y_len = count_labels(y_path, has_header)
    if x_shape[0] != y_len:
        raise ValidationError(
            f"label count mismatch: X has {x_shape[0]} samples but y has {y_len} labels"
        )

    if len(x_shape) == 2:
        contract = "(n_samples, n_timestamps)"
    else:
        contract = "(n_samples, n_features, n_timestamps)"
    return f"OK: X shape {x_shape}; y length {y_len}; pyts expects {contract}."


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate pyts classification X.npy and y labels."
    )
    parser.add_argument("x", type=Path, help="Path to X.npy")
    parser.add_argument("y", type=Path, help="Path to y.npy or CSV/TXT labels")
    parser.add_argument(
        "--expected-dim",
        type=int,
        choices=(2, 3),
        default=None,
        help="Require 2D univariate or 3D multivariate X",
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Treat the first row of CSV/TXT labels as a header",
    )
    args = parser.parse_args()

    try:
        print(validate(args.x, args.y, args.expected_dim, args.has_header))
    except ValidationError as exc:
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
