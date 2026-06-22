#!/usr/bin/env python3
"""Validate dl-4-tsc MTS/custom .npy array headers without importing numpy."""

from __future__ import annotations

import argparse
import ast
import struct
from pathlib import Path


class ValidationError(Exception):
    """Raised when an input does not satisfy the dl-4-tsc array contract."""


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


def label_count(path: Path) -> int:
    shape = read_npy_shape(path)
    if len(shape) == 1:
        return shape[0]
    if len(shape) == 2 and 1 in shape:
        return max(shape)
    raise ValidationError(
        f"{path} labels must have shape (n_samples,), (n_samples, 1), or (1, n_samples)"
    )


def validate_x_shape(path: Path, allow_2d: bool) -> tuple[int, ...]:
    shape = read_npy_shape(path)
    allowed_dims = (2, 3) if allow_2d else (3,)
    if len(shape) not in allowed_dims:
        expected = "2D or 3D" if allow_2d else "3D"
        raise ValidationError(f"{path} X must be {expected}; got shape {shape}")
    if any(dim <= 0 for dim in shape):
        raise ValidationError(f"{path} has invalid non-positive dimensions: {shape}")
    return shape


def validate(
    x_train: Path,
    y_train: Path,
    x_test: Path,
    y_test: Path,
    allow_2d: bool,
) -> str:
    train_shape = validate_x_shape(x_train, allow_2d)
    test_shape = validate_x_shape(x_test, allow_2d)

    if len(train_shape) != len(test_shape):
        raise ValidationError(
            f"train/test X dimensionality mismatch: {train_shape} vs {test_shape}"
        )
    if train_shape[1:] != test_shape[1:]:
        raise ValidationError(
            f"train/test timestamps/channels mismatch: {train_shape[1:]} vs {test_shape[1:]}"
        )

    train_labels = label_count(y_train)
    test_labels = label_count(y_test)
    if train_shape[0] != train_labels:
        raise ValidationError(
            f"train label mismatch: X has {train_shape[0]} samples but y has {train_labels}"
        )
    if test_shape[0] != test_labels:
        raise ValidationError(
            f"test label mismatch: X has {test_shape[0]} samples but y has {test_labels}"
        )

    if len(train_shape) == 2:
        contract = "univariate 2D, reshaped by main.py to (samples, timestamps, 1)"
    else:
        contract = "3D dl-4-tsc order (samples, timestamps, variables)"
    return (
        f"OK: train X {train_shape}, test X {test_shape}, "
        f"train labels {train_labels}, test labels {test_labels}; {contract}."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate dl-4-tsc MTS/custom x_train/y_train/x_test/y_test .npy files."
    )
    parser.add_argument("x_train", type=Path)
    parser.add_argument("y_train", type=Path)
    parser.add_argument("x_test", type=Path)
    parser.add_argument("y_test", type=Path)
    parser.add_argument(
        "--allow-2d",
        action="store_true",
        help="Also allow univariate 2D X arrays that main.py can reshape.",
    )
    args = parser.parse_args()

    try:
        print(validate(args.x_train, args.y_train, args.x_test, args.y_test, args.allow_2d))
    except ValidationError as exc:
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
