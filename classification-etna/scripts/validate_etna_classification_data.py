#!/usr/bin/env python3
"""Validate ETNA experimental binary classification data files."""

from __future__ import annotations

import argparse
import ast
import struct
from pathlib import Path


class ValidationError(Exception):
    """Raised when data violates the ETNA classification contract."""


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


def count_npy_labels(path: Path) -> int:
    shape = read_npy_shape(path)
    if len(shape) == 1:
        return shape[0]
    if len(shape) == 2 and 1 in shape:
        return max(shape)
    raise ValidationError(
        f"{path} labels must have shape (n_samples,), (n_samples, 1), or (1, n_samples)"
    )


def parse_label(raw: str, path: Path, line_no: int) -> int:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValidationError(f"{path}:{line_no} label is not numeric: {raw!r}") from exc
    if not value.is_integer():
        raise ValidationError(f"{path}:{line_no} label must be an integer class, got {value}")
    return int(value)


def validate_tsv(path: Path) -> tuple[int, int, set[int]]:
    row_count = 0
    width: int | None = None
    labels: set[int] = set()

    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            cells = line.strip().split()
            if not cells:
                continue
            if len(cells) < 2:
                raise ValidationError(f"{path}:{line_no} must contain label plus at least one value")
            label = parse_label(cells[0], path, line_no)
            labels.add(label)
            current_width = len(cells) - 1
            if width is None:
                width = current_width
            elif current_width != width:
                raise ValidationError(
                    f"{path}:{line_no} has {current_width} values; expected {width}"
                )
            row_count += 1

    if row_count == 0 or width is None:
        raise ValidationError(f"{path} contains no samples")
    return row_count, width, labels


def validate_ucr(train: Path, test: Path) -> str:
    n_train, width_train, train_labels = validate_tsv(train)
    n_test, width_test, test_labels = validate_tsv(test)
    if width_train != width_test:
        raise ValidationError(
            f"train/test timestamp count mismatch: {width_train} vs {width_test}"
        )
    labels = train_labels | test_labels
    if not labels <= {0, 1}:
        raise ValidationError(
            f"labels must be binary 0/1 for ETNA; got {sorted(labels)}. Map UCR -1/1 labels first."
        )
    return (
        f"OK: train {n_train}x{width_train}, test {n_test}x{width_test}; "
        f"binary labels {sorted(labels)}."
    )


def validate_npy(x_train: Path, y_train: Path, x_test: Path, y_test: Path) -> str:
    x_train_shape = read_npy_shape(x_train)
    x_test_shape = read_npy_shape(x_test)
    if len(x_train_shape) != 2 or len(x_test_shape) != 2:
        raise ValidationError(
            f"ETNA fixed-array validation expects 2D X arrays; got {x_train_shape} and {x_test_shape}"
        )
    if x_train_shape[1] != x_test_shape[1]:
        raise ValidationError(
            f"train/test timestamp count mismatch: {x_train_shape[1]} vs {x_test_shape[1]}"
        )
    y_train_len = count_npy_labels(y_train)
    y_test_len = count_npy_labels(y_test)
    if y_train_len != x_train_shape[0]:
        raise ValidationError(
            f"train label mismatch: X has {x_train_shape[0]} samples but y has {y_train_len}"
        )
    if y_test_len != x_test_shape[0]:
        raise ValidationError(
            f"test label mismatch: X has {x_test_shape[0]} samples but y has {y_test_len}"
        )
    return (
        f"OK: train X {x_train_shape}, test X {x_test_shape}, "
        f"train labels {y_train_len}, test labels {y_test_len}. "
        "Ensure label values are binary 0/1 before fitting."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate ETNA experimental binary classification train/test data."
    )
    mode = parser.add_subparsers(dest="mode", required=True)

    ucr = mode.add_parser("ucr", help="Validate UCR-style TSV files with label in first column")
    ucr.add_argument("train", type=Path)
    ucr.add_argument("test", type=Path)

    npy = mode.add_parser("npy", help="Validate fixed-length X/y .npy train/test files")
    npy.add_argument("x_train", type=Path)
    npy.add_argument("y_train", type=Path)
    npy.add_argument("x_test", type=Path)
    npy.add_argument("y_test", type=Path)

    args = parser.parse_args()

    try:
        if args.mode == "ucr":
            print(validate_ucr(args.train, args.test))
        else:
            print(validate_npy(args.x_train, args.y_train, args.x_test, args.y_test))
    except ValidationError as exc:
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
