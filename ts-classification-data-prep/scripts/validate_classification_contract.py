#!/usr/bin/env python3
"""Validate a minimal time-series classification data contract."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import struct
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class ValidationError(Exception):
    """Raised when an input violates the classification contract."""


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


def read_csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as handle:
        return [
            [cell.strip() for cell in row]
            for row in csv.reader(handle)
            if any(cell.strip() for cell in row)
        ]


def read_vector(path: Path, *, has_header: bool, column: str | None) -> tuple[list[str], bool]:
    if path.suffix == ".npy":
        shape = read_npy_shape(path)
        if len(shape) == 1:
            return [""] * shape[0], False
        if len(shape) == 2 and 1 in shape:
            return [""] * max(shape), False
        raise ValidationError(f"{path} vector must have shape (n,), (n, 1), or (1, n)")

    rows = read_csv_rows(path)
    if not rows:
        raise ValidationError(f"{path} contains no values")

    if column:
        header = rows[0]
        if column not in header:
            raise ValidationError(f"{path} does not contain column {column!r}")
        index = header.index(column)
        values = [row[index] for row in rows[1:] if len(row) > index and row[index]]
    else:
        data = rows[1:] if has_header else rows
        if len(data) == 1 and len(data[0]) > 1:
            values = [cell for cell in data[0] if cell]
        else:
            if any(len(row) != 1 for row in data):
                raise ValidationError(f"{path} must contain one value per row")
            values = [row[0] for row in data if row[0]]

    if not values:
        raise ValidationError(f"{path} contains no usable values")
    return values, True


def validate(
    x_path: Path,
    y_path: Path,
    splits_path: Path | None,
    *,
    label_col: str | None,
    split_col: str | None,
    has_header: bool,
) -> dict[str, Any]:
    x_shape = read_npy_shape(x_path)
    if len(x_shape) not in {2, 3}:
        raise ValidationError(f"X must be 2D or 3D sample-major data; got {x_shape}")
    if x_shape[0] <= 0:
        raise ValidationError("X contains no samples")
    if any(dim <= 0 for dim in x_shape):
        raise ValidationError(f"X has invalid non-positive dimensions: {x_shape}")

    labels, labels_known = read_vector(y_path, has_header=has_header, column=label_col)
    if len(labels) != x_shape[0]:
        raise ValidationError(f"sample mismatch: X has {x_shape[0]} samples; y has {len(labels)} labels")

    result: dict[str, Any] = {
        "status": "ok",
        "x_shape": list(x_shape),
        "n_samples": x_shape[0],
        "label_values_read": labels_known,
    }
    if labels_known:
        result["n_classes"] = len(set(labels))
        result["class_counts"] = dict(sorted(Counter(labels).items()))

    if splits_path:
        splits, splits_known = read_vector(splits_path, has_header=has_header, column=split_col)
        if len(splits) != x_shape[0]:
            raise ValidationError(
                f"sample mismatch: X has {x_shape[0]} samples; splits has {len(splits)} rows"
            )

        result["split_values_read"] = splits_known
        if splits_known:
            result["split_counts"] = dict(sorted(Counter(splits).items()))

        if labels_known and splits_known:
            classes_by_split: dict[str, set[str]] = defaultdict(set)
            for label, split in zip(labels, splits):
                classes_by_split[split].add(label)

            train_names = {"train", "training"}
            train_classes = set().union(
                *(classes for split, classes in classes_by_split.items() if split.lower() in train_names)
            )
            if train_classes and train_classes != set(labels):
                missing = sorted(set(labels) - train_classes)
                raise ValidationError(f"train split is missing classes: {missing}")

            result["classes_by_split"] = {
                split: sorted(classes) for split, classes in sorted(classes_by_split.items())
            }

    return result


def write_demo_npy(path: Path, shape: tuple[int, ...]) -> None:
    metadata = {"descr": "<f8", "fortran_order": False, "shape": shape}
    header = repr(metadata).encode("latin1")
    header += b" " * ((16 - ((10 + len(header) + 1) % 16)) % 16) + b"\n"
    path.write_bytes(b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", len(header)) + header)


def demo() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        x_path = base / "X.npy"
        y_path = base / "y.csv"
        splits_path = base / "splits.csv"
        write_demo_npy(x_path, (4, 12, 2))
        y_path.write_text("label\nnormal\nfault\nnormal\nfault\n", encoding="utf-8")
        splits_path.write_text("split\ntrain\ntrain\nvalidation\ntest\n", encoding="utf-8")
        return validate(
            x_path,
            y_path,
            splits_path,
            label_col="label",
            split_col="split",
            has_header=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate X/y/splits alignment for time-series classification."
    )
    parser.add_argument("x", nargs="?", type=Path, help="Path to sample-major X.npy")
    parser.add_argument("y", nargs="?", type=Path, help="Path to labels .npy, CSV, or TXT")
    parser.add_argument("--splits", type=Path, help="Optional split vector .npy, CSV, or TXT")
    parser.add_argument("--label-col", help="CSV label column name")
    parser.add_argument("--split-col", help="CSV split column name")
    parser.add_argument("--has-header", action="store_true", help="Treat CSV/TXT first row as a header")
    parser.add_argument("--demo", action="store_true", help="Run the built-in self-check")
    args = parser.parse_args()

    try:
        if args.demo:
            result = demo()
        else:
            if not args.x or not args.y:
                parser.error("x and y are required unless --demo is used")
            result = validate(
                args.x,
                args.y,
                args.splits,
                label_col=args.label_col,
                split_col=args.split_col,
                has_header=args.has_header,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
    except ValidationError as exc:
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
